import numpy as np

from . import image, utils, colour, phase
from .devices import projector, camera

def phase_to_coord(resolution, cam_coords, phasemap, stripe_count, use_x=True, bilinear=True):
    xp = utils.ProcessingContext().xp

    w, h = resolution

    proj_coords = xp.empty((len(cam_coords)))

    period = (w if use_x else h) / stripe_count

    for i in range(len(cam_coords)):
        if bilinear:
            phi = image.CoordBilinearInterp(phasemap, cam_coords[i]) # Interp phasemap using camera POI coords
        else:
            coords = cam_coords[i].astype(xp.uint16)
            phi = phasemap[coords[1], coords[0]]

        proj_coords[i] = (phi / (xp.pi * 2.0)) * period

    return proj_coords

def sinusoidal_pattern(resolution, num_stripes=[32.0], phases=[0.0], rotations=[0.0], intensities=[1.0]) -> image.Image:
    '''
        resolution: (width, height) in integer pixels\n
        num_stripes: list[float] for total number of oscillations per channel\n
        phases: list[float] in radians for channel phase shifts\n
        rotations: list[float] in radians for channel fringe orientations\n
    '''
    assert len(num_stripes) == len(phases) == len(rotations) == len(intensities)

    xp = utils.ProcessingContext().xp

    w, h = resolution

    c = len(num_stripes)
    
    raw_data = xp.empty(shape=(h, w, len(num_stripes)), dtype=xp.float32)

    xs, ys = xp.meshgrid(
        xp.linspace(0.0, 1.0, num=w, endpoint=False),
        xp.linspace(0.0, 1.0, num=h, endpoint=False),
        indexing='xy'
    )

    for i in range(c)[::-1]:
        pixels = (xp.cos(rotations[i]) * xs) - (xp.sin(rotations[i]) * ys)

        # I(x, y) = cos(2 * pi * f * x - phi)
        fringes = xp.cos((pixels * 2.0 * xp.pi * num_stripes[i]) + phases[i], dtype=xp.float32)

        # Normalise fringes from [-1..1] to [0..1]
        # Use BGR method as mostly in OpenCV land...
        raw_data[..., c - i - 1] = intensities[i] * ((fringes + 1.0) / 2.0)

    if c == 1: raw_data = xp.squeeze(raw_data)

    return image.Image(data=raw_data)

def gather_gamma_imgs(camera: camera.Camera, projector: projector.Projector, intensities):
    xp = utils.ProcessingContext().xp

    captured_imgs = []

    for intensity in intensities:
        project_img = xp.ones(projector.shape, dtype=xp.float32) * intensity

        # TODO: Callbacks
        projector.display(project_img)

        captured_imgs.append(camera._capture().raw_data)

    return xp.asarray(captured_imgs)

    # Find first observable change of values for averages (left and right sides) i.e >= delta
    # s, f = DetectableIndices(averages, delta)

    # vis_averages = averages[s:f+1]
    # vis_intensities = intensities[s:f+1]

    # coeffs = xp.polyfit(vis_averages, vis_intensities, order)
    # visible = intensities[s:f+1]

    # return coeffs, visible

class StereoFringeProjection:
    def __init__(self):
        pass

    def gather_imgs(self, camera: camera.Camera, projector: projector.Projector, phase_counts, stripe_counts, rotation, reverse=False, gamma_corrector:colour.GammaCorrector=None, out=None):
        ''' Captures and returns images using correct context '''
        assert(len(stripe_counts) == len(phase_counts))

        # Get correct context
        xp = utils.ProcessingContext().xp

        pattern = None

        cam_w, cam_h = camera.get_settings().resolution
        proj_w, proj_h = projector.get_settings().resolution

        if out is None:
            out = xp.empty(shape=(sum(phase_counts), cam_h, cam_w), dtype=xp.float32)

        l = 0

        for (stripe_count, phase_count) in zip(stripe_counts, phase_counts):
            phases = (xp.arange(phase_count) * 2.0 * np.pi) / phase_count

            for j, phase in enumerate(phases):
                index = l
                index += ((phase_count-j) % phase_count) if reverse else j

                # Generate fringes and display them on the projector
                pattern = image.make_fringe_pattern((proj_h, proj_w), stripe_count, phase, rotation)
                projector.project(pattern)

                # Capture an image using the camera, and ensure to load it to correct context
                out[index] = xp.asarray(camera.capture().raw_data)

                # Apply gamma correction to raw data if provided
                if gamma_corrector: gamma_corrector.apply(out[index])

            l += phase_count

        return out

    def calculate_phasemap(self, imgs, shifter: phase.Shifter, unwrapper: phase.Unwrapper):
        xp = utils.ProcessingContext().xp
        
        assert (imgs.shape[0] == sum(shifter.phase_counts))

        shifted = xp.empty(shape=(len(unwrapper.stripe_count), *imgs[0].shape), dtype=xp.float32)

        ac_img = None
        dc_img = None

        completed = 0
        for i, N in enumerate(shifter.phase_counts):
            shifted[i], ac, dc = shifter.shift(imgs[completed:completed+N])

            if i == 0:
                ac_img = ac
                dc_img = dc

            completed += N

        # Calculate unwrapped phase maps
        return unwrapper.Unwrap(shifted), ac_img, dc_img

    def reconstruct(self, phasemap, camera: camera.Camera, projector: projector.Projector, stripe_count, use_x=True):
        """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """
        xp = utils.ProcessingContext().xp

        # TODO: Check workingResolution with resolution being used
        # So correct scaling can be applied
        c_w, c_h = camera.get_settings().resolution
        p_w, p_h = projector.get_settings().resolution
        camY, camX = xp.mgrid[:c_h, :c_w].astype(xp.float32)

        period = (p_w if use_x else p_h) / stripe_count
        projCoords = (phasemap / (xp.pi * 2.0)) * period

        cam_char = camera.get_char()
        proj_char = projector.get_char()

        return self.__triangulate(
            xp.asarray(cam_char.projection_mat),
            xp.asarray(proj_char.projection_mat),
            camX, camY, projCoords, use_x
        )

    def __triangulate(self, cam_mat, proj_mat, cam_x, cam_y, proj, vertical=True):
        xp = utils.ProcessingContext().xp

        a1 = cam_mat[0, 0] - cam_x * cam_mat[2, 0]
        a2 = cam_mat[0, 1] - cam_x * cam_mat[2, 1]
        a3 = cam_mat[0, 2] - cam_x * cam_mat[2, 2]

        a4 = cam_mat[1, 0] - cam_y * cam_mat[2, 0]
        a5 = cam_mat[1, 1] - cam_y * cam_mat[2, 1]
        a6 = cam_mat[1, 2] - cam_y * cam_mat[2, 2]

        b1 = cam_x * cam_mat[2, 3] - cam_mat[0, 3]
        b2 = cam_y * cam_mat[2, 3] - cam_mat[1, 3]

        if vertical:
            a7 = proj_mat[0, 0] - proj * proj_mat[2, 0]
            a8 = proj_mat[0, 1] - proj * proj_mat[2, 1]
            a9 = proj_mat[0, 2] - proj * proj_mat[2, 2]

            b3 = proj * proj_mat[2, 3] - proj_mat[0, 3]

        else:
            a7 = proj_mat[1, 0] - proj * proj_mat[2, 0]
            a8 = proj_mat[1, 1] - proj * proj_mat[2, 1]
            a9 = proj_mat[1, 2] - proj * proj_mat[2, 2]

            b3 = proj * proj_mat[2, 3] - proj_mat[1, 3]

        D = -a3 * a5 * a7 + a2 * a6 * a7 + a3 * a4 * a8 - a1 * a6 * a8 - a2 * a4 * a9 + a1 * a5 * a9
        worldX = (1.0 / D) * ((a5 * a9 - a6 * a8) * b1 + (a3 * a8 - a2 * a9) * b2 + (a2 * a6 - a3 * a5) * b3)
        worldY = (1.0 / D) * ((a6 * a7 - a4 * a9) * b1 + (a1 * a9 - a3 * a7) * b2 + (a3 * a4 - a1 * a6) * b3)
        worldZ = (1.0 / D) * ((a4 * a8 - a5 * a7) * b1 + (a2 * a7 - a1 * a8) * b2 + (a1 * a5 - a2 * a4) * b3)

        points = xp.dstack([worldX, worldY, worldZ])

        h, w, *_ = cam_x.shape

        return points.reshape((h * w, 3))