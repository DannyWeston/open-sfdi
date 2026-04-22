import numpy as np

from . import image, utils, devices, colour, phase

def phase_to_coord(resolution, cam_coords, phasemap, stripe_count, use_x=True, bilinear=True):
    xp = utils.ProcessingContext().xp

    w, h = resolution

    projCoords = xp.empty((len(cam_coords)))

    period = (w if use_x else h) / stripe_count

    for i in range(len(cam_coords)):
        if bilinear:
            phi = image.CoordBilinearInterp(phasemap, cam_coords[i]) # Interp phasemap using camera POI coords
        else:
            coords = cam_coords[i].astype(xp.uint16)
            phi = phasemap[coords[1], coords[0]]

        projCoords[i] = (phi / (xp.pi * 2.0)) * period

    return projCoords

def sinusoidal_pattern(resolution, num_stripes, phase=0.0, rotation=0.0) -> image.Image:
    '''
        resolution: (width, height) in integer pixels\n
        num_stripes: float for total number of oscillations\n
        phase: float in radians for signal phase shift\n
        rotation: float in radians for orientation of fringes\n
    '''

    w, h = resolution

    xs, ys = np.meshgrid(
        np.linspace(0.0, 1.0, num=w),
        np.linspace(0.0, 1.0, num=h)
    )

    pixels =  (np.cos(rotation) * xs) - (np.sin(rotation) * ys)

    # I(x, y) = cos(2 * pi * f * x - phi)
    fringes = np.sin((pixels * 2.0 * np.pi * num_stripes) + phase, dtype=np.float32)

    # Normalise fringes from [-1..1] to [0..1]
    return image.Image(data=(fringes + 1.0) / 2.0)

def bgr_sinusoidal_pattern(resolution, num_stripes, phases, rotations, channels=(1.0, 1.0, 1.0)):
    r = channels[2] * sinusoidal_pattern(resolution, num_stripes[2], phases[2], rotations[2]).raw_data
    g = channels[1] * sinusoidal_pattern(resolution, num_stripes[1], phases[1], rotations[1]).raw_data
    b = channels[0] * sinusoidal_pattern(resolution, num_stripes[0], phases[0], rotations[0]).raw_data

    return image.Image(data=np.dstack([r, g, b]))

class StereoFringeProjection:
    def __init__(self):
        pass

    def gather_imgs(self, camera: devices.BaseCamera, projector: devices.BaseProjector, phase_counts, stripe_counts, rotation, reverse=False, gamma_corrector:colour.GammaCorrector=None, out=None):
        ''' Captures and returns images using correct context '''
        assert(len(stripe_counts) == len(phase_counts))

        # Get correct context
        xp = utils.ProcessingContext().xp

        pattern = None

        if out is None:
            out = xp.empty(shape=(sum(phase_counts), *camera.shape), dtype=xp.float32)

        l = 0

        for (stripe_count, phase_count) in zip(stripe_counts, phase_counts):
            phases = (xp.arange(phase_count) * 2.0 * np.pi) / phase_count

            for j, phase in enumerate(phases):
                index = l
                index += ((phase_count-j) % phase_count) if reverse else j

                # Generate fringes and display them on the projector
                pattern = image.make_fringe_pattern(projector.resolution, stripe_count, phase, rotation)
                projector.display(pattern)

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

    def reconstruct(self, phasemap, camera: devices.BaseCamera, projector: devices.BaseProjector, stripe_count, use_x=True):
        """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """

        xp = utils.ProcessingContext().xp

        # TODO: Check workingResolution with resolution being used
        # So correct scaling can be applied
        c_w, c_h = camera.resolution
        p_w, p_h = projector.resolution
        camY, camX = xp.mgrid[:c_h, :c_w].astype(xp.float32)

        period = (p_w if use_x else p_h) / stripe_count
        projCoords = (phasemap / (xp.pi * 2.0)) * period

        return self.__triangulate(
            xp.asarray(camera.char.projection_mat),
            xp.asarray(projector.char.projection_mat),
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

        return points.reshape((*cam_x.shape[:2], 3))

    @property
    def alignToCamera(self) -> bool:
        return self.m_AlignToCamera

    @alignToCamera.setter
    def alignToCamera(self, value: bool):
        self.m_AlignToCamera = value