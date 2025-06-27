import numpy as np

from numpy.polynomial import polynomial as P
from abc import ABC, abstractmethod

from .devices import camera, projector, board
from .phase import unwrap, shift


class ProfilRegistry:
    _registry = {}

    @classmethod
    def register(cls, clazz):
        cls._registry[clazz.__name__] = clazz
        return clazz

    @classmethod
    def get_class(cls, name):
        return cls._registry[name]

class IReconstructor(ABC):
    @abstractmethod
    def __init__(self):
        # Do I need to add metadata here? Not sure
        raise NotImplementedError

    @abstractmethod
    def Reconstruct(self, camera: camera.Camera, projector: projector.FringeProjector, phi_shift: shift.PhaseShift, phi_unwrap: unwrap.PhaseUnwrap, num_stripes):
        raise NotImplementedError
    
    @abstractmethod
    def GatherPhasemap(self, camera: camera.Camera, projector: projector.FringeProjector, phi_shift: shift.PhaseShift, phi_unwrap: unwrap.PhaseUnwrap):
        raise NotImplementedError
    
    def _gather_dc_img(self, camera: camera.Camera, projector: projector.FringeProjector):
        # Gather a single full intensity DC img
        return FringeProject(camera, projector, 0.0, [0.0])[0]

class PhaseHeightReconstructor(IReconstructor):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def Reconstruct(self, camera: camera.Camera, projector: projector.FringeProjector, phi_shift: shift.PhaseShift, phi_unwrap: unwrap.PhaseUnwrap, num_stripes):
        raise NotImplementedError
    
    def GatherPhasemap(self, camera: camera.Camera, projector: projector.FringeProjector, phi_unwrap: unwrap.PhaseUnwrap, phi_shift: shift.PhaseShift):
        # TODO: Add some flag to save the images whilst gathering?
        sfs = phi_unwrap.GetFringeCount()
        phases = phi_shift.GetPhases()

        shifted = np.empty((len(sfs), *camera.shape), dtype=np.float32)
        
        # Calculate the wrapped phase maps
        for j, sf in enumerate(sfs):
            imgs = FringeProject(camera, projector, sf, phases)
            shifted[j] = phi_shift.shift(imgs)

        # Calculate unwrapped phase maps
        return phi_unwrap.unwrap(shifted)

@ProfilRegistry.register
class PolynomialProfil(PhaseHeightReconstructor):
    def __init__(self, polydata):
        self.polydata = polydata

    def Reconstruct(self, camera: camera.Camera, projector: projector.FringeProjector, phi_shift: shift.PhaseShift, phi_unwrap: unwrap.PhaseUnwrap, num_stripes):
        # """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """

        # raw_imgs = np.array([img.data for img in imgs])
        # grey_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in raw_imgs])
        
        # # Calculate a mask to use for segmentation
        # dc_img = image.to_int8(image.dc_imgs(grey_imgs))
        # ret, mask = cv2.threshold(dc_img, 16, 255, cv2.THRESH_BINARY)
        # image.show_image(image.RawImage(dc_img))

        # # Apply mask to imgs
        # grey_imgs = [cv2.bitwise_and(img, img, mask=mask) for img in grey_imgs]
        # # contours, hierarchy = cv2.findContours(image=thresh_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # # Make sure to check if horizontal or vertical ! (assumes horizontal by default)
        # if unwrapper.phase_count == 1:
        #     shifted = shifter.shift(grey_imgs)
        # else:
        #     l = shifter.phase_count
        #     shifted = [shifter.shift(grey_imgs[i * l::l]) for i in range(unwrapper.phase_count)]

        # phasemap = unwrapper.unwrap(shifted)
        # phase.show_phasemap(phasemap)    

        print(self.polydata.shape)

        # Obtain phase difference
        phase_diff = meas_phasemap - ref_phasemap

        # Apply calibrated polynomial values to each pixel of the phase difference
        h, w = phase_diff.shape
        heightmap = np.zeros_like(phase_diff)

        for y in range(h):
            for x in range(w):
                heightmap[y, x] = P.polyval(phase_diff[y, x], self.polydata[:, y, x])

        return heightmap

class StereoReconstructor(IReconstructor):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    def Reconstruct(self, cam: camera.Camera, proj: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, stripeCount):
        if not isinstance(cam.config, camera.CalibratedCameraConfig):
            raise Exception("You must use a characterised camera for reconstruction")
        
        if not isinstance(proj.config, projector.CalibratedProjectorConfig):
            raise Exception("You must use a characterised projector for reconstruction")
        
        return None

    def GatherPhasemap(self, cam: camera.Camera, proj: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap):
        # TODO: Add some flag to save the images whilst gathering?
        sfs = unwrapper.GetFringeCount()
        phases = shifter.GetPhases()

        shifted = np.empty((len(sfs), *cam.config.shape), dtype=np.float32)
        
        # Calculate the wrapped phase maps
        for j, sf in enumerate(sfs):
            imgs = FringeProject(cam, proj, sf, phases)
            shifted[j] = shifter.shift(imgs)

        # Calculate unwrapped phase maps
        return unwrapper.unwrap(shifted)

@ProfilRegistry.register
class StereoProfil(StereoReconstructor):
    def __init__(self):
        pass

    def __Triangulate(self, camProjMat, projProjmat, camX, camY, projX):
        a1 = camProjMat[0, 0] - camX * camProjMat[2, 0]
        a2 = camProjMat[0, 1] - camX * camProjMat[2, 1]
        a3 = camProjMat[0, 2] - camX * camProjMat[2, 2]
        a4 = camProjMat[1, 0] - camY * camProjMat[2, 0]
        a5 = camProjMat[1, 1] - camY * camProjMat[2, 1]
        a6 = camProjMat[1, 2] - camY * camProjMat[2, 2]
        a7 = projProjmat[0, 0] - projX * projProjmat[2, 0]
        a8 = projProjmat[0, 1] - projX * projProjmat[2, 1]
        a9 = projProjmat[0, 2] - projX * projProjmat[2, 2]

        b1 = camX * camProjMat[2, 3] - camProjMat[0, 3]
        b2 = camY * camProjMat[2, 3] - camProjMat[1, 3]
        b3 = projX * projProjmat[2, 3] - projProjmat[0, 3]

        D = -a3 * a5 * a7 + a2 * a6 * a7 + a3 * a4 * a8 - a1 * a6 * a8 - a2 * a4 * a9 + a1 * a5 * a9
        x_w = (1. / D) * ((a5 * a9 - a6 * a8) * b1 + (a3 * a8 - a2 * a9) * b2 + (a2 * a6 - a3 * a5) * b3)
        y_w = (1. / D) * ((a6 * a7 - a4 * a9) * b1 + (a1 * a9 - a3 * a7) * b2 + (a3 * a4 - a1 * a6) * b3)
        z_w = (1. / D) * ((a4 * a8 - a5 * a7) * b1 + (a2 * a7 - a1 * a8) * b2 + (a1 * a5 - a2 * a4) * b3)

        points = np.dstack([x_w, y_w, z_w])

        return points.reshape(-1, 3)

    def __Triangulate2(self, camProjMat, projProjMat, camX, camY, projX):
        h, w = camX.shape

        A = np.empty((h, w, 3, 3))
        b = np.empty((h, w, 3, 1))

        A[:, :, 0, 0] = camProjMat[0, 0] - camProjMat[2, 0] * camX
        A[:, :, 0, 1] = camProjMat[0, 1] - camProjMat[2, 1] * camX
        A[:, :, 0, 2] = camProjMat[0, 2] - camProjMat[2, 2] * camX

        A[:, :, 1, 0] = camProjMat[1, 0] - camProjMat[2, 0] * camY
        A[:, :, 1, 1] = camProjMat[1, 1] - camProjMat[2, 1] * camY
        A[:, :, 1, 2] = camProjMat[1, 2] - camProjMat[2, 2] * camY

        A[:, :, 2, 0] = projProjMat[0, 0] - projProjMat[2, 0] * projX
        A[:, :, 2, 1] = projProjMat[0, 1] - projProjMat[2, 1] * projX
        A[:, :, 2, 2] = projProjMat[0, 2] - projProjMat[2, 2] * projX

        b[:, :, 0, 0] = camProjMat[0, 3] - camProjMat[2, 3] * camX
        b[:, :, 1, 0] = camProjMat[1, 3] - camProjMat[2, 3] * camY
        b[:, :, 2, 0] = projProjMat[0, 3] - projProjMat[2, 3] * projX

        pc = -np.linalg.solve(A, b)

        return pc.reshape(-1, 3)

    def Reconstruct(self, cam: camera.Camera, proj: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, stripeCount):
        """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """

        super().Reconstruct(cam, proj, shifter, unwrapper, stripeCount)

        camConfig = cam.config
        if not isinstance(camConfig, camera.CalibratedCameraConfig):
            raise Exception("Camera must be characterised in order to be used for reconstruction")
        
        projConfig = proj.config
        if not isinstance(projConfig, projector.CalibratedProjectorConfig):
            raise Exception("Projector must be characterised in order to be used for reconstruction")

        # Gather a phasemap by using the camera and projector
        phasemap = self.GatherPhasemap(cam, proj, shifter, unwrapper)

        # TODO: Check workingResolution with resolution being used
        # So correct scaling can be applied
        camHeight, camWidth = camConfig.resolution
        projHeight, projWidth = projConfig.resolution

        camX, camY = np.meshgrid(np.arange(camWidth, dtype=np.float32), np.arange(camHeight, dtype=np.float32))
        projX = (phasemap * projHeight) / (2.0 * np.pi * stripeCount)

        pc = self.__Triangulate(camConfig.visionConfig.projectionMat, projConfig.visionConfig.projectionMat, 
            camX, camY, projX)
        
        # pc = self.__Triangulate2(camConfig.visionConfig.projectionMat, projConfig.visionConfig.projectionMat, 
        #     camX, camY, projX)

        # Find the indices of any NaNs
        validPoints = ~np.isnan(pc).any(axis=1)

        # Remove any NaNs for masked values earlier
        return pc[validPoints], validPoints
    
def FringeProject(cam: camera.Camera, proj: projector.FringeProjector, sf, phases) -> np.ndarray:
    proj.numStripes = sf
    N = len(phases)

    imgs = np.empty(shape=(N, *cam.config.shape))

    for i in range(N):
        proj.phase = phases[i]
        proj.Display()
        imgs[i] = cam.Capture().raw_data
        # imgs[i] = image.add_gaussian(imgs[i], sigma=0.1)

    return imgs
