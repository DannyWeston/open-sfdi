import numpy as np

from numpy.polynomial import polynomial as P
from abc import ABC, abstractmethod

from .devices import camera, projector
from .phase import unwrap, shift, ShowPhasemap

def FringeProject(cam: camera.Camera, proj: projector.FringeProjector, sf, phases) -> np.ndarray:
    proj.numStripes = sf
    N = len(phases)

    imgs = np.empty(shape=(N, *cam.config.shape))

    for i in range(N):
        proj.phase = phases[i]
        proj.Display()
        imgs[i] = cam.Capture().rawData
        # imgs[i] = image.add_gaussian(imgs[i], sigma=0.1)

    return imgs

class ReconstructionResult:
    def __init__(self):
        self.time_taken = None

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


# Phase-height Methods

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
            shifted[j] = phi_shift.Shift(imgs)

        # Calculate unwrapped phase maps
        return phi_unwrap.Unwrap(shifted)

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


# Stereo Methods

class StereoReconstructor(IReconstructor):
    def __init__(self):
        self.m_AlignToCamera = True

    def GatherPhasemap(self, cam: camera.Camera, proj: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap):
        # TODO: Add some flag to save the images whilst gathering?
        sfs = unwrapper.GetFringeCount()
        phases = shifter.GetPhases()

        shifted = np.empty((len(sfs), *cam.config.shape), dtype=np.float32)
        
        # Calculate the wrapped phase maps
        for j, sf in enumerate(sfs):
            imgs = FringeProject(cam, proj, sf, phases)
            shifted[j] = shifter.Shift(imgs)

        # Calculate unwrapped phase maps
        return unwrapper.Unwrap(shifted)

    def __Triangulate(self, camProjMat, projProjmat, camX, camY, proj, useX=True):
        # TODO: Conditional branches are bad for CPU time
        # TODO: Move to separate functions instead of parameter controlled

        a1 = camProjMat[0, 0] - camX * camProjMat[2, 0]
        a2 = camProjMat[0, 1] - camX * camProjMat[2, 1]
        a3 = camProjMat[0, 2] - camX * camProjMat[2, 2]

        a4 = camProjMat[1, 0] - camY * camProjMat[2, 0]
        a5 = camProjMat[1, 1] - camY * camProjMat[2, 1]
        a6 = camProjMat[1, 2] - camY * camProjMat[2, 2]

        b1 = camX * camProjMat[2, 3] - camProjMat[0, 3]
        b2 = camY * camProjMat[2, 3] - camProjMat[1, 3]

        if useX:
            a7 = projProjmat[0, 0] - proj * projProjmat[2, 0]
            a8 = projProjmat[0, 1] - proj * projProjmat[2, 1]
            a9 = projProjmat[0, 2] - proj * projProjmat[2, 2]

            b3 = proj * projProjmat[2, 3] - projProjmat[0, 3]

        else:
            a7 = projProjmat[1, 0] - proj * projProjmat[2, 0]
            a8 = projProjmat[1, 1] - proj * projProjmat[2, 1]
            a9 = projProjmat[1, 2] - proj * projProjmat[2, 2]

            b3 = proj * projProjmat[2, 3] - projProjmat[1, 3]

        D = -a3 * a5 * a7 + a2 * a6 * a7 + a3 * a4 * a8 - a1 * a6 * a8 - a2 * a4 * a9 + a1 * a5 * a9
        worldX = (1.0 / D) * ((a5 * a9 - a6 * a8) * b1 + (a3 * a8 - a2 * a9) * b2 + (a2 * a6 - a3 * a5) * b3)
        worldY = (1.0 / D) * ((a6 * a7 - a4 * a9) * b1 + (a1 * a9 - a3 * a7) * b2 + (a3 * a4 - a1 * a6) * b3)
        worldZ = (1.0 / D) * ((a4 * a8 - a5 * a7) * b1 + (a2 * a7 - a1 * a8) * b2 + (a1 * a5 - a2 * a4) * b3)

        points = np.dstack([worldX, worldY, worldZ])

        return points.reshape(-1, 3)

    def Reconstruct(self, cam: camera.Camera, proj: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, stripeCount, useX=True):
        """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """

        if cam.visionConfig is None:
            raise Exception("Camera vision is not characterised: you must use a characterised camera for reconstruction")
        
        if proj.visionConfig is None:
            raise Exception("Projector vision is not characterised: you must use a characterised projector for reconstruction")

        # Gather a phasemap by using the camera and projector
        phasemap = self.GatherPhasemap(cam, proj, shifter, unwrapper)

        # TODO: Check workingResolution with resolution being used
        # So correct scaling can be applied
        camHeight, camWidth = cam.config.resolution
        projHeight, projWidth = proj.config.resolution

        camX, camY = np.meshgrid(np.arange(camWidth, dtype=np.float32), np.arange(camHeight, dtype=np.float32))

        phi = phasemap / (2.0 * np.pi * stripeCount)
        phi *= (projWidth if useX else projHeight)

        pc = self.__Triangulate(cam.visionConfig.projectionMat, proj.visionConfig.projectionMat, camX, camY, phi, useX)

        # Find the indices of any NaNs and filter them out
        validPoints = ~np.isnan(pc).any(axis=1)
        pc = pc[validPoints]

        # Remove any NaNs for masked values earlier
        return pc, validPoints

    def AlignCloudToCB(self, pc: np.ndarray, cam: camera.Camera, centre=False):
        if cam.visionConfig is None:
            raise Exception("Camera vision is not characterised")

        R = cam.visionConfig.rotation.T
        t = -cam.visionConfig.rotation.T @ cam.visionConfig.translation

        pc = (R @ pc.T).T + t.ravel()

        return pc

    @property
    def alignToCamera(self) -> bool:
        return self.m_AlignToCamera

    @alignToCamera.setter
    def alignToCamera(self, value: bool):
        self.m_AlignToCamera = value