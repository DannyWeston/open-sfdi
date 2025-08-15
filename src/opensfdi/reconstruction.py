import numpy as np

from abc import ABC, abstractmethod

from .utils import ProcessingContext
from .devices import camera, projector, FringeProject
from .phase import unwrap, shift, ShowPhasemap

from . import image

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
    def GatherPhasemap(self, camera: camera.Camera, projector: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, vertical=True):
        raise NotImplementedError

# Stereo Methods

class StereoReconstructor(IReconstructor):
    def __init__(self):
        self.m_AlignToCamera = True

        self.m_UseChannel = 0

    @property
    def useChannel(self):
        return self.m_UseChannel
    
    @useChannel.setter
    def useChannel(self, value):
        self.m_UseChannel = value

    def GatherPhasemap(self, camera: camera.Camera, projector: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, vertical=True):
        xp = ProcessingContext().xp
        
        # TODO: Add some flag to save the images whilst gathering?
        shifted = xp.empty((len(unwrapper.stripeCount), *camera.config.shape), dtype=xp.float32)

        unwrapper.vertical = vertical
        shifter.vertical = vertical

        for i, (numStripes, N) in enumerate(zip(unwrapper.stripeCount, shifter.phaseCounts)):
            imgs = xp.empty(shape=(N, *camera.config.shape), dtype=xp.float32)

            phases = (xp.arange(N) * 2.0 * xp.pi) / N

            for j in range(N):
                imgs[j] = xp.asarray(FringeProject(camera, projector, numStripes, phases[j]).rawData)

            shifted[i], dcImage = shifter.Shift(imgs)

        # Calculate unwrapped phase maps
        return unwrapper.Unwrap(shifted), dcImage

    def __Triangulate(self, camProjMat, projProjmat, camX, camY, proj, vertical=True):
        # TODO: Conditional branches are bad for CPU time
        # TODO: Move to separate functions instead of parameter controlled
        
        xp = ProcessingContext().xp

        a1 = camProjMat[0, 0] - camX * camProjMat[2, 0]
        a2 = camProjMat[0, 1] - camX * camProjMat[2, 1]
        a3 = camProjMat[0, 2] - camX * camProjMat[2, 2]

        a4 = camProjMat[1, 0] - camY * camProjMat[2, 0]
        a5 = camProjMat[1, 1] - camY * camProjMat[2, 1]
        a6 = camProjMat[1, 2] - camY * camProjMat[2, 2]

        b1 = camX * camProjMat[2, 3] - camProjMat[0, 3]
        b2 = camY * camProjMat[2, 3] - camProjMat[1, 3]

        if vertical:
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

        points = xp.dstack([worldX, worldY, worldZ])

        return points.reshape((*camX.shape[:2], 3))

    def Reconstruct(self, cam: camera.Camera, proj: projector.FringeProjector, shifter: shift.PhaseShift, unwrapper: unwrap.PhaseUnwrap, vertical=True):
        """ Obtain a heightmap using a set of reference and measurement images using the already calibrated values """

        xp = ProcessingContext().xp

        if cam.visionConfig is None:
            raise Exception("Camera vision is not characterised: you must use a characterised camera for reconstruction")
        
        if proj.visionConfig is None:
            raise Exception("Projector vision is not characterised: you must use a characterised projector for reconstruction")

        # Gather a phasemap by using the camera and projector
        phasemap, dcImage = self.GatherPhasemap(cam, proj, shifter, unwrapper, vertical=vertical)

        # TODO: Check workingResolution with resolution being used
        # So correct scaling can be applied
        hCam, wCam = cam.config.resolution
        hProj, wProj = proj.config.resolution

        camY, camX = xp.mgrid[:hCam, :wCam].astype(xp.float32)

        # Select phasemap channel to use - N phasemaps are produced from N channels
        # We can only use one currently
        # TODO: Allow for usage of multiple channels
        if phasemap.ndim == 3: phasemap = phasemap[:, :, self.useChannel]

        projCoords = phasemap / (2.0 * xp.pi * unwrapper.stripeCount[-1])
        projCoords *= (wProj if vertical else hProj)

        pc = self.__Triangulate(
            xp.asarray(cam.visionConfig.projectionMat),
            xp.asarray(proj.visionConfig.projectionMat),
            camX, camY, projCoords, vertical
        )

        # Find the indices of any NaNs and filter them out
        invalidPoints = xp.any(xp.isnan(pc), axis=2)
        validPoints = xp.bitwise_not(invalidPoints)

        pc = pc[validPoints]
        dcImage = dcImage[validPoints] 

        return pc, dcImage, validPoints

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