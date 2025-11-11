from .. import image, utils

def ShowPhasemap(phasemap, name='Phasemap', size=None):
    with utils.ProcessingContext.UseGPU(False):
        image.Show(image.Normalise(phasemap), name, size=size)