#!/usr/bin/python3

# Daniel Weston
# psydw2@nottingham.ac.uk
# OPTIMlab

import logging
import numpy as np

logging.basicConfig(
    level = logging.INFO, 
    format = "[%(levelname)s] %(message)s"
)

from sfdi.experiment import Experiment

from sfdi.video import CallbackCamera, Projector
from sfdi.args import handle_args

class BlenderProjector(Projector):
    def __init__(self, width, height):
        super().__init__(width, height)
        
    def display(self, img):
        print("Displaying object")

def on_capture():
    img = np.zeros((720, 1280), dtype=np.uint8)
    return img

if __name__ == "__main__":
    args = handle_args()

    p = Experiment( camera=CallbackCamera(on_capture),
                    projector=BlenderProjector(1280, 720),
                    debug=args["debug"]
    )

    # Run the experiment with some parameters
    p.run(  args["refr_index"],
            args["mu_a"],
            args["mu_sp"],
            args["runs"],
    )