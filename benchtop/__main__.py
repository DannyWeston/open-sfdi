#!/usr/bin/python3

# Daniel Weston
# psydw2@nottingham.ac.uk
# OPTIMlab

import logging

logging.basicConfig(
    level = logging.INFO, 
    format = "[%(levelname)s] %(message)s"
)

from sfdi.experiment import Experiment

from benchtop.args import handle_args
from benchtop.video import PiCamera, PygameProjector

if __name__ == "__main__":
    args = handle_args()

    p = Experiment( camera=PiCamera(),
                    projector=PygameProjector(1280, 720), 
                    debug=args["debug"]
    )

    # Run the experiment with some parameters
    p.run(  args["refr_index"],
            args["mu_a"], 
            args["mu_sp"], 
            args["runs"], 
            args["fringes"]
    )