#!/usr/bin/python3

# Daniel Weston
# psydw2@nottingham.ac.uk
# OPTIMlab

import os, sys, logging

logging.basicConfig(
    level = logging.INFO, 
    format="[%(levelname)s %(asctime)s] %(message)s"
)

from sfdi.measurement.experiment import Experiment
from sfdi.measurement.args import handle_args

def main():
    args = handle_args()

    p = Experiment()
    p.run(args["fringes"], args["refr_index"], 
          args["mu_a"], args["mu_sp"], args["runs"])

if __name__ == "__main__":
    main()