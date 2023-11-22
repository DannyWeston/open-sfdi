#!/usr/bin/python3

# Daniel Weston
# psydw2@nottingham.ac.uk
# OPTIMlab

import os, sys, logging

logging.basicConfig(
    level = logging.INFO, 
    format="[%(levelname)s %(asctime)s] %(message)s"
)
from args import handle_args

from sfdi.measurement.experiment import Experiment

from args import handle_args

def main():
    args = handle_args()

    p = Experiment(args)
    p.start()

if __name__ == "__main__":
    main()