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

from experiment import Experiment

import utils

def main():
    args = handle_args()

    print(utils.get_local_address())

    p = Experiment(args)
    p.start()

if __name__ == "__main__":
    main()