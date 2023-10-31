#!/usr/bin/python3

import os, sys, logging

logging.basicConfig(
    level = logging.INFO, 
    format="[%(levelname)s %(asctime)s] %(message)s"
)
from args import handle_args

from experiment import Experiment

# Daniel Weston
# psydw2@nottingham.ac.uk
# OPTIMlab

def main():
    args = handle_args()
    
    p = Experiment(args)
    p.start()

if __name__ == "__main__":
    main()