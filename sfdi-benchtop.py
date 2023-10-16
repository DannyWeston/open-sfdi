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

    img_func = lambda img: img[:, :, 2] # Only keep red channel in images

    p = Experiment(args, img_func=img_func)

    for i in range(args["runs"]):
        p.run(i)

if __name__ == "__main__":
    main()