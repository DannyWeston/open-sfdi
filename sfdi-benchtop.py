import os, sys

from args import handle_args

from experiment import Projection

# Daniel Weston
# psydw2@nottingham.ac.uk
# OPTIMlab

def main():
    args = handle_args()

    if not args["proj_imgs"]:
        args["proj_imgs"] = os.getcwd() + '\\.data\\'

    proj_imgs = [f'{args["proj_imgs"]}default{i}.jpg' for i in range(3)]

    #img_func = lambda img: img[:, :, 2] # Only keep red channel in images

    p = Projection(proj_imgs)
    p.run()

if __name__ == "__main__":
    main()