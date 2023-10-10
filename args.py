import argparse

def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--proj_imgs", help="Path to the images for the projector to display")
    #parser.add_argument("--proj_imgs", help="Path to the images for the projector to display", required=True) # Required

    return vars(parser.parse_args())