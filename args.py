import argparse, os

def handle_args():
    parser = argparse.ArgumentParser()

    #parser.add_argument("--proj_imgs", help="Path to the images for the projector to display")
    parser.add_argument("--proj_imgs", nargs='+', type=str,
        default=[f'{os.getcwd()}/data/test_a{i + 1}.jpg' for i in range(3)] + [f'{os.getcwd()}/data/test_b{i + 1}.jpg' for i in range(3)],
        help="Path to the three images for the projector to display")

    parser.add_argument("--runs", type=int, default=1,
        help="How many times to collect a sample for each projection image")

    parser.add_argument("--camera", type=str, default='/dev/video0',
        help="Path to the camera device to use")

    parser.add_argument("--debug", type=bool, default=False,
        help="Toggle debug mode")
    
    parser.add_argument("--interactive", type=bool, default=False,
        help="Toggle interactive console mode (will ignore runs flag)")
    
    parser.add_argument("--refr_index", type=float, default=1.43,
        help="Refractive index to use")
    
    parser.add_argument("--output_dir", type=float, default='/home/admin/OPTIMlab-benchtop-sfdi/results',
        help="Output directory to place results from an experiment (will create last directory if not present)")
    
    parser.add_argument("--mu_a", type=float, default=0.018,
        help="TODO: Write help")
    
    parser.add_argument("--mu_sp", type=float, default=0.77,
        help="TODO: Write help")

    args = vars(parser.parse_args())

    return args