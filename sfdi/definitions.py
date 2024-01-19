import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))   # Root of the entire codebase

TEST_DIR = os.path.join(ROOT_DIR, 'test')               # Root of the application unittest code

DATA_DIR = os.path.join(ROOT_DIR, 'data')               # IO data location

RESULTS_DIR = os.path.join(DATA_DIR, "results")         # Directory for results to be written to

FRINGES_DIR = os.path.join(DATA_DIR, "fringes")         # Fringes are to be used from this directory

CALIBRATION_DIR = os.path.join(DATA_DIR, "calibration") # Location where calibration data is dumped