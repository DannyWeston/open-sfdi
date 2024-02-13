import os

from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))   # Root of the entire codebase

DATA_DIR = os.path.join(ROOT_DIR, 'data')               # IO data location

RESULTS_DIR = os.path.join(DATA_DIR, "results")         # Directory for results to be written to

FRINGES_DIR = os.path.join(DATA_DIR, "fringes")         # Fringes are to be used from this directory

CALIBRATION_DIR = os.path.join(DATA_DIR, "calibration") # Location where calibration data is dumped

def make_structure():
    global ROOT_DIR
    global DATA_DIR
    global RESULTS_DIR
    global FRINGES_DIR
    global CALIBRATION_DIR
    
    Path(ROOT_DIR).mkdir(exist_ok=True)
    Path(DATA_DIR).mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    Path(FRINGES_DIR).mkdir(exist_ok=True)
    Path(CALIBRATION_DIR).mkdir(exist_ok=True)

def update_root(new_root, mkdirs=True):
    global ROOT_DIR
    global DATA_DIR
    global RESULTS_DIR
    global FRINGES_DIR
    global CALIBRATION_DIR
    
    ROOT_DIR = new_root
    DATA_DIR = os.path.join(ROOT_DIR, 'data')               # IO data location
    RESULTS_DIR = os.path.join(DATA_DIR, "results")         # Directory for results to be written to
    FRINGES_DIR = os.path.join(DATA_DIR, "fringes")         # Fringes are to be used from this directory
    CALIBRATION_DIR = os.path.join(DATA_DIR, "calibration") # Location where calibration data is dumped
    
    if mkdirs: make_structure()
    
make_structure()