import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Root of the whole project

DATA_DIR = os.path.join(ROOT_DIR, 'data') # IO data location

RESULTS_DIR = os.path.join(DATA_DIR, "results") # Directory for results to be written to

FRINGES_DIR = os.path.join(DATA_DIR, "fringes") # Fringes are to be used from this directory