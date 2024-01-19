import unittest
import os

# Update definitions directory to use test directory
from sfdi.definitions import ROOT_DIR
ROOT_DIR = os.path.join(ROOT_DIR, 'test')
print(f'ROOT_DIR definition set to "{ROOT_DIR}" for tests')