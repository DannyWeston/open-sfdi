import unittest
import os
import sys

# sys.path mangling to make src directory visible
source_code = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, source_code)

# Update ROOT_DIR to use test data directory (don't want to place test stuff in production)
from opensfdi.definitions import ROOT_DIR, update_root
update_root(os.path.join(os.path.dirname(__file__)))