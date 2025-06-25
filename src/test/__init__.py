import os
import sys
import tkinter as tk

from cv2 import ocl

# sys.path mangling to make src directory visible
source_code = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, source_code)

root = tk.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()
print(f"\n\nOpenCL supported: {ocl.haveOpenCL()}")

# # Update ROOT_DIR to use test data directory (don't want to place test stuff in production)
# from opensfdi.definitions import ROOT_DIR, update_root
# update_root(Path(os.path.join(os.path.dirname(__file__))))