import os
from scipy.misc import imread
import numpy as np


DATA_DIR = "//Desktop-sa1evjv/h/ARM_scans/"
SAVE_FILE = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/misc/shapes.npy"


files = [DATA_DIR+f for f in os.listdir(DATA_DIR) if f[-4:] == ".tif"]

shapes = []
for i, f in enumerate(files):

    if not i%50:
        print(f"Iter {i} of {len(files)}")
    print(imread(f, mode="F").shape)
    shapes.append(imread(f, mode="F").shape[:2])

shapes = [imread(f, mode="F").shape[:2] for f in files]
shapes = np.array(shapes)

np.save(SAVE_FILE, shapes)
    