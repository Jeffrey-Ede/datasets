import numpy as np

import os

from scipy.misc import imread

FILE_DIR = "F:/ARM_scans/"
SAVE_DIR = "H:/"

CROP_SIZE = 96

files = [FILE_DIR + f for f in os.listdir(FILE_DIR) if ".tif" in f]

crops = np.zeros([len(files), CROP_SIZE, CROP_SIZE, 1])

for i, f in enumerate(files):
    if not i%50:
        print(f"Iter {i} of {crops.shape[0]}")

    try:
        crops[i,:,:,0] = imread(f, mode='F')[:CROP_SIZE, :CROP_SIZE]
    except:
        pass

np.save(SAVE_DIR+"96x96_stem_crops.npy", crops)
