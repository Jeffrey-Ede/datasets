"""Downsample to a 96x96 version of a wavefunctions dataset"""

import numpy as np
import os

from skimage.transform import resize

DATA_DIRS = [
    "//Desktop-sa1evjv/h/wavefunctions_refined/",
    "//Desktop-sa1evjv/f/tmp/wavefunctions_refined_fifth/",
    "//Desktop-sa1evjv/f/wavefunctions_single_refined/wavefunctions/"
    ]

SAVE_DIR = "//Desktop-sa1evjv/h/wavefunctions_96x96/"

SAVE_FILES = [
    "wavefunctions_n=3", #24530, 3399, 8395
    "wavefunctions_restricted_n=3", #8002, 1105, 2763
    "wavefunctions_single_n=3", #3861, 964
    ]

for dir, save_file in zip(DATA_DIRS, SAVE_FILES):
    dataset = []
    for subset in ["train", "val", "test"]:
        print(f"Starting {dir}{subset}")
        if os.path.isdir(dir+subset):
            files = [dir+subset+"/"+x for x in os.listdir(dir+subset) if x[-4:]==".npy"]
            print("Num files:", len(files))
            for i, f in enumerate(files):
                if not i%100:
                    print(f"Iter {i}")

                wavefunction = np.load(f)
                small = resize(np.stack([wavefunction.real, wavefunction.imag], axis=-1), [96,96,2])
                dataset.append(small)
    dataset = np.stack(dataset).astype(np.float32)

    np.save(SAVE_DIR+save_file, dataset)