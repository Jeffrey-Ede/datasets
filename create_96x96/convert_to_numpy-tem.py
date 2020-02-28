import os
import numpy as np

from scipy.misc import imread

PARENT_DIR = "H:/small_scans-tem/"

print(np.load(PARENT_DIR+"96x96-tem.npy").shape)
quit()

imgs = []
for i, f in enumerate([PARENT_DIR+f for f in os.listdir(PARENT_DIR) if f[-4:] == ".tif"]):
    if not i % 100:
       print(i)

    img = imread(f, mode="F")
    imgs.append(img)
imgs = np.stack(imgs)
imgs = np.expand_dims(imgs, axis=-1)

np.save(PARENT_DIR+"96x96-tem.npy", imgs)
