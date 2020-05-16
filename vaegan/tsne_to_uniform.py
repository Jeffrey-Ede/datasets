"""Simple algorithm to uniformly distribute tSNE points.

This implementation is for 2D and has not been optimized in any way.

Author: Jeffrey Ede
Email: j.m.ede@warwick.ac.uk
"""

import numpy as np

TSNE_POINTS_FILE = r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/vaegan/vae_tsne_stem_crops_96x96.npy"
SAVE_FILE = r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/vaegan/vae_tsne_stem_crops_96x96_uniform.npy"
GRID_SIZE = 100
TOL = 1e-4 # Stop iteration after maximum change is below this proportion of point support
MAX_ITER = 500

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    img = (img - min)/(max - min)

    return img.astype(np.float32)

tsne = np.load(TSNE_POINTS_FILE)

x = tsne[:,0]
y = tsne[:,1]

x = scale0to1(x)
y = scale0to1(y)

full_idxs = np.array([i for i in range(tsne.shape[0])])

err = TOL
iters = 0
while err >= TOL and iters < MAX_ITER:
    iters += 1

    x0 = np.copy(x)
    y0 = np.copy(y)

    _max = -1.e-6
    for i in range(GRID_SIZE):
        _min = _max
        _max = (i+1)/GRID_SIZE

        #Grid row or column
        select = (x > _min)*(x <= _max)

        idxs = full_idxs[select]

        #Map to uniformly separated new positions
        y[idxs[np.argsort(y[select])]] = np.linspace(0, 1, num=int(np.sum(select)))

    _max = -1.e-6
    for i in range(GRID_SIZE):
        _min = _max
        _max = (i+1)/GRID_SIZE

        #Grid row or column
        select = (y > _min)*(y <= _max)

        idxs = full_idxs[select]

        #Map to uniformly separated new positions
        x[idxs[np.argsort(x[select])]] = np.linspace(0, 1, num=int(np.sum(select)))

    err = np.mean(np.sqrt( (x-x0)**2 + (y-y0)**2 ))


    print(f"Error: {err}")

tsne = np.stack([x,y], axis=-1)
np.save(SAVE_FILE, tsne)