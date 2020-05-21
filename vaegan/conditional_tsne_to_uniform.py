"""Simple algorithm to uniformly distribute tSNE points.

This 2D implementation is for demonstration purposes andand has not been optimized.

Author: Jeffrey Ede
Email: j.m.ede@warwick.ac.uk
"""

import numpy as np
from scipy import interpolate

BASE = r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/"
NAME = "stem_crops_96x96"
TSNE_POINTS_FILE = BASE + r"vaegan/vae_tsne_" + NAME + ".npy"
SAVE_FILE = BASE + r"vaegan/vae_tsne_" + NAME + "_uniform.npy"
GAMMA = 0.3
GRID_SIZE_X = GRID_SIZE_Y = 25
TOL = 1e-4 # Stop iteration after maximum change is below this proportion of point support
MAX_ITER = 100

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

x_probs = []
y_probs_for_x = []
for i in range(GRID_SIZE_X):
    _min = i/GRID_SIZE_X
    _max = (i+1)/GRID_SIZE_X

    select_x = (x > _min)*(x <= _max)
    num_x = np.sum(select_x)
    x_probs.append( num_x )

    if num_x: #If points in this column
        y_probs = []
        for j in range(GRID_SIZE_Y):
            _min = j/GRID_SIZE_Y
            _max = (j+1)/GRID_SIZE_Y

            select_y = select_x*(y > _min)*(y <= _max)
            num_y = np.sum(select_y)
            y_probs.append( num_y )

        y_probs = np.cumsum(y_probs) / num_x
    else:
        y_probs = np.zeros([GRID_SIZE_Y])

    y_probs_for_x.append(y_probs)

#Compute cumulative probabilities
x_probs = np.cumsum(x_probs) / tsne.shape[0]
print(x_probs)

#Create map from grid to distribution
grid_to_map = np.zeros([GRID_SIZE_X, GRID_SIZE_Y, 2])
for i in range(GRID_SIZE_X):
    for j in range(GRID_SIZE_Y):
        idx_x = next((idx for idx, p in enumerate(x_probs) if (i + 0.5)/GRID_SIZE_X <= p ))
        idx_y = next((idx for idx, p in enumerate(y_probs_for_x[idx_x]) if (j + 0.5)/GRID_SIZE_Y <= p ))

        grid_to_map[i, j, 0] = (idx_x+0.5)/GRID_SIZE_X
        grid_to_map[i, j, 1] = (idx_y+0.5)/GRID_SIZE_Y 

##Interpolate map locations at edges of cells
#lin_x = np.linspace(0.5, GRID_SIZE_X - 0.5, GRID_SIZE_X)
#lin_y = np.linspace(0.5, GRID_SIZE_Y - 0.5, GRID_SIZE_Y)

#f0 = interpolate.interp2d(x, y, z[:,:,0], kind='cubic')
#f1 = interpolate.interp2d(x, y, z[:,:,1], kind='cubic')

#lin_x = np.linspace(0.0, GRID_SIZE_X, GRID_SIZE_X+1)
#lin_y = np.linspace(0.0, GRID_SIZE_Y, GRID_SIZE_Y+1)

#full_grid_to_map_x = f0(lin_x, lin_y)
#full_grid_to_map_y = f1(lin_x, lin_y)

#grid_x = np.zeros(x.shape)
#grid_y = np.zeros(y.shape)
#for i in range(GRID_SIZE_X):
#    for i in range(GRID_SIZE_Y):
#        select = (x > full_grid_to_map_x[i])*(x <= full_grid_to_map_x[i+1]) * \
#            (y > full_grid_to_map_y[i])*(y <= full_grid_to_map_y[i+1])

#        #Distances from cell corners
#        d_ll = np.sqrt( (x-full_grid_to_map_x[i])**2 + (y-full_grid_to_map_y[i])**2 )
#        d_lu = np.sqrt( (x-full_grid_to_map_x[i])**2 + (y-full_grid_to_map_y[i+1])**2 )
#        d_ul = np.sqrt( (x-full_grid_to_map_x[i+1])**2 + (y-full_grid_to_map_y[i])**2 )
#        d_uu = np.sqrt( (x-full_grid_to_map_x[i+1])**2 + (y-full_grid_to_map_y[i+1])**2 )

#        grid_x[select] = 

#        for _x, _y in zip(x[select], y[select]):

#Interpolate map locations at edges of cells
lin_x = np.linspace(0.5, GRID_SIZE_X - 0.5, GRID_SIZE_X) / GRID_SIZE_X
lin_y = np.linspace(0.5, GRID_SIZE_Y - 0.5, GRID_SIZE_Y) / GRID_SIZE_Y
xx, yy = np.meshgrid(lin_x, lin_y)

tsne = np.stack([x, y], axis=-1)

x = interpolate.griddata(grid_to_map.reshape(-1, 2), xx.reshape(-1), tsne, method='cubic')
y = interpolate.griddata(grid_to_map.reshape(-1, 2), yy.reshape(-1), tsne, method='cubic')

tsne = np.stack([x, y], axis=-1)

np.save(SAVE_FILE, tsne)
