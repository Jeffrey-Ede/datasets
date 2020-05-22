import numpy as np
from scipy.misc import imread
from scipy.stats import entropy

import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 100
fontsize = 11
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize

import matplotlib.mlab as mlab

import cv2

from PIL import Image
from PIL import ImageDraw

columns = 6
rows = 7

DATA_LOC = r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/vaegan/"

#data_file = "//Desktop-sa1evjv/h/small_scans/96x96.npy"
data_file = "//Desktop-sa1evjv/h/small_scans-tem/96x96-tem.npy"

#encoding_loc = r"Y:\HCSS6\Shared305\Microscopy\Jeffrey-Ede\models\visualize_data\vaegan\4"
encoding_loc = r"Y:\HCSS6\Shared305\Microscopy\Jeffrey-Ede\models\visualize_data\vaegan\2"

parent = "Z:/Jeffrey-Ede/models/visualize_data/vaegan/tem"

def preprocess(img):

    img[np.isnan(img)] = 0.
    img[np.isinf(img)] = 0.

    return img

def scale0to1(img):
    
    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)


data = np.load(data_file)
data = np.stack([scale0to1(preprocess(x)) for x in data])
data = data[...,0]

embeddings_file = encoding_loc + r"\vae_embeddings.npy"
mu = np.load(embeddings_file)

sigma_file = encoding_loc + r"\vae_errors.npy"
sigma = np.load(sigma_file)

#np.random.seed(1)

imgs = []
x_titles = []
for i in np.random.choice(np.arange(0, data.shape[0]), rows, replace=False):
    #dists = np.sum( (mu - mu[i:i+1])**2/(sigma**2 + sigma[i:i+1]**2 + 0.01), axis=-1 )
    #dists /= np.sum( 1/(sigma**2 + sigma[i:i+1]**2 + 0.01), axis=-1 )
    #dists = np.sqrt(dists)

    dists = np.sqrt( np.sum( (mu - mu[i:i+1])**2, axis=-1) )

    sorted_idx = np.argsort(dists)

    imgs.append(data[i])
    x_titles.append("Input")

    for j in range(1, columns):
        imgs.append(data[sorted_idx[j]])
        x_titles.append("Distance: {:.2f}".format(dists[sorted_idx[j]]))


def block_resize(img, new_size):

    x = np.zeros(new_size)
    
    dx = int(new_size[0]/img.shape[0])
    dy = int(new_size[1]/img.shape[1])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            px = img[i,j]

            for u in range(dx):
                for v in range(dy):
                    x[i*dx+u, j*dy+v] = px

    return x
    
#Width as measured in inkscape
scale = 4
width = scale * 2.2
height = 1.15*scale* (width / 1.618) / 2.08

w = h = 96

f=plt.figure(figsize=(rows, columns))

for i in range(rows):
    for j in range(1, columns+1):
        img = imgs[i*columns+j-1]
        #extra = 3*rows if j >= 4 else 0
        ##extra = 3*rows if (columns*i+j-1)%6 >= 3 else 0
        
        #excess_j = j if j < 4 else j-3
        #img = imgs[3*i+excess_j-1+extra]
        ##img = imgs[columns*i+j-1+extra]

        k = i*columns+j
        ax = f.add_subplot(rows, columns, k)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])

        ax.set_frame_on(False)
        ax.set_title(x_titles[i*columns+j-1])#, fontsize=fontsize)

f.subplots_adjust(wspace=0.06, hspace=0.04)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)
#f.tight_layout()

f.set_size_inches(width, height)

#plt.show()

f.savefig(parent+'_search_results.png', bbox_inches='tight')
