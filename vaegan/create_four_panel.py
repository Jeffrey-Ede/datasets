import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 400
fontsize = 8
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize

from matplotlib.offsetbox import OffsetImage, AnnotationBbox


DATASET_FILES = [
    "//Desktop-sa1evjv/h/96x96_stem_crops.npy",
    "//Desktop-sa1evjv/h/96x96_stem_crops.npy",
    "//Desktop-sa1evjv/h/96x96_stem_crops.npy",
    "//Desktop-sa1evjv/h/96x96_stem_crops.npy"
    ]

DATASET_NAMES = [
    r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/tsne_stem_crops_96x96",
    r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/vaegan/vae_tsne_stem_crops_96x96_no_regul_we",
    r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/vaegan/vae_tsne_stem_crops_96x96_no_sobel_we",
    r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/vaegan/vae_tsne_stem_crops_96x96_we",
    ]

DATASET_PARTITION_SIZES = [
    [14826, 1977, 2966],
    [14826, 1977, 2966],
    [11350, 2431, 3486],
    [24530, 3399, 8395],
    [8002, 1105, 2763],
    [3861, 964, 0]
    ]

SOBEL = False

IS_CONSIDERING_ERRORS = False

if SOBEL:
    DATASET_NAMES = [n+"_sobel" for n in DATASET_NAMES]

SEEDS = [1, 1, 1, 1, 1, 1]

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img - min)/(max - min)

    return img.astype(np.float32)


letters = ["a", "b", "c", "d"]

titles = ["a): PCA", "b): Traditional VAE", "c): Normalized VAE Without Gradient Loss", "d): Normalized VAE With Gradient Loss"]

fig = plt.figure(figsize=(5, 4))

scale = 4
width = scale * 2.2
height = scale* (width / 1.618) / 2.5
fig.set_size_inches(width, height)


for i, (data_file, data_name, seed, P, title) in enumerate(zip(
    DATASET_FILES, DATASET_NAMES, SEEDS, DATASET_PARTITION_SIZES, titles)):

    if IS_CONSIDERING_ERRORS:
        data_name += "_we"

    dataset_filepath = data_name + ".npy"
    tsne = np.load(dataset_filepath)
    print(tsne.shape)

    x = tsne[:,0]
    y = tsne[:,1]

    if not i:
        x0to1 = scale0to1(x)
        y0to1 = scale0to1(y)

        keep = (x0to1 > 0.1)

        x = x[keep]
        y = y[keep]

    y = scale0to1(y)
    x = scale0to1(x)

    tsne = np.stack([x,y], axis=-1)

    #arr = np.sqrt(np.sum(np.load(data_file)**2, axis=-1))
    arr = np.load(data_file)

    arr = arr[...,0]
    arr = np.stack([scale0to1(x) for x in arr])


    ax = fig.add_subplot("22"+str(i+1))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    #ax.text(-0.05, 1.023, letters[i]+")")
    plt.grid()
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    np.random.seed(seed)
    idxs = np.random.choice(np.arange(0, tsne.shape[0]), 500, replace=False)
    for idx in idxs:

        im = OffsetImage(arr[idx,:,:], zoom=0.15, cmap="gray", norm=mpl.colors.Normalize(vmin=0.,vmax=1.))

        ab = AnnotationBbox(
            im, tsne[idx], xybox=tsne[idx], xycoords='data',  pad=0., frameon=False,
            arrowprops=dict(arrowstyle="-"))

        ax.add_artist(ab) 

    #fig.subplots_adjust(hspace=0.07)

    plt.title(title, fontweight="bold")

fig.subplots_adjust(wspace=0.06, hspace=0.11)

plt.draw()
fig.savefig(r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/vaegan/four_panel_tsne.png", bbox_inches='tight')
