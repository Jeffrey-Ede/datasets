import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 300
fontsize = 11
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize

from matplotlib.offsetbox import OffsetImage, AnnotationBbox


DATASET_FILES = [
    "//Desktop-sa1evjv/h/96x96_stem_crops.npy",
    "//Desktop-sa1evjv/h/small_scans/96x96.npy",
    "//Desktop-sa1evjv/h/small_scans-tem/96x96-tem.npy",
    "//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_n=3.npy",
    "//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_restricted_n=3.npy",
    "//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_single_n=3.npy",
    ]

DATA_LOC = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/"

DATASET_NAMES = [
    "stem_crops_96x96",
    "stem_downsampled_96x96",
    "tem_downsampled_96x96",
    "wavefunctions_n=3",
    "wavefunctions_restricted_n=3",
    "wavefunctions_single_n=3",
    ]


SEEDS = [3, 1, 1, 1, 1, 1]

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


for i, (data_file, data_name, seed) in enumerate(zip(DATASET_FILES, DATASET_NAMES, SEEDS)):

    print(i)

    dataset_filepath = DATA_LOC + "tsne_" + data_name + ".npy"
    tsne = np.load(dataset_filepath)


    x = tsne[:,0]
    y = tsne[:,1]

    if i == 0: #Remove outliers to improve visualization 
        idx = x.argmin()
        x = np.concatenate([x[:idx], x[idx+1:]], axis=0)
        y = np.concatenate([y[:idx], y[idx+1:]], axis=0)
    elif i == 2:
        x0to1 = scale0to1(x)
        y0to1 = scale0to1(y)

        keep = (x0to1 > 0.3)*(x0to1 < 0.9)*(y0to1 > 0.2)*(y0to1 < 0.8)

        x = x[keep]
        y = y[keep]
    #elif i == 3:
    #    x0to1 = scale0to1(x)
    #    y0to1 = scale0to1(y)

    #    keep = (x0to1 > 0.15)*(x0to1 < 0.75)*(y0to1 > 0.05)

    #    x = x[keep]
    #    y = y[keep]
    #elif i == 4:
    #    x0to1 = scale0to1(x)
    #    y0to1 = scale0to1(y)

    #    keep = (x0to1 > 0.07)

    #    x = x[keep]
    #    y = y[keep]


    y = scale0to1(y)
    x = scale0to1(x)

    tsne = np.stack([x,y], axis=-1)

    arr = np.sqrt(np.sum(np.load(data_file)**2, axis=-1))

    arr = np.stack([scale0to1(x) for x in arr])

    fig = plt.figure(figsize=(5, 4))

    scale = 4
    width = scale * 2.2
    height = 1.1*scale* (width / 1.618) / 2.2
    fig.set_size_inches(width, height)


    ax = fig.add_subplot(211)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.text(-0.05, 1.021, "a)")
    plt.grid()
    ax.tick_params(axis=u'both', which=u'both',length=0)
    #plt.setp(ax.get_xticklabels(), visible=False)
    #plt.setp(ax.get_yticklabels(), visible=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    line, = ax.plot(x, y, ls="", marker="o", color="black", alpha=1, markersize=0.5)

    ax = fig.add_subplot(212)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.text(-0.05, 1.021, "b)")
    plt.grid()
    ax.tick_params(axis=u'both', which=u'both',length=0)
    #plt.setp(ax.get_xticklabels(), visible=False)
    #plt.setp(ax.get_yticklabels(), visible=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    np.random.seed(seed)
    idxs = np.random.choice(np.arange(0, tsne.shape[0]), 500, replace=False)
    for idx in idxs:
        im = OffsetImage(arr[idx,:,:], zoom=0.2, cmap="gray", norm=mpl.colors.Normalize(vmin=0.,vmax=1.))
        ab = AnnotationBbox(
            im, tsne[idx], xybox=tsne[idx], xycoords='data',  pad=0., frameon=False,
            arrowprops=dict(arrowstyle="-"))

        ax.add_artist(ab) 

    fig.subplots_adjust(hspace=0.07)

    plt.draw()
    #plt.show()
    fig.savefig(DATA_LOC+data_name+'.png', bbox_inches='tight')
