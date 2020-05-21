import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 400
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
    "//Desktop-sa1evjv/h/96x96_stem_crops.npy",
    "//Desktop-sa1evjv/h/96x96_stem_crops.npy",
    #"//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_n=3.npy",
    #"//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_restricted_n=3.npy",
    "//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_single_n=3.npy",
    ]

DATA_LOC = r"Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/vaegan/"

DATASET_NAMES = [
    "stem_crops_96x96",
    "stem_downsampled_96x96",
    "tem_downsampled_96x96",
    "stem_crops_96x96_no_sobel",
    "stem_crops_96x96_no_regul",
    #"wavefunctions_n=3",
    #"wavefunctions_restricted_n=3",
    "wavefunctions_single_n=3",
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
IS_UNIFORM = True

if SOBEL:
    DATASET_NAMES = [n+"_sobel" for n in DATASET_NAMES]

SEEDS = [11, 11, 11, 1, 1, 1]

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


for i, (data_file, data_name, seed, P) in enumerate(zip(DATASET_FILES, DATASET_NAMES, SEEDS, DATASET_PARTITION_SIZES)):

    if not i in [0,1,2]:
        continue

    if IS_CONSIDERING_ERRORS:
        data_name += "_we"

    if IS_UNIFORM:
        data_name += "_uniform"

    dataset_filepath = DATA_LOC + "vae_tsne_" + data_name + ".npy"
    tsne = np.load(dataset_filepath)
    print(tsne.shape)

    is_in_convex_hull = np.isfinite(tsne[:,0])*np.isfinite(tsne[:,1])

    tsne = tsne[is_in_convex_hull]

    x = tsne[:,0]
    y = tsne[:,1]

    y = scale0to1(y)
    x = scale0to1(x)

    tsne = np.stack([x,y], axis=-1)

    arr = np.load(data_file)
    arr = arr[is_in_convex_hull]

    if i <= 4:
        arr = arr[...,0]
        arr = np.stack([scale0to1(x) for x in arr])
    else:
        std = np.std(arr)
        arr /= 7*std 
        arr += ( 0.5 - np.mean(arr) )
        arr = arr.clip(0., 1.)
        arr = np.concatenate( (arr[...,:1], np.zeros(list(arr.shape)[:-1] + [1]), arr[...,1:]), axis=-1 )

    fig = plt.figure(figsize=(5, 4))

    scale = 4
    width = scale * 2.2
    height = 1.1*scale* (width / 1.618) / 2.2 / 2.1
    fig.set_size_inches(width, height)


    #ax = fig.add_subplot(211)
    #ax.set_xlim(-0.05, 1.05)
    #ax.set_ylim(-0.05, 1.05)
    #ax.text(-0.05, 1.021, "a)")
    #plt.grid()
    #ax.tick_params(axis=u'both', which=u'both',length=0)
    #plt.setp(ax.get_xticklabels(), visible=False)
    #plt.setp(ax.get_yticklabels(), visible=False)
    #for spine in plt.gca().spines.values():
    #    spine.set_visible(False)

    #line, = ax.plot(x, y, ls="", marker="o", color="black", alpha=1, markersize=0.5)
    ##line, = ax.plot(x[:P[0]], y[:P[0]], ls="", marker="o", color="black", alpha=1, markersize=0.5)
    ##line, = ax.plot(x[P[0]:P[0]+P[1]], y[P[0]:P[0]+P[1]], ls="", marker="o", color="green", alpha=1, markersize=0.5)
    ##if P[2]:
    ##    line, = ax.plot(x[-P[2]:], y[-P[2]:], ls="", marker="o", color="blue", alpha=1, markersize=0.5)

    ax = fig.add_subplot(111)
    #ax.set_xlim(-0.05, 1.05)
    #ax.set_ylim(-0.05, 1.05)
    #ax.text(-0.05, 1.021, "b)")
    #plt.grid()
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    if not IS_UNIFORM:
        np.random.seed(seed)
        idxs = np.random.choice(np.arange(0, tsne.shape[0]), 500, replace=False)
    else:
        idxs = []
        positions = []
        h_size = 35
        w_size = 20
        for a in range(h_size):
            for b in range(w_size):
                _x = (a + 0.5)/h_size
                _y = (b + 0.5)/w_size

                idx = np.argmin( (x-_x)**2 + (y-_y)**2 )
                idxs.append(idx)
                positions.append( np.array([_x, _y]) )

    for j, idx in enumerate(idxs):

        if IS_UNIFORM:
            pos = positions[j]
        else: 
            pos = tsne[idx]

        if i <= 5:
            im = OffsetImage(arr[idx,:,:], zoom=0.15, cmap="gray", norm=mpl.colors.Normalize(vmin=0.,vmax=1.))
        else:
            im = OffsetImage(arr[idx,:,:,:], zoom=0.2)

        ab = AnnotationBbox(
            im, tsne[idx], xybox=pos, xycoords='data',  pad=0., frameon=False,
            arrowprops=dict(arrowstyle="-"))

        ax.add_artist(ab) 

    fig.subplots_adjust(hspace=0.07)

    plt.draw()
    #plt.show()
    fig.savefig(DATA_LOC+"vae_"+data_name+'.png', bbox_inches='tight')
