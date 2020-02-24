import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 300
fontsize = 9
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

    if not i in [2]:
        continue

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

    
    if i == 1:
        s_num = 0
        examples = [
            [[0.2612, 0.7293], [0.3280, 0.7124], [0.4194, 0.6800]],
            [[0.8101, 0.3855], [0.8350, 0.3659], [0.6528, 0.5311]],
            [[0.9384, 0.2963], [0.9279, 0.2496], [0.0695, 0.7595]],
            [[0.0764, 0.7630], [0.0776, 0.7463], [0.0545, 0.7474]],
            [[0.2441, 0.2872], [0.1581, 0.3542], [0.2067, 0.3158]],
            [[0.5197, 0.6324], [0.5737, 0.8413], [0.5867, 0.8611]],
            [[0.5636, 0.3498], [0.8861, 0.2978], [0.5410, 0.5614]],
            [[0.3871, 0.9391], [0.4212, 0.0777], [0.8406, 0.5966]],
            ]
    elif i == 2:
        s_num = 1
        examples = [
            [[0.3806, 0.9502], [0.3464, 0.9987], [0.3673, 0.9372]],
            [[0.4714, 0.7242], [0.4322, 0.7621], [0.3286, 0.7526]],
            [[0.7917, 0.1910], [0.7561, 0.3079], [0.6212, 0.3715]],
            [[0.8137, 0.1618], [0.7532, 0.1936], [0.8059, 0.1097]],
            [[0.2335, 0.4627], [0.0684, 0.5358], [0.0568, 0.5332]],
            [[0.7528, 0.1751], [0.1967, 0.6090], [0.2450, 0.6206]],
            [[0.6787, 0.4903], [0.4970, 0.5970], [0.6983, 0.2090]],
            [[0.5602, 0.1585], [0.4948, 0.4340], [0.2857, 0.6722]],
            ]

    for e_num, positions in enumerate(examples):

        imgs = []
        labels = []
        for p in positions:
            labels.append( f"{p[0]:.3}, {p[1]:.3}" )
            imgs.append( arr[np.sum( (np.expand_dims(p, 0) - tsne)**2, axis=-1).argmin()] )

        columns = 3
        rows = 1
        fig = plt.figure(figsize=(columns, rows))
    
        for i in range(rows):
            for j in range(1, columns+1):
                k = i*columns+j
                ax = fig.add_subplot(rows, columns, k)

                plt.imshow(imgs[k-1], cmap="gray", norm=mpl.colors.Normalize(vmin=0.,vmax=1.))
                plt.xticks([])
                plt.yticks([])

                ax.set_frame_on(False)
                #ax.set_xlabel(labels[k-1])

        fig.savefig(DATA_LOC+f"table_examples/{s_num}-{e_num}.png", bbox_inches='tight')
