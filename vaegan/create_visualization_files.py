import numpy as np


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#from tsne import tsne
from bhtsne.bhtsne import run_bh_tsne
from bhtsne_unmodified.bhtsne import run_bh_tsne as run_bh_tsne_unmodified

import cv2


def preprocess(img):

    try:
        img[np.isnan(img)] = 0.
        img[np.isinf(img)] = 0.
    except:
        img = np.zeros([96,96,1])

    return img

def to_sobel(img):

    g1 = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    g2 = cv2.Scharr(img, cv2.CV_32F, 1, 0)

    x = np.sqrt(g1**2 + g2**2)

    return x

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    if img.shape[-1] != 1:
        img = np.sqrt(np.sum(img**2, axis=-1, keepdims=True))
    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img - min)/(max - min)

    return img.astype(np.float32)


def batch_PCA(data, n_components):

    data = np.stack([np.ndarray.flatten(x) for x in data])
    pc = PCA(n_components=n_components).fit_transform(data)

    print(pc.shape)

    return pc

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    SAVE_LOC = "Y:/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/vaegan/"

    DATASET_FILES = [
        r"Y:\HCSS6\Shared305\Microscopy\Jeffrey-Ede\models\visualize_data\vaegan\1",
        r"Y:\HCSS6\Shared305\Microscopy\Jeffrey-Ede\models\visualize_data\vaegan\4",
        r"Y:\HCSS6\Shared305\Microscopy\Jeffrey-Ede\models\visualize_data\vaegan\2",
        r"Y:\HCSS6\Shared305\Microscopy\Jeffrey-Ede\models\visualize_data\vaegan\3",
        r"Y:\HCSS6\Shared305\Microscopy\Jeffrey-Ede\models\visualize_data\vaegan\5",
        "//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_single_n=3.npy",
        ]

    DATASET_NAMES = [
        "stem_crops_96x96",
        "stem_downsampled_96x96",
        "tem_downsampled_96x96",
        "stem_crops_96x96_no_sobel",
        "stem_crops_96x96_no_regul",
        "wavefunctions_single_n=3",
        ]

    SOBEL = False

    if SOBEL:
        DATASET_NAMES = [n+"_sobel" for n in DATASET_NAMES]

    PCA_DIMENSIONS = [
        50,
        50,
        50,
        50,
        50,
        50,
        ]

    IS_SCALE0TO1S = [
        True,
        True,
        True,
        True,
        True,
        True,
        ]

    IS_CONSIDERING_ERRORS = False

    for i, (dataset_file, dataset_name, pca_dimension, is_scale0to1) in enumerate(zip(
        DATASET_FILES, DATASET_NAMES, PCA_DIMENSIONS, IS_SCALE0TO1S)):

        if not i in [4]:
            continue

        embeddings_file = dataset_file + r"\vae_embeddings.npy"
        embeddings = np.load(embeddings_file)

        if IS_CONSIDERING_ERRORS:
            sigma_file = dataset_file + r"\vae_errors.npy"
            sigma = np.load(sigma_file)

            dataset_name += "_we"
        else:
            sigma = np.array([])

        perplexity = int(np.sqrt(embeddings.shape[0]))

        if IS_CONSIDERING_ERRORS:
            tsne_points = run_bh_tsne(embeddings, sigma, no_dims=2, perplexity=perplexity, theta=0.5, randseed=-1, verbose=True, 
                        initial_dims=embeddings.shape[-1], use_pca=False, max_iter=10_000)
            tsne_points = tsne_points[:tsne_points.shape[0]//2]
        else:
            tsne_points = run_bh_tsne_unmodified(embeddings, no_dims=2, perplexity=perplexity, theta=0.5, randseed=-1, verbose=True, 
                        initial_dims=embeddings.shape[-1], use_pca=False, max_iter=10_000)
        #tsne_points = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=10_000).fit_transform(embeddings)
        #tsne_points = tsne(X=embeddings, no_dims=2, perplexity=perplexity, sigma=sigma, max_iter=10_000)

        np.save(SAVE_LOC+"vae_tsne_"+dataset_name+".npy", tsne_points)