import numpy as np


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def preprocess(img):

    try:
        img[np.isnan(img)] = 0.
        img[np.isinf(img)] = 0.
    except:
        img = np.zeros([96,96,1])

    return img

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

    SAVE_LOC = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/"

    DATASET_FILES = [
        "//Desktop-sa1evjv/h/96x96_stem_crops.npy",
        "//Desktop-sa1evjv/h/small_scans/96x96.npy",
        "//Desktop-sa1evjv/h/small_scans-tem/96x96-tem.npy",
        "//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_n=3.npy",
        "//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_restricted_n=3.npy",
        "//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_single_n=3.npy",
        ]

    DATASET_NAMES = [
        "stem_crops_96x96",
        "stem_downsampled_96x96",
        "tem_downsampled_96x96",
        "wavefunctions_n=3",
        "wavefunctions_restricted_n=3",
        "wavefunctions_single_n=3",
        ]

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

    for i, (dataset_file, dataset_name, pca_dimension, is_scale0to1) in enumerate(zip(
        DATASET_FILES, DATASET_NAMES, PCA_DIMENSIONS, IS_SCALE0TO1S)):

        if i <= 2:
            continue

        dataset = np.load(dataset_file)
        print(dataset.shape)

        if is_scale0to1 and i <= 2:
            dataset = np.stack([scale0to1(preprocess(x)) for x in dataset])

        pc = batch_PCA(dataset, pca_dimension)

        if i <= 2:
            perplexity = int(np.sqrt(pc.shape[0]))
        else:
            perplexity = int(5*np.sqrt(pc.shape[0]))
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=10000).fit_transform(pc)

        np.save(SAVE_LOC+"pca_"+dataset_name+".npy", pc)
        np.save(SAVE_LOC+"tsne_"+dataset_name+".npy", tsne)