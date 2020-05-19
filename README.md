# Warwick Electron Microscopy Datasets

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3708904.svg)](https://doi.org/10.5281/zenodo.3708904)

This repository is for the [preprint](https://arxiv.org/abs/2003.01113)|paper "Warwick Electron Microscopy Datasets". It supplements datasets by providing scripts used to curate datasets and their variants, and to create both static and interactive visualizations.

There are three main datasets containing 19769 experimental STEM images, 17266 experimental TEM images and 98340 simulated TEM exit wavefunctions. Datasets are available [here](https://github.com/Jeffrey-Ede/datasets/wiki).

Scrips and data for variational autoencoders (VAEs) and modifications to t-distributed stochastic neighbor embedding (tSNE) are in the `vaegan` subdirectory. Pretrained models are [here](https://drive.google.com/drive/folders/1vdEKgrg6ymsvBO0LnwCPbfpeqZ9Z7Kan?usp=sharing).

# Interactive Visualizations

Interactive visualizations can be created by running `display_visualization_files.py`. Change values of file location variables (in the script) to display their visualization:

SAVE_DATA: Full save location of a NumPy file containing a dataset. For example, from the datasets [main page](https://github.com/Jeffrey-Ede/datasets/wiki).  
SAVE_FILE: Full save location of a NumPy file containing tSNE map points. Files for each visualization are in this repository and have filenames in the form "tsne_*.npy" for PCA and `vae_tsne_*.npy` for VAE, where * is a wildcard.  

An optional extra parameter, USE_FRAC, controls the portion of data points that are displayed. Use a value lower than 1 if a visualization is slow/unresponsive for a large dataset. 

# Other Contents

There are a few folders:

`create_96x96`: Scripts to downsample examples to 96x96.  
`cropping`: Scripts to crop 512x512 regions from full images.  
`mining_scripts`: An assortment of mining scrips used to curate micrographs.  
`stem_full_shapes`: Scripts to investigate the distribution of STEM full images shapes.  
`vaegan`: Source code and pretrained models for VAEs, and source code and precompiled binaries for modified tSNE implementations.

In addition, there are a few noteable fles:

`create_static_displays`: Creates tSNE visualizations with map points and images.  
`create_table_images`: Example TEM and STEM images are selected using their positions in tSNE visualizations.  
`create_visualization_files`: Ouputs [NumPy files](https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html) containing dataset principal componets and tSNE visualizations.  

Scripts to simulate wavefunctions are [here](https://github.com/Jeffrey-Ede/One-Shot/tree/master/wavefunctions). They have the form "run_simulations*.py", where * is a wildcard.

# Contact

Jeffrey Ede: j.m.ede@warwick.ac.uk

# Example Visualization

An example tSNE visualization for 19769 STEM images downsampled to 96x96 is shown below. It was created by training a VAE to encode images means and standard deviations in 64 dimensions. Standard deviations were then used to weight the clustering of means in 2 dimensions by tSNE. Finally, map points were uniformly separated. Images are shown for a 20x35 grid of uniformly separated tSNE points closest to grid points. 


<p align="center">
  <img src="vaegan/vae_stem_downsampled_96x96_uniform.png">
</p>
