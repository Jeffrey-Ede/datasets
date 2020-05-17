# Variational Autoencoders

This directory is for variational autoecoders (VAEs). It contains their source code, dataset encodings, points in tSNE visualizations (for interactive visualizations), and modified tSNE implementations.

Our VAE code is adapted from VAE-GAN code [here](https://github.com/zhangqianhui/vae-gan-tensorflow).

## Experiments

There are five directories containing source code for VAE experiments:

`1`: Stem crops  
`2`: Downsampled STEM  
`3`: Downsampled TEM  
`4`: STEM crops without gradient loss  
`5`: STEM crops with Kullback-Liebler divergence, rather than feature normalization and regularization.

The files `vae_embeddings.npy` and `vae_errors` in each folder contain means and standard deviations, respectively, of latent spaces encoded by VAEs for each datasets. 

In this folder, `.npy` files starting with `vae_tsne` contain points in tSNE visualizations created from dataset embeddings. If `_we` is in a file name, it means that standard deviations were accounted for during tSNE optimization. 

## Uniform Separation tSNE

An algorithm to uniformly separate points in tSNE visualizations (or other points plots) is in `tsne_to_uniform.py`. Uniformly separated points of a tSNE visualizations for downsampled STEM images (`2`) are in `vae_tsne_stem_crops_96x96_uniform.npy`.

## tSNE with Standard Deviations

A fast Barnes-Hut tSNE implementation that has been adapted to account for standard deviations is in `bhtsne`. It is adapted from code [here](https://lvdmaaten.github.io/tsne/) and an unmodified version is in `unmodified_bhtsne`. There are precompiled Windows binaries in both folders.

Example usage:

```python
from bhtsne.bhtsne import run_bh_tsne

... #Prepare dataset and hyperparameters for tSNE

#Run tSNE
tsne_points = run_bh_tsne(
  embeddings, 
  sigma, 
  no_dims=2, 
  perplexity=perplexity, 
  theta=0.5, 
  randseed=-1, 
  verbose=True, 
  initial_dims=embeddings.shape[-1], 
  use_pca=False, 
  max_iter=10_000
)

#Discard last half of returned array
tsne_points = tsne_points[:tsne_points.shape[0]//2] 
```

## Pretrained Models

Pretrained VAEs for `1`, `2`, `3`, `4` and `5` can be downloaded [here](https://drive.google.com/drive/folders/1vdEKgrg6ymsvBO0LnwCPbfpeqZ9Z7Kan?usp=sharing).

## Contact

Jeffrey Ede: j.m.ede@warwick.ac.uk
