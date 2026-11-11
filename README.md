# Code accompanying the osteosarcoma XANES latent-space clustering study

This repository contains the reusable methodological core associated with the manuscript:

**Unsupervised Latent Space Clustering of Osteosarcoma X-ray Spectra Using PCA and Deep Autoencoders**

## Included code

- `os_xanes_latent/models.py` — 1D convolutional autoencoder used to obtain low-dimensional latent representations.
- `os_xanes_latent/losses.py` — peak- and gradient-sensitive L1 reconstruction loss and Gaussian-kernel MMD regularization.
- `os_xanes_latent/clustering.py` — DBSCAN/HDBSCAN Silhouette-based hyperparameter selection, Silhouette/DBCV evaluation, and SVC-based out-of-sample label projection.

## Minimal usage

```python
import numpy as np
import torch

from os_xanes_latent import (
    SpectralAutoencoder,
    SpectralL1Loss,
    optimize_dbscan,
    clustering_scores,
    project_labels_with_svc,
)

# Autoencoder input shape: (batch, channels, spectral_length)
x = torch.randn(8, 1, 345)
model = SpectralAutoencoder(
    bottleneck_size=2,
    n_layers=1,
    siglen=345,
    n_signals=1,
    h_dims=32,
    final_conv_dim=4,
    kernel_size=8,
    stride=1,
)
z, x_hat = model(x, return_code=True)

criterion = SpectralL1Loss(peak_weight=2.0, gradient_weight=1.0)
reconstruction_loss = criterion(x_hat, x)

# Example clustering on a two-dimensional representation.
X_dev = np.asarray(z.detach().cpu())
result = optimize_dbscan(
    X_dev,
    eps_values=np.linspace(0.1, 1.0, 20),
    min_samples_values=range(2, 15),
)
labels_dev = result["labels"]

# DBCV requires the optional `dbcv` package used by the study code.
scores = clustering_scores(X_dev, labels_dev)

# Density-based cluster labels can be approximately projected to held-out embeddings.
X_test = X_dev[:2]
labels_test = project_labels_with_svc(X_dev, labels_dev, X_test)
```

The SVC projection is an approximate out-of-sample extension of the density-based clustering solution. Noise points (`-1`) are retained as an additional SVC class, matching the study implementation.

## Scope
This repo is meant to provide the code backbone of the paper but not to reproduce all the numerical results contained in it.
Exact reproduction of the numerical results reported in the manuscript requires authorized access to the original dataset and the study-specific procedures described in the paper.
