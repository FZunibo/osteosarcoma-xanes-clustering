# Osteosarcoma XANES latent-space clustering

Code accompanying **Unsupervised Latent Space Clustering of Osteosarcoma X Ray Spectra Using PCA and Deep Autoencoders**.

## Contents

- `models.py` — 1D convolutional autoencoder with a two-dimensional latent space.
- `losses.py` — peak- and gradient-sensitive L1 loss and Gaussian-RBF MMD.
- `clustering.py` — DBSCAN/HDBSCAN selection, Silhouette/DBCV scores, and SVC label projection.

All files are in `os_xanes_latent/`.

## Usage

Inputs are PyTorch tensors of shape `(batch, channels, length)`, with one or two channels. The study used 345 spectral points with 20-point mean padding on each side (`length=385`); preprocessing is external to this library.

```python
from os_xanes_latent import SpectralAutoencoder, SpectralL1Loss, mmd_to_standard_normal

model = SpectralAutoencoder(
    bottleneck_size=2, n_layers=4, siglen=385, n_signals=2,
    h_dims=128, final_conv_dim=8, kernel_size=8, stride=2,
)

# x: prepared tensor of shape (batch, 2, 385)
z, x_hat = model(x, return_code=True)
criterion = SpectralL1Loss(peak_weight=2.0, gradient_weight=1.0)
loss_rec = sum(criterion(x_hat[:, i:i+1], x[:, i:i+1]) for i in range(2))
loss_mmd = mmd_to_standard_normal(z, sigma=1.0)
```

For single-channel inputs, use `n_signals=1` and `loss_rec = criterion(x_hat, x)`. The manuscript specifies the training objective and MMD schedule.

## Clustering

The clustering functions accept NumPy embeddings of shape `(n_samples, 2)`. DBSCAN/HDBSCAN parameters are selected by Silhouette with noise excluded and a noise fraction below 5%; DBCV is evaluated afterward. SVC projects labels to held-out embeddings for visualization only.

## Dependencies and scope

Requires PyTorch, NumPy, and scikit-learn (with `sklearn.cluster.HDBSCAN`); DBCV additionally requires `dbcv`.

This is a compact implementation of the main methods, **not** the complete experimental pipeline. It excludes preprocessing, splitting, training scripts, model weights, and private clinical data. Study-specific procedures are documented in the manuscript; exact numerical reproduction requires authorized access to the original data.

