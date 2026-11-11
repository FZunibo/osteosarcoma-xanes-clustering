"""Minimal methodological code for latent-space XANES clustering."""

from .models import SpectralAutoencoder
from .losses import SpectralL1Loss, compute_mmd, mmd_to_standard_normal
from .clustering import (
    clustering_scores,
    dbcv_score,
    optimize_dbscan,
    optimize_hdbscan,
    project_labels_with_svc,
    silhouette_without_noise,
)

__all__ = [
    "SpectralAutoencoder",
    "SpectralL1Loss",
    "compute_mmd",
    "mmd_to_standard_normal",
    "silhouette_without_noise",
    "dbcv_score",
    "clustering_scores",
    "optimize_dbscan",
    "optimize_hdbscan",
    "project_labels_with_svc",
]
