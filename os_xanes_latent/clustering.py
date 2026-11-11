"""Density-based clustering, validation metrics, and test-label projection."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC


def silhouette_without_noise(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score after excluding samples labelled as noise (-1)."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    mask = labels != -1
    unique_clusters = np.unique(labels[mask])
    if len(unique_clusters) <= 1:
        return float("nan")
    return float(silhouette_score(X[mask], labels[mask]))


def dbcv_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute DBCV on the full label vector, including noise labels.

    The study code used the ``dbcv`` package. It is imported lazily so that the
    rest of this module can be used without that optional dependency.
    """
    try:
        from dbcv import dbcv
    except ImportError as exc:
        raise ImportError(
            "DBCV evaluation requires the optional 'dbcv' package used by the study code."
        ) from exc
    return float(dbcv(np.asarray(X), np.asarray(labels)))


def clustering_scores(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return the Silhouette and DBCV metrics used in the manuscript."""
    return {
        "silhouette": silhouette_without_noise(X, labels),
        "dbcv": dbcv_score(X, labels),
    }


def _noise_fraction(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    return float(np.mean(labels == -1))


def optimize_dbscan(
    X: np.ndarray,
    eps_values: Iterable[float],
    min_samples_values: Iterable[int],
    max_noise_fraction: float = 0.05,
) -> dict:
    """Select DBSCAN hyperparameters by Silhouette score.
    """
    X = np.asarray(X)
    best = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(X)
            mask = labels != -1
            if len(np.unique(labels[mask])) <= 1:
                continue
            score = float(silhouette_score(X[mask], labels[mask]))
            noise_fraction = _noise_fraction(labels)
            if noise_fraction < max_noise_fraction and (
                best is None or score > best["silhouette"]
            ):
                best = {
                    "eps": float(eps),
                    "min_samples": int(min_samples),
                    "silhouette": score,
                    "noise_fraction": noise_fraction,
                    "labels": labels,
                }

    if best is None:
        raise ValueError("No DBSCAN configuration satisfied the selection criteria.")
    return best


def optimize_hdbscan(
    X: np.ndarray,
    min_cluster_sizes: Iterable[int],
    min_samples_values: Iterable[int],
    max_noise_fraction: float = 0.05,
) -> dict:
    """Select HDBSCAN hyperparameters by Silhouette score using the study rule."""
    X = np.asarray(X)
    best = None

    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            labels = HDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples),
            ).fit_predict(X)
            mask = labels != -1
            if len(np.unique(labels[mask])) <= 1:
                continue
            score = float(silhouette_score(X[mask], labels[mask]))
            noise_fraction = _noise_fraction(labels)
            if noise_fraction < max_noise_fraction and (
                best is None or score > best["silhouette"]
            ):
                best = {
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": int(min_samples),
                    "silhouette": score,
                    "noise_fraction": noise_fraction,
                    "labels": labels,
                }

    if best is None:
        raise ValueError("No HDBSCAN configuration satisfied the selection criteria.")
    return best


def project_labels_with_svc(
    X_train: np.ndarray,
    labels_train: np.ndarray,
    X_test: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
) -> np.ndarray:
    """Project density-based cluster labels to held-out embeddings with an SVC.
    """
    X_train = np.asarray(X_train)
    labels_train = np.asarray(labels_train)
    X_test = np.asarray(X_test)

    if len(np.unique(labels_train[labels_train != -1])) <= 1:
        return np.full(X_test.shape[0], -1, dtype=int)

    classifier = SVC(kernel=kernel, C=C)
    classifier.fit(X_train, labels_train)
    return classifier.predict(X_test)
