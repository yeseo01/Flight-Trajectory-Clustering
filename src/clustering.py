"""Trajectory-distance and clustering-evaluation utilities."""

import numpy as np
from sklearn.metrics import silhouette_score


def hausdorff_distance(traj1, traj2):
    """
    traj1, traj2: Trajectories with shape (n_points, 2), ordered as [lon, lat]
    return: Symmetric Hausdorff distance using Euclidean point distances
    """
    # (n1, n2, 2)
    diff = traj1[:, None, :] - traj2[None, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))  # (n1, n2)

    h_ab = dist_mat.min(axis=1).max()
    h_ba = dist_mat.min(axis=0).max()
    return max(h_ab, h_ba)


def compute_hausdorff_distance_matrix(coords_arr):
    """
    coords_arr: All trajectories with shape (N, n_points, 2)
    return: Symmetric pairwise distance matrix with shape (N, N)
    """
    N = coords_arr.shape[0]
    D = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i+1, N):
            d = hausdorff_distance(coords_arr[i], coords_arr[j])
            D[i, j] = D[j, i] = d

    return D


def evaluate_clustering(dist_matrix, labels):
    """
    dist_matrix: Hausdorff distance matrix with shape (N, N)
    labels: HDBSCAN cluster labels
    """
    # Exclude HDBSCAN noise points (label = -1) from silhouette evaluation.
    non_noise_mask = labels != -1
    non_noise_idx = np.where(non_noise_mask)[0]

    # Fewer than two non-noise samples cannot be evaluated.
    if len(non_noise_idx) < 2:
        return -1.0, {
            'n_clusters': 0,
            'noise_ratio': 1.0,
            'silhouette': -1.0
        }

    labels_non_noise = labels[non_noise_idx]
    dist_non_noise = dist_matrix[np.ix_(non_noise_idx, non_noise_idx)]

    # Silhouette score requires at least two non-noise clusters.
    n_clusters = len(np.unique(labels_non_noise))
    if n_clusters < 2:
        return -1.0, {
            'n_clusters': n_clusters,
            'noise_ratio': float(np.mean(labels == -1)),
            'silhouette': -1.0
        }

    # Silhouette ranges from -1 to 1; invalid distance matrices should raise
    # an error rather than being silently treated as poor clustering.
    sil = silhouette_score(
        dist_non_noise,
        labels_non_noise,
        metric='precomputed'
    )

    # Course-project heuristic:
    # 90% silhouette quality + 10% penalty for trajectories classified as noise.
    noise_ratio = float(np.mean(labels == -1))
    combined = 0.9 * sil - 0.1 * noise_ratio

    # Return the individual evaluation metrics together
    info = {
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'silhouette': sil
    }
    return combined, info
