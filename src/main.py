"""Run the flight-trajectory clustering pipeline."""

import hdbscan
import numpy as np

from .clustering import (
    compute_hausdorff_distance_matrix,
    evaluate_clustering,
)
from .data import (
    read_data,
    resample_trajectory,
)
from .visualization import (
    plot_cluster_mean_trajectories,
    plot_paths,
)


def main():
    # 1) Load data
    flights, _ = read_data()
    print(f"Total flights: {len(flights)}\n")

    # 2) Resample trajectories
    n_points = 300
    coords_list = []

    for df in flights:
        coords, *_ = resample_trajectory(df, n_points=n_points)
        coords_list.append(coords)

    coords_arr = np.stack(coords_list)  # (N, n_points, 2)

    # 3) Compute the Hausdorff distance matrix
    print("Computing Hausdorff distance matrix...")
    dist_matrix = compute_hausdorff_distance_matrix(coords_arr)
    print("Done.\n")

    # 4) Fit HDBSCAN
    model = hdbscan.HDBSCAN(
        min_cluster_size=80,
        min_samples=98,
        metric="precomputed",
    )
    labels = model.fit_predict(dist_matrix)

    # 5) Evaluate clustering
    score, info = evaluate_clustering(dist_matrix, labels)
    print("=== Clustering Evaluation ===")
    print(f"Silhouette score: {info['silhouette']:.6f}")
    print(f"Noise ratio: {info['noise_ratio']:.6f}")
    print(f"Project heuristic score: {score:.6f}")

    # Report cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== Cluster Sizes ===")
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c}")

    # 6) Plot clustered trajectories
    plot_paths(coords_arr, labels)

    # 7) Plot cluster-wise mean trajectories
    plot_cluster_mean_trajectories(coords_arr, labels)


if __name__ == "__main__":
    main()
