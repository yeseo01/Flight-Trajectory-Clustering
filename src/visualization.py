"""Visualization utilities for clustered flight trajectories."""

import matplotlib.pyplot as plt
import numpy as np


def plot_paths(coords_arr, labels):
    """
    coords_arr: Trajectories with shape (N, n_points, 2)
    """
    plt.figure(figsize=(10, 5))

    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap('Paired')

    for cluster_label in unique_labels:
        mask = labels == cluster_label
        flight_idxs = np.where(mask)[0]

        if cluster_label == -1:
            color = 'gray'
            alpha = 0.7
            lw = 1.5
            label_name = 'Noise'
        else:
            color = cmap(cluster_label)
            alpha = 0.5
            lw = 0.7
            label_name = f'Cluster {cluster_label}'

        for idx in flight_idxs:
            lon = coords_arr[idx][:, 0]
            lat = coords_arr[idx][:, 1]
            plt.plot(lon, lat, color=color, alpha=alpha, linewidth=lw, label=label_name)

    plt.title("HDBSCAN Cluster Result (ICN → SIN) [Hausdorff]")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    plt.show()


def plot_cluster_mean_trajectories(coords_arr, labels):
    """
    coords_arr: Trajectories with shape (N, n_points, 2)
    """
    plt.figure(figsize=(10, 5))
    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap('tab10')

    for cluster_label in unique_labels:
        mask = labels == cluster_label
        flight_idxs = np.where(mask)[0]

        for idx in flight_idxs:
            lon = coords_arr[idx][:, 0]
            lat = coords_arr[idx][:, 1]
            plt.plot(lon, lat, color='gray', alpha=0.6, linewidth=1.0)

        if cluster_label == -1:
            continue

        mask = labels == cluster_label
        if np.sum(mask) < 1:
            continue

        cluster_coords = coords_arr[mask]  # (n_cluster, n_points, 2)
        mean_lon = cluster_coords[:, :, 0].mean(axis=0)
        mean_lat = cluster_coords[:, :, 1].mean(axis=0)

        plt.plot(mean_lon, mean_lat,
                 label=f'Cluster {cluster_label} (mean)',
                 linewidth=2.5,
                 color=cmap(cluster_label % 10))

    plt.title("Cluster-wise Mean Trajectories (ICN → SIN) [Hausdorff]")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cluster_profiles(kts_arr, mph_arr, feet_arr, labels):
    """
    kts_arr, mph_arr, feet_arr: (n_flights, n_points)
    """
    n_points = kts_arr.shape[1]
    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap('tab10')
    x = np.linspace(0, 1, n_points)  # normalized trajectory progress from 0 to 1

    # -------------------------
    # 1) Speed (kts)
    # -------------------------
    plt.figure(figsize=(10, 4))
    for cluster_label in unique_labels:

        mask = labels == cluster_label
        if np.sum(mask) < 1:
            continue

        mean_kts = kts_arr[mask].mean(axis=0)

        # Plot noise in black
        if cluster_label == -1:
            color = 'k'
            name = 'Noise'
        else:
            color = cmap(cluster_label % 10)
            name = f"Cluster {cluster_label}"

        plt.plot(x, mean_kts,
                 label=name,
                 linewidth=2,
                 color=color)

    plt.title("Cluster-wise Mean Speed Profile (kts)")
    plt.xlabel("Normalized Path Position")
    plt.ylabel("Speed (kts)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------
    # 2) Speed (mph)
    # -------------------------
    plt.figure(figsize=(10, 4))
    for cluster_label in unique_labels:

        mask = labels == cluster_label
        if np.sum(mask) < 1:
            continue

        mean_mph = mph_arr[mask].mean(axis=0)

        if cluster_label == -1:
            color = 'k'
            name = 'Noise'
        else:
            color = cmap(cluster_label % 10)
            name = f"Cluster {cluster_label}"

        plt.plot(x, mean_mph,
                 label=name,
                 linewidth=2,
                 color=color)

    plt.title("Cluster-wise Mean Speed Profile (mph)")
    plt.xlabel("Normalized Path Position")
    plt.ylabel("Speed (mph)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------
    # 3) Altitude (feet)
    # -------------------------
    plt.figure(figsize=(10, 4))
    for cluster_label in unique_labels:

        mask = labels == cluster_label
        if np.sum(mask) < 1:
            continue

        mean_feet = feet_arr[mask].mean(axis=0)

        if cluster_label == -1:
            color = 'k'
            name = 'Noise'
        else:
            color = cmap(cluster_label % 10)
            name = f"Cluster {cluster_label}"

        plt.plot(x, mean_feet,
                 label=name,
                 linewidth=2,
                 color=color)

    plt.title("Cluster-wise Mean Altitude Profile (feet)")
    plt.xlabel("Normalized Path Position")
    plt.ylabel("Altitude (feet)")
    plt.legend()
    plt.grid(True)
    plt.show()
