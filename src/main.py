"""Flight trajectory clustering with HDBSCAN."""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# =========================
# 1) Load data
# =========================
FLIGHT_DATA_FILES = (
    ('AAR', 'ICN_SIN_AAR751_final.xlsx'),
    ('JJA', 'ICN_SIN_JJA2623_final.xlsx'),
    ('KAL643', 'ICN_SIN_KAL643_final.xlsx'),
    ('KAL645', 'ICN_SIN_KAL645_final.xlsx'),
    ('SIA601', 'ICN_SIN_SIA601_final.xlsx'),
    ('SIA605', 'ICN_SIN_SIA605_final.xlsx'),
    ('TGW', 'ICN_SIN_TGW843_final.xlsx'),
    ('TWB', 'ICN_SIN_TWB161_final.xlsx'),
)

TRAJECTORY_COLUMNS = (
    'Time (KST)',
    'Latitude',
    'Longitude',
    'kts',
    'mph',
    'feet',
)


def read_data():
    flights = []
    meta = []

    for airline, filename in FLIGHT_DATA_FILES:
        workbook = pd.read_excel(
            DATA_DIR / filename,
            sheet_name=None
        )

        for i in range(1, 100):
            sheet_name = f'Sheet{i}'

            if sheet_name not in workbook:
                continue

            df = workbook[sheet_name][
                list(TRAJECTORY_COLUMNS)
            ].copy()

            df[['kts', 'mph', 'feet']] = df[
                ['kts', 'mph', 'feet']
            ].interpolate(
                method='linear',
                limit_direction='both'
            )

            df = df.dropna(
                subset=[
                    'Time (KST)',
                    'Latitude',
                    'Longitude',
                ]
            )

            if len(df) == 0:
                continue

            flights.append(df)
            meta.append({
                'airline': airline,
                'sheet': sheet_name,
                'flight_index': len(flights) - 1,
            })

    return flights, meta


# =========================
# 2) Resample trajectory, speed, and altitude
# =========================
def resample_trajectory(data, n_points=100):
    """
    data: DataFrame for one flight containing time, position, speed, and altitude
    n_points: Number of points in the resampled trajectory
    return:
        coords:   (n_points, 2)  [lon, lat]
        kts_new:  (n_points,)
        mph_new:  (n_points,)
        feet_new: (n_points,)
    """
    lon = data['Longitude'].values
    lat = data['Latitude'].values
    kts = data['kts'].values
    mph = data['mph'].values
    feet = data['feet'].values

    n = len(data)

    # Normalize the original sample positions to the interval [0, 1]
    orig_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, n_points)

    # Linear interpolation
    lon_new = np.interp(new_idx, orig_idx, lon)
    lat_new = np.interp(new_idx, orig_idx, lat)
    kts_new = np.interp(new_idx, orig_idx, kts)
    mph_new = np.interp(new_idx, orig_idx, mph)
    feet_new = np.interp(new_idx, orig_idx, feet)

    coords = np.column_stack([lon_new, lat_new])  # (n_points, 2)
    return coords, kts_new, mph_new, feet_new


# =========================
# 3) Hausdorff distance
# =========================
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


# =========================
# 4) Plot individual trajectories
# =========================
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


# =========================
# 5) Plot cluster-wise mean trajectories
# =========================
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


# =========================
# 6) Optional analysis: cluster-wise speed and altitude profiles
# =========================
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


# =========================
# 7) Evaluate HDBSCAN clustering
# =========================
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


# =========================
# 8) main()
# =========================
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

    coords_arr = np.stack(coords_list)   # (N, n_points, 2)

    # 3) Compute the Hausdorff distance matrix
    print("Computing Hausdorff distance matrix...")
    dist_matrix = compute_hausdorff_distance_matrix(coords_arr)
    print("Done.\n")

    # 4) Fit HDBSCAN
    model = hdbscan.HDBSCAN(
        min_cluster_size=80,
        min_samples=98,
        metric='precomputed'
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
