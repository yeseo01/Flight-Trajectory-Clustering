"""Flight trajectory clustering with HDBSCAN."""


# Hausdorff HDBSCAN

# 실행시간: 약 5분 소요
# 해당 코드와 같은 폴더에 데이터를 넣고 실행해야합니다.

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# =========================
# 1) 데이터 불러오기
# =========================
def read_data():
    # 파일 읽기
    data_AAR = pd.read_excel(DATA_DIR / 'ICN_SIN_AAR751_final.xlsx', sheet_name=None)
    data_JJA = pd.read_excel(DATA_DIR / 'ICN_SIN_JJA2623_final.xlsx', sheet_name=None)
    data_KAL643 = pd.read_excel(DATA_DIR / 'ICN_SIN_KAL643_final.xlsx', sheet_name=None)
    data_KAL645 = pd.read_excel(DATA_DIR / 'ICN_SIN_KAL645_final.xlsx', sheet_name=None)
    data_SIA601 = pd.read_excel(DATA_DIR / 'ICN_SIN_SIA601_final.xlsx', sheet_name=None)
    data_SIA605 = pd.read_excel(DATA_DIR / 'ICN_SIN_SIA605_final.xlsx', sheet_name=None)
    data_TGW = pd.read_excel(DATA_DIR / 'ICN_SIN_TGW843_final.xlsx', sheet_name=None)
    data_TWB = pd.read_excel(DATA_DIR / 'ICN_SIN_TWB161_final.xlsx', sheet_name=None)

    airline_data = {
        'AAR': data_AAR,
        'JJA': data_JJA,
        'KAL643': data_KAL643,
        'KAL645': data_KAL645,
        'SIA601': data_SIA601,
        'SIA605': data_SIA605,
        'TGW': data_TGW,
        'TWB': data_TWB
    }

    flights = []   # 각 비행의 DataFrame
    meta = []      # 각 비행의 메타정보 (airline, sheet 등)

    for name in ['AAR', 'JJA', 'KAL643', 'KAL645', 'SIA601', 'SIA605', 'TGW', 'TWB']:
        data_dict = airline_data[name]

        for i in range(1, 100):  # Sheet1~54 시도 (실제 없는 시트는 그냥 스킵)
            sheet_name = f'Sheet{i}'
            if sheet_name not in data_dict:
                continue

            # 필요한 컬럼만 선택해서 복사
            df = data_dict[sheet_name][[
                'Time (KST)', 'Latitude', 'Longitude',
                'kts', 'mph', 'feet'
            ]].copy()

            # kts, mph, feet NaN 선형 보간 (앞/뒤까지 채움)
            df[['kts', 'mph', 'feet']] = df[['kts', 'mph', 'feet']].interpolate(
                method='linear',
                limit_direction='both'
            )

            # 위·경도/시간이 NaN인 행은 버림
            df = df.dropna(subset=['Time (KST)', 'Latitude', 'Longitude'])

            if len(df) == 0:
                continue

            flights.append(df)
            meta.append({
                'airline': name,
                'sheet': sheet_name,
                'flight_index': len(flights) - 1
            })

    return flights, meta


# =========================
# 2) 경로 + 속도/고도 리샘플링
# =========================
def resample_trajectory(data, n_points=100):
    """
    data: 하나의 비행 데이터 (Time, Latitude, Longitude, kts, mph, feet 포함)
    n_points: 새로 생성할 경로의 포인트 개수
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

    # 원래 인덱스를 0~1 구간으로 정규화
    orig_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, n_points)

    # 선형 보간
    lon_new = np.interp(new_idx, orig_idx, lon)
    lat_new = np.interp(new_idx, orig_idx, lat)
    kts_new = np.interp(new_idx, orig_idx, kts)
    mph_new = np.interp(new_idx, orig_idx, mph)
    feet_new = np.interp(new_idx, orig_idx, feet)

    coords = np.column_stack([lon_new, lat_new])  # (n_points, 2)
    return coords, kts_new, mph_new, feet_new


# =========================
# 3) Hausdorff distance 구현
# =========================
def hausdorff_distance(traj1, traj2):
    """
    traj1, traj2: (n_points, 2) 형태의 경로 (lon, lat)
    대칭 Hausdorff distance (Euclidean) 반환
    """
    # (n1, n2, 2)
    diff = traj1[:, None, :] - traj2[None, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))  # (n1, n2)

    h_ab = dist_mat.min(axis=1).max()
    h_ba = dist_mat.min(axis=0).max()
    return max(h_ab, h_ba)

def compute_hausdorff_distance_matrix(coords_arr):
    """
    coords_arr: (N, n_points, 2) 모든 비행 경로
    return: (N, N) 대칭 distance matrix
    """
    N = coords_arr.shape[0]
    D = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i+1, N):
            d = hausdorff_distance(coords_arr[i], coords_arr[j])
            D[i, j] = D[j, i] = d

    return D


# =========================
# 4) 기본 경로 플롯 (개별 경로)
# =========================
def plot_paths(coords_arr, labels):
    """
    coords_arr: (N, n_points, 2)
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
# 5) 클러스터별 평균 경로
# =========================
def plot_cluster_mean_trajectories(coords_arr, labels):
    """
    coords_arr: (N, n_points, 2)
    """
    N, n_points, _ = coords_arr.shape
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
# 6) 클러스터별 평균 속도/고도 프로파일
# =========================
def plot_cluster_profiles(kts_arr, mph_arr, feet_arr, labels):
    """
    kts_arr, mph_arr, feet_arr: (n_flights, n_points)
    """
    n_flights, n_points = kts_arr.shape
    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap('tab10')
    x = np.linspace(0, 1, n_points)  # 0~1: 경로 진행도

    # -------------------------
    # 1) 속도 (kts)
    # -------------------------
    plt.figure(figsize=(10, 4))
    for cluster_label in unique_labels:

        mask = labels == cluster_label
        if np.sum(mask) < 1:
            continue

        mean_kts = kts_arr[mask].mean(axis=0)

        # 노이즈 색상(검정)
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
    # 2) 속도 (mph)
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
    # 3) 고도 (feet)
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
# 7) HDBSCAN 평가
# =========================
def evaluate_clustering(dist_matrix, labels):
    """
    dist_matrix: (N, N) Hausdorff distance matrix
    labels: HDBSCAN 클러스터 라벨
    """
    # Exclude HDBSCAN noise points (label = -1) from silhouette evaluation.
    mask_core = labels != -1
    core_idx = np.where(mask_core)[0]

    # Fewer than two non-noise samples cannot be evaluated.
    if len(core_idx) < 2:
        return -1.0, {
            'n_clusters': 0,
            'noise_ratio': 1.0,
            'silhouette': -1.0
        }

    labels_core = labels[core_idx]
    dist_core = dist_matrix[np.ix_(core_idx, core_idx)]

    # Silhouette score requires at least two non-noise clusters.
    n_clusters = len(np.unique(labels_core))
    if n_clusters < 2:
        return -1.0, {
            'n_clusters': n_clusters,
            'noise_ratio': float(np.mean(labels == -1)),
            'silhouette': -1.0
        }

    # Silhouette ranges from -1 to 1; invalid distance matrices should raise
    # an error rather than being silently treated as poor clustering.
    sil = silhouette_score(
        dist_core,
        labels_core,
        metric='precomputed'
    )

    # Course-project heuristic:
    # 90% silhouette quality + 10% penalty for trajectories classified as noise.
    noise_ratio = float(np.mean(labels == -1))
    combined = 0.9 * sil - 0.1 * noise_ratio

    # 모든 평가 지표를 info 딕셔너리로 묶어서 반환
    info = {
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'silhouette': sil
    }
    return combined, info


# =========================
# 9) main()
# =========================
def main():
    # 1) 데이터 불러오기
    flights, meta = read_data()
    print(f"총 비행 개수: {len(flights)}\n")

    # 2) 리샘플링
    n_points = 300  # 500이 너무 느리면 200~300 정도로 줄여도 됨
    coords_list = []

    for df in flights:
        coords, *_ = resample_trajectory(df, n_points=n_points)
        coords_list.append(coords)

    coords_arr = np.stack(coords_list)   # (N, n_points, 2)

    # 3) Hausdorff distance matrix 계산
    print("Hausdorff distance matrix 계산 중...")
    dist_matrix = compute_hausdorff_distance_matrix(coords_arr)
    print("완료!\n")

    # 4) 학습
    model = hdbscan.HDBSCAN(
        min_cluster_size=80,
        min_samples=98,
        metric='precomputed'
    )
    labels = model.fit_predict(dist_matrix)

    # 5) 평가
    score, info = evaluate_clustering(dist_matrix, labels)
    print("=== Clustering Score ===")
    print(score)

    # --- 클러스터 크기 출력 ---
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== 클러스터 크기 요약 ===")
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c}")

    # 6) 경로 플롯
    plot_paths(coords_arr, labels)

    # 7) 클러스터별 평균 경로
    plot_cluster_mean_trajectories(coords_arr, labels)



if __name__ == "__main__":
    main()
