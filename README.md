# Flight Trajectory Clustering with HDBSCAN

Clustering 591 real-world flight trajectories between Incheon and Singapore using HDBSCAN with a precomputed Hausdorff distance matrix.

This repository contains the reproducible core clustering pipeline from a Fall 2025 team course project on identifying representative flight-path patterns and trajectories labeled as noise.

## Key Highlights

- Clustered 591 real-world flight trajectories on the Incheon-Singapore route.
- Normalized variable-length trajectories to 300 sample-position points for pairwise comparison.
- Used symmetric Hausdorff distances with HDBSCAN through a precomputed distance matrix.
- Reproduced two non-noise clusters containing 87 and 476 trajectories, with 28 noise trajectories and a silhouette score of 0.869918.


## Reproduced Results

The current repository reproduces the final course-project clustering result:

| Result | Value |
| --- | ---: |
| Total trajectories | 591 |
| Cluster 0 | 87 |
| Cluster 1 | 476 |
| Noise | 28 |
| Silhouette score | 0.869918 |
| Noise ratio | 0.047377 |
| Project heuristic score | 0.778189 |

The cluster counts exactly match the final presentation result.


## Pipeline Overview

The project studies variations among flight trajectories on the Incheon–Singapore route.

The core pipeline:

1. loads flight trajectory records,
2. cleans and resamples trajectories to a common number of points,
3. computes pairwise symmetric Hausdorff distances,
4. clusters the trajectories with HDBSCAN,
5. evaluates the resulting clusters using silhouette score and noise ratio, and
6. visualizes the clustered paths and cluster-wise mean trajectories.

## Project Context & My Role

This was a three-person Fall 2025 course project:

**Clustering Flight Trajectories using HDBSCAN: Identifying Operational Patterns and Anomalies**

The assignment required an end-to-end workflow for collecting, preprocessing, clustering, and analyzing real-world flight trajectory data.

**Yeseo Kim**

- proposed normalizing trajectories with different sample counts to a common length,
- implemented the HDBSCAN clustering pipeline for the collected flight data,
- experimented with HDBSCAN hyperparameters, and
- coordinated the overall project workflow and presentation preparation.

Other team contributions included flight-data collection, adapting the trajectory comparison to Hausdorff distance, proposing additional noise-based evaluation and analysis, and investigating trajectories with unusual flight-path patterns.


## Methodology

### 1. Data preprocessing

Each trajectory contains:

- time,
- latitude,
- longitude,
- speed in knots,
- speed in mph, and
- altitude in feet.

Rows missing time, latitude, or longitude are removed. Missing speed and altitude values are linearly interpolated.

The original trajectories contain different numbers of sampled points. Each trajectory is therefore parameterized by normalized sample position from 0 to 1 and linearly resampled to **300 points**.

> This is sample-position resampling, not elapsed-time resampling.

### 2. Trajectory distance

The project uses the symmetric Hausdorff distance between pairs of trajectories.

For trajectories \(A\) and \(B\),

```math
H(A,B) = \max\left(
\max_{a \in A}\min_{b \in B} d(a,b),
\max_{b \in B}\min_{a \in A} d(a,b)
\right).
```

The original course methodology computes pointwise distance using Euclidean distance on longitude/latitude coordinates.

The resulting pairwise distance matrix is passed directly to HDBSCAN using `metric="precomputed"`.

### 3. HDBSCAN

The validated final configuration is:

```text
min_cluster_size = 80
min_samples      = 98
metric           = precomputed
```

### 4. Evaluation

Silhouette score is calculated using only trajectories assigned to non-noise clusters.

The course project also used the following heuristic to balance cluster separation against the fraction of trajectories classified as noise:

```math
\mathrm{Score}
=
0.9 \times \mathrm{Silhouette}
-
0.1 \times \mathrm{NoiseRatio}
```

This score is a project-specific heuristic rather than a standard HDBSCAN evaluation metric.

## Reproducibility Audit

As part of post-project reproducibility validation, several additional checks were performed without changing the original project methodology.

### Resampling

The original flight records have irregular observation intervals, so sample-position resampling and elapsed-time resampling are not equivalent.

However, for 100 reproducibly sampled flight pairs, comparing Hausdorff distance on the original trajectories with the current 300-point resampled trajectories produced:

```text
Median relative difference: 0.263%
Pearson correlation:        0.999085
```

This indicates that the 300-point representation closely preserves the original Hausdorff-distance structure for the tested pairs while reducing computation.

### Geographic distance approximation

Longitude/latitude coordinates are treated as a Euclidean plane in the original project.

A post-project comparison of degree-space Hausdorff distance against great-circle (haversine) Hausdorff distance on 500 reproducibly sampled trajectory pairs produced:

```text
Pearson correlation:       0.999858
Spearman rank correlation: 0.993617
```

The approximation therefore preserved pairwise distance ordering closely for this dataset, although it remains a geographic limitation of the methodology.

### Environment reproduction

The project was reproduced in a clean Python 3.13.3 virtual environment created only from `requirements.txt`.

The clean environment successfully reproduced:

```text
591 loaded trajectories
Cluster 0: 87
Cluster 1: 476
Noise:     28
Score:     0.7781885097560663
```

## Repository Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── clustering.py
│   ├── visualization.py
│   └── main.py
├── data/
│   └── raw/          # local trajectory workbooks; not tracked by Git
├── requirements.txt
├── .gitignore
└── README.md
```

## Data

Raw trajectory workbooks are **not included in this repository**.

The course assignment required the use of publicly available or free-tier ADS-B trajectory sources. The project presentation references FlightAware for the RKSI–WSSS route, but redistribution rights for the local collected workbooks have not been established.

To run the pipeline with the original local dataset, place the following files in:

```text
data/raw/
```

Expected filenames:

```text
ICN_SIN_AAR751_final.xlsx
ICN_SIN_JJA2623_final.xlsx
ICN_SIN_KAL643_final.xlsx
ICN_SIN_KAL645_final.xlsx
ICN_SIN_SIA601_final.xlsx
ICN_SIN_SIA605_final.xlsx
ICN_SIN_TGW843_final.xlsx
ICN_SIN_TWB161_final.xlsx
```

Each workbook is expected to contain trajectory sheets named `Sheet1`, `Sheet2`, etc., with the columns:

```text
Time (KST)
Latitude
Longitude
kts
mph
feet
```

## Setup

The validated environment uses Python 3.13.3.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then create the local data directory and add the required workbooks:

```bash
mkdir -p data/raw
```

Run the clustering pipeline with:

```bash
python -m src.main
```

The pairwise Hausdorff distance matrix is the computationally expensive stage. In one validation run on the development machine, computation for 591 trajectories resampled to 300 points took approximately 4.6 minutes.

## Repository Scope

The current repository reproduces the core HDBSCAN workflow: data loading and preprocessing, fixed-length trajectory resampling, Hausdorff-distance computation, HDBSCAN clustering, evaluation, and trajectory visualization.

The original team presentation also included K-Means re-clustering of HDBSCAN noise trajectories, investigation of individual trajectories with unusual flight-path patterns, and qualitative discussion of possible weather and ATC-related deviations. Those follow-up analyses are not implemented in the current codebase and are therefore not presented here as reproducible outputs.


## Limitations

- Hausdorff point distances are computed directly in longitude/latitude degree space rather than using a geodesic metric.
- Fixed-length preprocessing uses normalized sample position rather than actual elapsed time.
- The project-specific evaluation score is a heuristic, not a standard HDBSCAN metric.
- Raw data are not redistributed, so full reproduction requires access to trajectory files matching the documented schema.
- The current repository does not reproduce every follow-up analysis shown in the original team presentation.
