"""Data loading and trajectory preprocessing utilities."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

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
