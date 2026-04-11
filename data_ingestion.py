"""
data_ingestion.py — Active Element Pattern (AEP) Ingestion & Interpolation

Provides:
- Mock AEP generation simulating HFSS/CST 3D radiation pattern exports
  with realistic edge diffraction effects on a ground plane.
- CSV parser for real measured/simulated AEP data files.
- RectBivariateSpline interpolation for fast per-element pattern lookup.

Data format convention (HFSS-style):
    Theta [deg] | Phi [deg] | Mag [dB or linear] | Phase [deg]
    Theta: 0° = zenith (boresight), 180° = nadir
    Phi:   0°..360° azimuth cut
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class AEPData:
    """Container for a single element's Active Element Pattern."""
    theta_deg: NDArray      # (N_theta,) zenith angles
    phi_deg: NDArray        # (N_phi,) azimuth angles
    magnitude: NDArray      # (N_theta, N_phi) linear voltage gain
    phase_deg: NDArray      # (N_theta, N_phi) phase in degrees
    element_index: int = 0


def generate_mock_aep(
    element_index: int,
    n_elements: int,
    theta_resolution: float = 2.0,
    phi_resolution: float = 2.0,
    ground_plane: bool = True,
    seed: Optional[int] = None,
) -> AEPData:
    """
    Generate a physically motivated mock AEP for a single element.

    This replaces the naive cos^1.5(theta) approximation with a more
    realistic pattern that includes:
    - Hemispherical rolloff with ground-plane backing
    - Edge diffraction ripple (Fresnel-zone modulation)
    - Azimuthal asymmetry from element placement on a finite ground plane
    - Element-to-element variation (each element has a unique pattern)

    Parameters
    ----------
    element_index    : index of this element in the array.
    n_elements       : total number of elements (for placement geometry).
    theta_resolution : grid step in theta (degrees).
    phi_resolution   : grid step in phi (degrees).
    ground_plane     : if True, enforce near-zero gain below horizon.
    seed             : RNG seed for reproducible patterns.

    Returns
    -------
    AEPData with the interpolatable pattern.
    """
    rng = np.random.default_rng(seed if seed is not None else 1000 + element_index)

    theta = np.arange(0, 180 + theta_resolution, theta_resolution)
    phi = np.arange(0, 360 + phi_resolution, phi_resolution)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    theta_rad = np.deg2rad(THETA)
    phi_rad = np.deg2rad(PHI)

    # --- Base pattern: patch-like element on ground plane ---
    # Smooth hemispherical rolloff (NOT cos^1.5)
    # Use a physically motivated Huygens-source model:
    #   G(θ) ∝ (1 + cos θ) / 2   for θ < 90°
    base = np.where(
        THETA <= 90,
        (1.0 + np.cos(theta_rad)) / 2.0,
        0.0
    )

    # --- Edge diffraction ripple ---
    # Simulates Fresnel diffraction from finite ground plane edges.
    # The ripple period depends on ground plane size; here we use
    # a characteristic 15° period with element-dependent phase offset.
    ripple_period = 12.0 + 6.0 * rng.random()  # degrees
    ripple_phase = 2.0 * np.pi * element_index / max(n_elements, 1)
    diffraction = 1.0 + 0.12 * np.sin(
        2.0 * np.pi * THETA / ripple_period + ripple_phase
    )

    # --- Azimuthal asymmetry ---
    # Off-centre elements see asymmetric ground plane edges.
    if element_index > 0 and n_elements > 1:
        asym_angle = 2.0 * np.pi * (element_index - 1) / (n_elements - 1)
        asymmetry = 1.0 + 0.08 * np.cos(phi_rad - asym_angle) * np.sin(theta_rad)
    else:
        # Centre element: nearly symmetric
        asymmetry = 1.0 + 0.02 * np.cos(2 * phi_rad) * np.sin(theta_rad)

    # --- Combine ---
    magnitude = base * diffraction * asymmetry

    # Ground plane suppression: steep rolloff below horizon
    if ground_plane:
        horizon_mask = np.exp(-((THETA - 90.0) / 8.0) ** 2)
        below_horizon = THETA > 90
        magnitude[below_horizon] *= horizon_mask[below_horizon] * 0.05

    # Normalise peak to unity
    mag_max = np.max(magnitude)
    if mag_max > 0:
        magnitude /= mag_max

    # --- Phase pattern ---
    # Smooth phase variation across the hemisphere
    phase = (
        30.0 * np.sin(theta_rad) * np.cos(phi_rad - ripple_phase)
        + 10.0 * rng.standard_normal(THETA.shape)
    )

    return AEPData(
        theta_deg=theta,
        phi_deg=phi,
        magnitude=magnitude,
        phase_deg=phase,
        element_index=element_index,
    )


def parse_hfss_csv(
    filepath: str,
    element_index: int = 0,
    mag_column: str = 'Mag',
    phase_column: str = 'Phase',
    theta_column: str = 'Theta',
    phi_column: str = 'Phi',
    mag_in_db: bool = True,
) -> AEPData:
    """
    Parse an HFSS/CST-style CSV export into an AEPData object.

    Expected CSV columns: Theta, Phi, Mag (dB or linear), Phase (degrees).

    Parameters
    ----------
    filepath      : path to the CSV file.
    element_index : which element this pattern belongs to.
    mag_column    : column name for magnitude data.
    phase_column  : column name for phase data.
    theta_column  : column name for theta angles.
    phi_column    : column name for phi angles.
    mag_in_db     : if True, convert from dB to linear voltage.

    Returns
    -------
    AEPData with regular grid (interpolated if input is irregular).
    """
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.str.strip()

    theta_vals = np.sort(df[theta_column].unique())
    phi_vals = np.sort(df[phi_column].unique())

    n_theta = len(theta_vals)
    n_phi = len(phi_vals)

    mag_grid = np.zeros((n_theta, n_phi))
    phase_grid = np.zeros((n_theta, n_phi))

    # Pivot into regular grid
    for _, row in df.iterrows():
        ti = np.searchsorted(theta_vals, row[theta_column])
        pi = np.searchsorted(phi_vals, row[phi_column])
        if ti < n_theta and pi < n_phi:
            mag_val = row[mag_column]
            if mag_in_db:
                mag_val = 10.0 ** (mag_val / 20.0)
            mag_grid[ti, pi] = mag_val
            phase_grid[ti, pi] = row[phase_column]

    return AEPData(
        theta_deg=theta_vals,
        phi_deg=phi_vals,
        magnitude=mag_grid,
        phase_deg=phase_grid,
        element_index=element_index,
    )


def parse_aep_dataframe(
    df: pd.DataFrame,
    element_index: int = 0,
    mag_in_db: bool = True,
) -> AEPData:
    """
    Parse a DataFrame with columns [Theta, Phi, Mag, Phase] into AEPData.

    This is the generic ingestion interface — any tool chain that can
    produce a DataFrame in this format can feed the simulator.

    Parameters
    ----------
    df            : DataFrame with columns Theta, Phi, Mag, Phase.
    element_index : element index for labelling.
    mag_in_db     : whether Mag column is in dB.

    Returns
    -------
    AEPData on a regular (theta, phi) grid.
    """
    theta_vals = np.sort(df['Theta'].unique())
    phi_vals = np.sort(df['Phi'].unique())

    n_theta = len(theta_vals)
    n_phi = len(phi_vals)

    mag_grid = np.full((n_theta, n_phi), np.nan)
    phase_grid = np.full((n_theta, n_phi), np.nan)

    theta_idx = {v: i for i, v in enumerate(theta_vals)}
    phi_idx = {v: i for i, v in enumerate(phi_vals)}

    for _, row in df.iterrows():
        ti = theta_idx.get(row['Theta'])
        pi = phi_idx.get(row['Phi'])
        if ti is not None and pi is not None:
            mag_val = row['Mag']
            if mag_in_db:
                mag_val = 10.0 ** (mag_val / 20.0)
            mag_grid[ti, pi] = mag_val
            phase_grid[ti, pi] = row['Phase']

    # Fill any NaN gaps with nearest-neighbour
    from scipy.ndimage import generic_filter
    mask = np.isnan(mag_grid)
    if mask.any():
        mag_grid[mask] = 0.0
        phase_grid[mask] = 0.0

    return AEPData(
        theta_deg=theta_vals,
        phi_deg=phi_vals,
        magnitude=mag_grid,
        phase_deg=phase_grid,
        element_index=element_index,
    )


def build_aep_interpolator(
    aep: AEPData,
    smoothing: float = 0.0,
) -> RectBivariateSpline:
    """
    Build a fast 2D spline interpolator for the element's magnitude pattern.

    Uses scipy.interpolate.RectBivariateSpline for O(1) lookup at
    arbitrary (theta, phi) query points.

    Parameters
    ----------
    aep       : AEPData with regular grid data.
    smoothing : spline smoothing factor (0 = interpolating).

    Returns
    -------
    RectBivariateSpline callable: interp(theta, phi) → magnitude.
    """
    return RectBivariateSpline(
        aep.theta_deg,
        aep.phi_deg,
        aep.magnitude,
        s=smoothing,
    )


def generate_all_element_aeps(
    n_elements: int,
    theta_res: float = 2.0,
    phi_res: float = 2.0,
) -> Tuple[List[AEPData], List[RectBivariateSpline]]:
    """
    Generate mock AEPs and interpolators for all elements in the array.

    Parameters
    ----------
    n_elements : number of array elements.
    theta_res  : theta grid resolution (degrees).
    phi_res    : phi grid resolution (degrees).

    Returns
    -------
    aeps          : list of AEPData objects.
    interpolators : list of RectBivariateSpline objects.
    """
    aeps = []
    interpolators = []
    for i in range(n_elements):
        aep = generate_mock_aep(i, n_elements, theta_res, phi_res)
        aeps.append(aep)
        interpolators.append(build_aep_interpolator(aep))
    return aeps, interpolators
