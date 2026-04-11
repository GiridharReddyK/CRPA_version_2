"""
rf_physics.py — Core RF Signal Processing Engine for CRPA Null Steering

Implements physically-grounded RF math including:
- Ideal and coupled steering vector generation
- S-parameter to coupling matrix conversion
- Sample Matrix Inversion (SMI) covariance-based nulling
- Hardware quantization (phase shifter + attenuator discretization)
- Wideband null dispersion (squint) analysis

All formulations follow standard phased-array DSP conventions.
Reference: Van Trees, "Optimum Array Processing", Wiley, 2002.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy.interpolate import RectBivariateSpline


# ---------------------------------------------------------------------------
# Configuration Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ArrayConfig:
    """Physical array configuration."""
    n_elements: int = 7
    radius_m: float = 0.095  # ~λ/2 at L1 for 7-element CRPA
    freq_hz: float = 1575.42e6  # GPS L1
    element_positions_m: Optional[NDArray] = None  # (N, 3) xyz

    def __post_init__(self) -> None:
        if self.element_positions_m is None:
            self.element_positions_m = self._default_circular_array()

    def _default_circular_array(self) -> NDArray:
        """Generate a canonical circular array with a centre element."""
        positions = np.zeros((self.n_elements, 3))
        if self.n_elements == 1:
            return positions
        # Element 0 at centre
        n_ring = self.n_elements - 1
        angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        positions[1:, 0] = self.radius_m * np.cos(angles)
        positions[1:, 1] = self.radius_m * np.sin(angles)
        return positions


@dataclass
class JammerConfig:
    """Single jammer source descriptor."""
    azimuth_deg: float = 90.0
    elevation_deg: float = 30.0
    jsr_db: float = 40.0  # Jammer-to-signal ratio


@dataclass
class QuantConfig:
    """Hardware quantization parameters."""
    phase_bits: int = 6           # Phase shifter resolution
    amp_step_db: float = 0.5      # Attenuator LSB
    amp_range_db: float = 31.5    # Max attenuation depth
    enabled: bool = True


@dataclass
class ScenarioConfig:
    """Full scenario descriptor for the SMI engine."""
    array: ArrayConfig = field(default_factory=ArrayConfig)
    jammers: list[JammerConfig] = field(default_factory=list)
    quant: QuantConfig = field(default_factory=QuantConfig)
    desired_az_deg: float = 0.0
    desired_el_deg: float = 90.0
    noise_floor_dbm: float = -114.0  # kTB at GPS BW ≈ 2 MHz
    signal_power_dbm: float = -130.0  # Typical GPS received power


# ---------------------------------------------------------------------------
# Physical Constants & Helpers
# ---------------------------------------------------------------------------

C_LIGHT: float = 299_792_458.0  # m/s


def _deg2rad(deg: float | NDArray) -> float | NDArray:
    return np.deg2rad(deg)


def wavelength(freq_hz: float) -> float:
    """Free-space wavelength in metres."""
    return C_LIGHT / freq_hz


def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to linear watts."""
    return 10.0 ** ((dbm - 30.0) / 10.0)


def db_to_linear(db: float) -> float:
    """Convert dB ratio to linear power ratio."""
    return 10.0 ** (db / 10.0)


# ---------------------------------------------------------------------------
# Steering Vector Generation
# ---------------------------------------------------------------------------

def direction_cosines(
    az_deg: float, el_deg: float
) -> NDArray:
    """
    Convert azimuth/elevation to a unit direction vector (3,).

    Convention:
        az: angle in the xy-plane from +x, CCW positive.
        el: angle from the xy-plane toward +z (zenith = 90°).
    """
    az = _deg2rad(az_deg)
    el = _deg2rad(el_deg)
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])


def ideal_steering_vector(
    positions: NDArray,
    freq_hz: float,
    az_deg: float,
    el_deg: float,
) -> NDArray:
    """
    Compute the ideal (no coupling) steering vector for a planar wave.

    Parameters
    ----------
    positions : (N, 3) element coordinates in metres.
    freq_hz   : carrier frequency.
    az_deg    : source azimuth (degrees).
    el_deg    : source elevation (degrees).

    Returns
    -------
    a : (N,) complex steering vector.
    """
    k = 2.0 * np.pi * freq_hz / C_LIGHT
    d = direction_cosines(az_deg, el_deg)
    phase_shifts = positions @ d  # (N,)
    return np.exp(1j * k * phase_shifts)


def steering_vector_with_aep(
    positions: NDArray,
    freq_hz: float,
    az_deg: float,
    el_deg: float,
    aep_interpolators: Optional[list[RectBivariateSpline]] = None,
) -> NDArray:
    """
    Steering vector modulated by per-element Active Element Patterns.

    Each element's complex gain at (az, el) is looked up from its
    interpolated AEP and multiplied onto the ideal phase term.
    """
    a_ideal = ideal_steering_vector(positions, freq_hz, az_deg, el_deg)
    if aep_interpolators is None:
        return a_ideal

    n = len(a_ideal)
    theta = 90.0 - el_deg  # Convert elevation to theta (zenith angle)
    phi = az_deg % 360.0

    gains = np.ones(n, dtype=complex)
    for i, interp in enumerate(aep_interpolators):
        if interp is not None:
            mag = float(interp(theta, phi, grid=False))
            gains[i] = mag  # Phase from AEP can be added similarly
    return a_ideal * gains


# ---------------------------------------------------------------------------
# Mutual Coupling Matrix
# ---------------------------------------------------------------------------

def s_to_coupling_matrix(s_matrix: NDArray) -> NDArray:
    """
    Convert an NxN S-parameter matrix to the mutual coupling matrix.

    C = (I + S)(I - S)^{-1}

    This transforms port-domain scattering to element-domain coupling
    so that the effective steering vector becomes a_coupled = C @ a_ideal.

    Parameters
    ----------
    s_matrix : (N, N) complex S-parameter matrix.

    Returns
    -------
    C : (N, N) complex coupling matrix.
    """
    n = s_matrix.shape[0]
    I = np.eye(n, dtype=complex)
    return (I + s_matrix) @ np.linalg.inv(I - s_matrix)


def generate_mock_s_matrix(
    n_elements: int,
    coupling_mag: float = 0.15,
    isolation_db: float = -20.0,
    seed: int = 42,
) -> NDArray:
    """
    Generate a physically plausible mock S-parameter matrix.

    Diagonal (return loss) is set from isolation_db.
    Off-diagonal coupling decays with element index distance,
    mimicking a circular array's geometric coupling falloff.

    Parameters
    ----------
    n_elements   : number of array elements.
    coupling_mag : baseline mutual coupling magnitude for adjacent elements.
    isolation_db : diagonal element return loss in dB (e.g., -20 dB).
    seed         : RNG seed for reproducible phase scatter.

    Returns
    -------
    S : (N, N) complex, symmetric S-parameter matrix.
    """
    rng = np.random.default_rng(seed)
    S = np.zeros((n_elements, n_elements), dtype=complex)

    # Diagonal: return loss
    rl_mag = 10.0 ** (isolation_db / 20.0)
    for i in range(n_elements):
        S[i, i] = rl_mag * np.exp(1j * rng.uniform(-np.pi, np.pi))

    # Off-diagonal: mutual coupling with distance-based decay
    for i in range(n_elements):
        for j in range(i + 1, n_elements):
            # Circular distance metric
            dist = min(abs(i - j), n_elements - abs(i - j))
            mag = coupling_mag / (dist ** 0.8)
            phase = rng.uniform(-np.pi, np.pi)
            S[i, j] = mag * np.exp(1j * phase)
            S[j, i] = S[i, j]  # Reciprocal network

    return S


def coupled_steering_vector(
    coupling_matrix: NDArray,
    a_ideal: NDArray,
) -> NDArray:
    """
    Apply mutual coupling to an ideal steering vector.

    a_coupled = C @ a_ideal
    """
    return coupling_matrix @ a_ideal


# ---------------------------------------------------------------------------
# SMI Covariance Engine
# ---------------------------------------------------------------------------

def build_covariance_matrix(
    positions: NDArray,
    freq_hz: float,
    jammers: list[JammerConfig],
    noise_floor_dbm: float,
    signal_power_dbm: float,
    coupling_matrix: Optional[NDArray] = None,
    aep_interpolators: Optional[list[RectBivariateSpline]] = None,
) -> NDArray:
    """
    Synthesize the spatial covariance matrix R for SMI beamforming.

    R = Σ_k P_k a_k a_k^H  +  σ_n² I

    where P_k is the jammer power (derived from JSR and signal power)
    and σ_n² is the thermal noise floor.

    Parameters
    ----------
    positions        : (N, 3) element positions.
    freq_hz          : carrier frequency.
    jammers          : list of JammerConfig descriptors.
    noise_floor_dbm  : receiver thermal noise in dBm.
    signal_power_dbm : desired signal power in dBm.
    coupling_matrix  : optional (N, N) coupling matrix.
    aep_interpolators: optional per-element AEP interpolators.

    Returns
    -------
    R : (N, N) Hermitian positive-definite covariance matrix.
    """
    n = positions.shape[0]
    sigma_n_sq = dbm_to_watts(noise_floor_dbm)
    R = sigma_n_sq * np.eye(n, dtype=complex)

    p_signal = dbm_to_watts(signal_power_dbm)

    for jammer in jammers:
        p_jammer = p_signal * db_to_linear(jammer.jsr_db)
        a = steering_vector_with_aep(
            positions, freq_hz, jammer.azimuth_deg, jammer.elevation_deg,
            aep_interpolators
        )
        if coupling_matrix is not None:
            a = coupled_steering_vector(coupling_matrix, a)
        R += p_jammer * np.outer(a, a.conj())

    return R


def smi_weights(
    R: NDArray,
    a_desired: NDArray,
    diagonal_loading_db: float = 0.0,
) -> NDArray:
    """
    Compute SMI (Minimum Variance Distortionless Response) weights.

    w = R^{-1} a_d / (a_d^H R^{-1} a_d)

    Optionally applies diagonal loading for robustness.

    Parameters
    ----------
    R                 : (N, N) covariance matrix.
    a_desired         : (N,) desired-signal steering vector.
    diagonal_loading_db : loading level in dB above noise floor.

    Returns
    -------
    w : (N,) complex weight vector (normalised for unit gain on desired).
    """
    n = R.shape[0]
    if diagonal_loading_db > 0:
        loading = db_to_linear(diagonal_loading_db) * np.min(np.abs(np.diag(R)))
        R = R + loading * np.eye(n, dtype=complex)

    R_inv = np.linalg.inv(R)
    w = R_inv @ a_desired
    # MVDR normalisation
    w = w / (a_desired.conj() @ w)
    return w


# ---------------------------------------------------------------------------
# Hardware Quantization Layer
# ---------------------------------------------------------------------------

def quantize_weights(
    w: NDArray,
    quant: QuantConfig,
) -> NDArray:
    """
    Apply hardware-realistic quantization to complex beamformer weights.

    Phase is quantized to nearest Δφ = 360°/2^B.
    Amplitude is quantized to nearest step in dB, clamped to dynamic range.

    Parameters
    ----------
    w     : (N,) complex weight vector.
    quant : QuantConfig with bit depth and step sizes.

    Returns
    -------
    w_q : (N,) quantized complex weight vector.
    """
    if not quant.enabled:
        return w.copy()

    # Extract magnitude and phase
    mag = np.abs(w)
    phase_rad = np.angle(w)

    # --- Phase quantization ---
    n_phase_states = 2 ** quant.phase_bits
    phase_step_rad = 2.0 * np.pi / n_phase_states
    phase_q = np.round(phase_rad / phase_step_rad) * phase_step_rad

    # --- Amplitude quantization ---
    # Normalise magnitudes so peak = 0 dB attenuation
    mag_max = np.max(mag)
    if mag_max < 1e-30:
        return np.zeros_like(w)

    mag_norm = mag / mag_max
    # Convert to dB attenuation (always ≤ 0)
    with np.errstate(divide='ignore'):
        mag_db = 20.0 * np.log10(np.maximum(mag_norm, 1e-15))

    # Clamp to dynamic range
    mag_db = np.maximum(mag_db, -quant.amp_range_db)

    # Quantize to step grid
    mag_db_q = np.round(mag_db / quant.amp_step_db) * quant.amp_step_db

    # Back to linear
    mag_q = mag_max * 10.0 ** (mag_db_q / 20.0)

    return mag_q * np.exp(1j * phase_q)


def quantization_error_db(w_ideal: NDArray, w_quant: NDArray) -> float:
    """RMS weight error in dB between ideal and quantized weight vectors."""
    err = np.linalg.norm(w_ideal - w_quant)
    ref = np.linalg.norm(w_ideal)
    if ref < 1e-30:
        return -np.inf
    return 20.0 * np.log10(err / ref)


# ---------------------------------------------------------------------------
# Radiation Pattern Evaluation (Vectorised)
# ---------------------------------------------------------------------------

def evaluate_pattern_2d(
    w: NDArray,
    positions: NDArray,
    freq_hz: float,
    az_sweep_deg: NDArray,
    el_deg: float = 90.0,
    coupling_matrix: Optional[NDArray] = None,
    aep_interpolators: Optional[list[RectBivariateSpline]] = None,
) -> NDArray:
    """
    Evaluate the array factor over an azimuth sweep at fixed elevation.

    AF(az) = |w^H a(az)|²   [linear power]

    Vectorised over all azimuth samples.

    Parameters
    ----------
    w             : (N,) complex weight vector.
    positions     : (N, 3) element coordinates.
    freq_hz       : carrier frequency.
    az_sweep_deg  : (M,) azimuth angles in degrees.
    el_deg        : fixed elevation angle.
    coupling_matrix : optional coupling matrix.
    aep_interpolators : optional AEP interpolators.

    Returns
    -------
    pattern_db : (M,) array factor power in dB.
    """
    k = 2.0 * np.pi * freq_hz / C_LIGHT

    # Build steering matrix: (N, M)
    az_rad = _deg2rad(az_sweep_deg)
    el_rad = _deg2rad(el_deg)
    cos_el = np.cos(el_rad)
    sin_el = np.sin(el_rad)

    # Direction vectors: (3, M)
    dirs = np.stack([
        cos_el * np.cos(az_rad),
        cos_el * np.sin(az_rad),
        sin_el * np.ones_like(az_rad),
    ], axis=0)

    # Phase matrix: (N, M)
    phase_matrix = k * (positions @ dirs)
    A = np.exp(1j * phase_matrix)  # (N, M)

    # Apply AEP modulation if available
    if aep_interpolators is not None:
        theta = 90.0 - el_deg
        for i, interp in enumerate(aep_interpolators):
            if interp is not None:
                phi_vals = az_sweep_deg % 360.0
                theta_arr = np.full_like(phi_vals, theta)
                gains = interp(theta_arr, phi_vals, grid=False)
                A[i, :] *= gains

    # Apply coupling
    if coupling_matrix is not None:
        A = coupling_matrix @ A  # (N, N) @ (N, M) = (N, M)

    # Array factor
    af = w.conj() @ A  # (M,)
    power = np.abs(af) ** 2
    power_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    # Normalise to peak
    power_db -= np.max(power_db)
    return power_db


def evaluate_pattern_3d(
    w: NDArray,
    positions: NDArray,
    freq_hz: float,
    az_grid_deg: NDArray,
    el_grid_deg: NDArray,
    coupling_matrix: Optional[NDArray] = None,
    aep_interpolators: Optional[list[RectBivariateSpline]] = None,
) -> NDArray:
    """
    Evaluate the full 3D pattern on an (az, el) meshgrid.

    Parameters
    ----------
    az_grid_deg : (M,) azimuth sample points.
    el_grid_deg : (P,) elevation sample points.

    Returns
    -------
    pattern_db : (P, M) normalised pattern in dB.
    """
    k = 2.0 * np.pi * freq_hz / C_LIGHT
    az_rad = _deg2rad(az_grid_deg)
    el_rad = _deg2rad(el_grid_deg)

    AZ, EL = np.meshgrid(az_rad, el_rad)  # (P, M)
    # Direction cosines on the grid
    dx = np.cos(EL) * np.cos(AZ)
    dy = np.cos(EL) * np.sin(AZ)
    dz = np.sin(EL)

    # Stack to (P*M, 3)
    dirs_flat = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=-1)

    # Phase: (N, P*M)
    phase_matrix = k * (positions @ dirs_flat.T)
    A = np.exp(1j * phase_matrix)

    if coupling_matrix is not None:
        A = coupling_matrix @ A

    af = w.conj() @ A
    power = np.abs(af.reshape(EL.shape)) ** 2
    power_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    power_db -= np.max(power_db)
    return power_db


# ---------------------------------------------------------------------------
# Wideband Null Dispersion (Squint) Analysis
# ---------------------------------------------------------------------------

def wideband_null_response(
    w_narrowband: NDArray,
    positions: NDArray,
    f_center_hz: float,
    f_offsets_hz: NDArray,
    jammer_az_deg: float,
    jammer_el_deg: float,
    coupling_matrix_fc: Optional[NDArray] = None,
    aep_interpolators: Optional[list[RectBivariateSpline]] = None,
) -> NDArray:
    """
    Evaluate narrowband null weights across a frequency sweep.

    P_null(f) = |w(fc)^H  a_coupled(f)|²

    This reveals how bandwidth degrades narrowband nulls ("squint").

    Parameters
    ----------
    w_narrowband      : (N,) weights computed at f_center.
    positions         : (N, 3) element positions.
    f_center_hz       : centre frequency.
    f_offsets_hz      : (K,) frequency offsets from centre.
    jammer_az_deg     : jammer azimuth.
    jammer_el_deg     : jammer elevation.
    coupling_matrix_fc: coupling matrix at fc (reused for simplicity).
    aep_interpolators : optional AEP interpolators.

    Returns
    -------
    null_depth_db : (K,) null depth in dB at each frequency.
    """
    freqs = f_center_hz + f_offsets_hz
    null_power = np.zeros(len(freqs))

    for i, f in enumerate(freqs):
        a = steering_vector_with_aep(
            positions, f, jammer_az_deg, jammer_el_deg, aep_interpolators
        )
        if coupling_matrix_fc is not None:
            a = coupled_steering_vector(coupling_matrix_fc, a)
        null_power[i] = np.abs(w_narrowband.conj() @ a) ** 2

    null_db = 10.0 * np.log10(np.maximum(null_power, 1e-30))
    # Normalise to desired signal response
    return null_db
