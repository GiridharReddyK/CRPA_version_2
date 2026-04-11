"""
visualizations.py — Plotting Engine for CRPA Null Steering Dashboard

All visualization logic is encapsulated here to keep the Streamlit
app.py clean. Uses Plotly for interactive 3D and Matplotlib for
publication-quality 2D polar/Cartesian plots.

Colour scheme: dark-theme engineering dashboard palette.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple
from rf_physics import JammerConfig, QuantConfig


# ---------------------------------------------------------------------------
# Colour Constants (Dark Engineering Theme)
# ---------------------------------------------------------------------------

DARK_BG = '#0a0e17'
PANEL_BG = '#101825'
GRID_COLOR = '#1e2d42'
TEXT_COLOR = '#c8d6e5'
ACCENT_CYAN = '#00d2ff'
ACCENT_GREEN = '#00ff88'
ACCENT_RED = '#ff3355'
ACCENT_AMBER = '#ffaa00'
ACCENT_MAGENTA = '#ff44cc'
JAMMER_COLORS = ['#ff3355', '#ff8800', '#ffdd00', '#ff44cc']
WEIGHT_COLORS = [
    '#00d2ff', '#00ff88', '#ff3355', '#ffaa00',
    '#ff44cc', '#8855ff', '#44ddff',
]

# Matplotlib dark style
MPL_RC = {
    'figure.facecolor': DARK_BG,
    'axes.facecolor': PANEL_BG,
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.4,
    'legend.facecolor': PANEL_BG,
    'legend.edgecolor': GRID_COLOR,
    'legend.labelcolor': TEXT_COLOR,
    'font.family': 'monospace',
}


def _apply_dark_style() -> None:
    """Apply the dark engineering theme to matplotlib."""
    plt.rcParams.update(MPL_RC)


def _plotly_dark_layout(fig: go.Figure, title: str = '') -> go.Figure:
    """Apply consistent dark theme to Plotly figures."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=16, family='JetBrains Mono, monospace')),
        font=dict(color=TEXT_COLOR, family='JetBrains Mono, monospace', size=11),
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(bgcolor='rgba(16,24,37,0.8)', bordercolor=GRID_COLOR),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


# ---------------------------------------------------------------------------
# Array Geometry Plot
# ---------------------------------------------------------------------------

def plot_array_geometry(
    positions: NDArray,
    desired_az: float = 0.0,
    jammers: Optional[List[JammerConfig]] = None,
) -> Figure:
    """
    Plot the array element positions with desired and jammer DOAs.

    Parameters
    ----------
    positions  : (N, 3) element coordinates in metres.
    desired_az : desired signal azimuth (degrees).
    jammers    : list of JammerConfig for jammer markers.

    Returns
    -------
    Matplotlib Figure.
    """
    _apply_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Array elements
    x = positions[:, 0] * 100  # convert to cm
    y = positions[:, 1] * 100
    ax.scatter(x, y, s=180, c=ACCENT_CYAN, zorder=5, edgecolors='white',
               linewidths=0.8, marker='h')
    for i in range(len(x)):
        ax.annotate(f'E{i}', (x[i], y[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color=ACCENT_CYAN, fontweight='bold')

    # Radius circle
    r_cm = np.max(np.sqrt(x**2 + y**2))
    if r_cm > 0:
        circle = Circle((0, 0), r_cm, fill=False, edgecolor=GRID_COLOR,
                         linestyle='--', linewidth=1)
        ax.add_patch(circle)

    # Desired signal direction
    arrow_len = r_cm * 1.6 if r_cm > 0 else 15
    az_rad = np.deg2rad(desired_az)
    ax.annotate('', xy=(arrow_len * np.cos(az_rad), arrow_len * np.sin(az_rad)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=ACCENT_GREEN, lw=2.5))
    ax.text(arrow_len * 1.1 * np.cos(az_rad), arrow_len * 1.1 * np.sin(az_rad),
            f'Signal\n({desired_az:.0f}°)', color=ACCENT_GREEN, fontsize=9,
            ha='center', va='center', fontweight='bold')

    # Jammer directions
    if jammers:
        for idx, j in enumerate(jammers):
            color = JAMMER_COLORS[idx % len(JAMMER_COLORS)]
            jaz = np.deg2rad(j.azimuth_deg)
            ax.annotate('', xy=(arrow_len * np.cos(jaz), arrow_len * np.sin(jaz)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                        linestyle='--'))
            ax.text(arrow_len * 1.15 * np.cos(jaz), arrow_len * 1.15 * np.sin(jaz),
                    f'J{idx+1}\n({j.azimuth_deg:.0f}°, {j.jsr_db:.0f}dB)',
                    color=color, fontsize=8, ha='center', va='center', fontweight='bold')

    ax.set_xlim(-arrow_len * 1.5, arrow_len * 1.5)
    ax.set_ylim(-arrow_len * 1.5, arrow_len * 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title('CRPA Array Geometry & Scenario', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2D Polar Pattern Plot
# ---------------------------------------------------------------------------

def plot_polar_pattern(
    az_deg: NDArray,
    pattern_db: NDArray,
    desired_az: float = 0.0,
    jammers: Optional[List[JammerConfig]] = None,
    title: str = 'Adapted Radiation Pattern',
    pattern_db_quant: Optional[NDArray] = None,
    dynamic_range: float = 60.0,
) -> Figure:
    """
    Polar plot of the azimuthal radiation pattern.

    Parameters
    ----------
    az_deg          : (M,) azimuth angles.
    pattern_db      : (M,) pattern in dB (normalised).
    desired_az      : desired signal azimuth for marker.
    jammers         : jammer list for null markers.
    title           : plot title.
    pattern_db_quant: optional quantized pattern overlay.
    dynamic_range   : dB range to display.

    Returns
    -------
    Matplotlib Figure.
    """
    _apply_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(PANEL_BG)

    az_rad = np.deg2rad(az_deg)

    # Clamp to dynamic range
    patt_clamped = np.maximum(pattern_db, -dynamic_range)

    ax.plot(az_rad, patt_clamped, color=ACCENT_CYAN, linewidth=2.0, label='Ideal Weights')
    ax.fill(az_rad, patt_clamped, alpha=0.08, color=ACCENT_CYAN)

    if pattern_db_quant is not None:
        patt_q_clamped = np.maximum(pattern_db_quant, -dynamic_range)
        ax.plot(az_rad, patt_q_clamped, color=ACCENT_AMBER, linewidth=1.5,
                linestyle='--', label='Quantized Weights', alpha=0.85)

    # Desired signal marker
    ax.axvline(np.deg2rad(desired_az), color=ACCENT_GREEN, linestyle='-',
               linewidth=1.5, alpha=0.7, label=f'Desired ({desired_az:.0f}°)')

    # Jammer markers
    if jammers:
        for idx, j in enumerate(jammers):
            color = JAMMER_COLORS[idx % len(JAMMER_COLORS)]
            ax.axvline(np.deg2rad(j.azimuth_deg), color=color, linestyle='--',
                       linewidth=1.5, alpha=0.7,
                       label=f'J{idx+1} ({j.azimuth_deg:.0f}°, {j.jsr_db:.0f}dB)')

    ax.set_ylim(-dynamic_range, 5)
    ax.set_rticks(np.arange(-dynamic_range, 10, 10))
    ax.set_rlabel_position(135)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2D Cartesian Pattern Cut
# ---------------------------------------------------------------------------

def plot_cartesian_pattern(
    az_deg: NDArray,
    pattern_db: NDArray,
    desired_az: float = 0.0,
    jammers: Optional[List[JammerConfig]] = None,
    pattern_db_quant: Optional[NDArray] = None,
    dynamic_range: float = 60.0,
) -> Figure:
    """
    Cartesian dB plot of the azimuthal pattern for detailed null inspection.
    """
    _apply_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    patt_clamped = np.maximum(pattern_db, -dynamic_range)
    ax.plot(az_deg, patt_clamped, color=ACCENT_CYAN, linewidth=2.0, label='Ideal Weights')

    if pattern_db_quant is not None:
        patt_q = np.maximum(pattern_db_quant, -dynamic_range)
        ax.plot(az_deg, patt_q, color=ACCENT_AMBER, linewidth=1.5,
                linestyle='--', label='Quantized Weights', alpha=0.85)

    ax.axvline(desired_az, color=ACCENT_GREEN, linestyle='-', linewidth=1.5,
               alpha=0.7, label=f'Desired ({desired_az:.0f}°)')

    if jammers:
        for idx, j in enumerate(jammers):
            color = JAMMER_COLORS[idx % len(JAMMER_COLORS)]
            ax.axvline(j.azimuth_deg, color=color, linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f'J{idx+1} ({j.azimuth_deg:.0f}°)')

    ax.set_xlim(0, 360)
    ax.set_ylim(-dynamic_range, 5)
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('Normalised Gain (dB)')
    ax.set_title('Pattern Cut — Azimuth Plane', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3D Surface Pattern (Plotly)
# ---------------------------------------------------------------------------

def plot_3d_pattern(
    az_deg: NDArray,
    el_deg: NDArray,
    pattern_db: NDArray,
    dynamic_range: float = 40.0,
    title: str = '3D Radiation Pattern',
) -> go.Figure:
    """
    Interactive 3D spherical radiation pattern using Plotly.

    Converts (az, el, gain) to Cartesian for surface rendering.

    Parameters
    ----------
    az_deg       : (M,) azimuth grid.
    el_deg       : (P,) elevation grid.
    pattern_db   : (P, M) normalised pattern.
    dynamic_range: dB floor for display.
    title        : figure title.

    Returns
    -------
    Plotly Figure.
    """
    AZ, EL = np.meshgrid(az_deg, el_deg)
    patt = np.maximum(pattern_db, -dynamic_range)

    # Map gain to radius (shift so minimum maps to small positive radius)
    r = patt + dynamic_range + 5  # ensure all positive
    r = np.maximum(r, 0.5)

    az_rad = np.deg2rad(AZ)
    el_rad = np.deg2rad(EL)

    X = r * np.cos(el_rad) * np.cos(az_rad)
    Y = r * np.cos(el_rad) * np.sin(az_rad)
    Z = r * np.sin(el_rad)

    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=patt,
        colorscale=[
            [0.0, '#0a0e17'],
            [0.2, '#1a1a5e'],
            [0.4, '#0066cc'],
            [0.6, '#00cc88'],
            [0.8, '#ffaa00'],
            [1.0, '#ff3355'],
        ],
        colorbar=dict(
            title=dict(text='dB', font=dict(color=TEXT_COLOR)),
            tickfont=dict(color=TEXT_COLOR),
        ),
        opacity=0.92,
        showscale=True,
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title='', visible=False),
            yaxis=dict(showgrid=False, showticklabels=False, title='', visible=False),
            zaxis=dict(showgrid=False, showticklabels=False, title='', visible=False),
            bgcolor=DARK_BG,
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.9)),
        ),
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR, family='JetBrains Mono, monospace'),
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        margin=dict(l=10, r=10, t=50, b=10),
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# Weight Vector Visualisation
# ---------------------------------------------------------------------------

def plot_weights(
    w_ideal: NDArray,
    w_quant: Optional[NDArray] = None,
    quant_config: Optional[QuantConfig] = None,
) -> Figure:
    """
    Visualise complex beamformer weights (magnitude & phase).

    Shows bar charts for amplitude (dB) and phase (degrees) with
    ideal vs quantized overlays.
    """
    _apply_dark_style()
    n = len(w_ideal)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    x = np.arange(n)
    bar_w = 0.35

    # Amplitude
    mag_ideal = 20 * np.log10(np.abs(w_ideal) / np.max(np.abs(w_ideal)) + 1e-15)
    ax1.bar(x - bar_w / 2, mag_ideal, bar_w, color=ACCENT_CYAN, alpha=0.85,
            label='Ideal', edgecolor='white', linewidth=0.5)

    if w_quant is not None:
        mag_quant = 20 * np.log10(np.abs(w_quant) / np.max(np.abs(w_quant)) + 1e-15)
        ax1.bar(x + bar_w / 2, mag_quant, bar_w, color=ACCENT_AMBER, alpha=0.85,
                label='Quantized', edgecolor='white', linewidth=0.5)

    ax1.set_ylabel('Relative Amplitude (dB)')
    ax1.set_title('Weight Vector — Amplitude', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'E{i}' for i in range(n)])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Phase
    phase_ideal = np.rad2deg(np.angle(w_ideal))
    ax2.bar(x - bar_w / 2, phase_ideal, bar_w, color=ACCENT_CYAN, alpha=0.85,
            label='Ideal', edgecolor='white', linewidth=0.5)

    if w_quant is not None:
        phase_quant = np.rad2deg(np.angle(w_quant))
        ax2.bar(x + bar_w / 2, phase_quant, bar_w, color=ACCENT_AMBER, alpha=0.85,
                label='Quantized', edgecolor='white', linewidth=0.5)

    ax2.set_ylabel('Phase (°)')
    ax2.set_xlabel('Element')
    ax2.set_title('Weight Vector — Phase', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'E{i}' for i in range(n)])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    if quant_config is not None and w_quant is not None:
        phase_step = 360.0 / (2 ** quant_config.phase_bits)
        ax2.axhline(phase_step, color=GRID_COLOR, linestyle=':', alpha=0.5)
        ax2.axhline(-phase_step, color=GRID_COLOR, linestyle=':', alpha=0.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Wideband Squint Plot
# ---------------------------------------------------------------------------

def plot_wideband_squint(
    freq_offsets_mhz: NDArray,
    null_depths: dict[str, NDArray],
    title: str = 'Wideband Null Dispersion (Squint)',
) -> go.Figure:
    """
    Plot null depth vs frequency offset for each jammer.

    Parameters
    ----------
    freq_offsets_mhz : (K,) offset from centre in MHz.
    null_depths      : dict mapping jammer label to (K,) null depth in dB.
    title            : figure title.

    Returns
    -------
    Plotly Figure.
    """
    fig = go.Figure()

    colors = JAMMER_COLORS
    for idx, (label, depth) in enumerate(null_depths.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=freq_offsets_mhz,
            y=depth,
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=4, color=color),
        ))

    # Reference lines
    fig.add_hline(y=-20, line_dash='dot', line_color=ACCENT_GREEN,
                  annotation_text='-20 dB threshold',
                  annotation_font_color=ACCENT_GREEN)
    fig.add_hline(y=-30, line_dash='dot', line_color=ACCENT_AMBER,
                  annotation_text='-30 dB threshold',
                  annotation_font_color=ACCENT_AMBER)
    fig.add_vline(x=0, line_dash='solid', line_color=GRID_COLOR, line_width=1)

    fig = _plotly_dark_layout(fig, title)
    fig.update_xaxes(title_text='Frequency Offset (MHz)')
    fig.update_yaxes(title_text='Null Depth (dB)')
    fig.update_layout(height=450)
    return fig


# ---------------------------------------------------------------------------
# S-Parameter / Coupling Matrix Heatmap
# ---------------------------------------------------------------------------

def plot_coupling_matrix(
    matrix: NDArray,
    title: str = 'Mutual Coupling Matrix |C|',
    show_phase: bool = False,
) -> Figure:
    """
    Heatmap of the coupling matrix magnitude (or phase).
    """
    _apply_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    data = np.angle(matrix, deg=True) if show_phase else 20 * np.log10(np.abs(matrix) + 1e-15)
    label = 'Phase (°)' if show_phase else '|C| (dB)'

    im = ax.imshow(data, cmap='inferno', aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

    n = matrix.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'E{i}' for i in range(n)])
    ax.set_yticklabels([f'E{i}' for i in range(n)])
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            txt_color = 'white' if val < np.median(data) else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=8, color=txt_color)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# AEP Pattern Visualization
# ---------------------------------------------------------------------------

def plot_aep_pattern(
    theta_deg: NDArray,
    phi_deg: NDArray,
    magnitude: NDArray,
    element_index: int = 0,
) -> Figure:
    """
    2D colour map of an element's Active Element Pattern.
    """
    _apply_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    im = ax.pcolormesh(
        phi_deg, theta_deg, magnitude,
        cmap='viridis', shading='auto',
    )
    plt.colorbar(im, ax=ax, label='Normalised Gain (linear)')
    ax.set_xlabel('Phi (°)')
    ax.set_ylabel('Theta (°)')
    ax.set_title(f'Active Element Pattern — Element {element_index}',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Theta=0 (zenith) at top
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics Summary (Plotly Table)
# ---------------------------------------------------------------------------

def create_metrics_table(
    metrics: dict[str, str],
) -> go.Figure:
    """
    Compact metrics table rendered as a Plotly table for dark dashboards.
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Parameter</b>', '<b>Value</b>'],
            fill_color='#1a2332',
            font=dict(color=TEXT_COLOR, size=12, family='JetBrains Mono'),
            align='left',
            line=dict(color=GRID_COLOR, width=1),
        ),
        cells=dict(
            values=[list(metrics.keys()), list(metrics.values())],
            fill_color=[['#101825'] * len(metrics)],
            font=dict(color=[ACCENT_CYAN, ACCENT_GREEN], size=11,
                       family='JetBrains Mono'),
            align='left',
            line=dict(color=GRID_COLOR, width=1),
            height=28,
        ),
    )])
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        margin=dict(l=5, r=5, t=5, b=5),
        height=35 + 28 * len(metrics),
    )
    return fig
