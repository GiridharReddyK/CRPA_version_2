"""
app.py — CRPA Null Steering Dashboard (Production-Grade v2)
==========================================================

CORRECTED VERSION – Fixes TypeError in projection_matrix_weights by replacing
AEP interpolators with RectBivariateSpline (compatible with grid=False).

"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy.interpolate import RectBivariateSpline  # <-- added for AEP fix

from rf_physics import (
    ArrayConfig, ArrayTopology, JammerConfig, QuantConfig, STAPConfig,
    PolarisationConfig, ScenarioConfig, NullingAlgorithm, Polarisation,
    wavelength, ideal_steering_vector, steering_vector_with_aep,
    coupled_steering_vector, generate_mock_s_matrix, s_to_coupling_matrix,
    build_covariance_matrix, smi_weights, projection_matrix_weights,
    compute_weights, quantize_weights, quantization_error_db,
    evaluate_pattern_2d, evaluate_pattern_3d, wideband_null_response,
    polarisation_mismatch_loss_db, polarisation_jones_vector,
    axial_ratio_from_jones,
    build_stap_steering_vector, stap_weights, stap_null_response,
    dbm_to_watts, HAS_NUMBA, HAS_SKRF,
)
from data_ingestion import (
    generate_mock_aep, generate_all_element_aeps,
    build_aep_interpolator, AEPData,
)
from visualizations import (
    plot_array_geometry, plot_polar_pattern, plot_cartesian_pattern,
    plot_3d_pattern, plot_weights, plot_wideband_squint,
    plot_coupling_matrix, plot_aep_pattern, create_metrics_table,
    plot_polarisation_response, plot_stap_weights,
    DARK_BG, ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    TEXT_COLOR, PANEL_BG,
)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title='CRPA Null Steering Dashboard',
    page_icon='📡',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(f"""
<style>
    .stApp {{
        background-color: {DARK_BG};
        color: {TEXT_COLOR};
    }}
    section[data-testid="stSidebar"] {{
        background-color: #0d1220;
        border-right: 1px solid #1e2d42;
    }}
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {ACCENT_CYAN};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background-color: #0d1220;
        border-radius: 8px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {PANEL_BG};
        border-radius: 6px;
        color: {TEXT_COLOR};
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #1a3355 !important;
        border-bottom: 2px solid {ACCENT_CYAN} !important;
    }}
    div[data-testid="stMetric"] {{
        background-color: {PANEL_BG};
        border: 1px solid #1e2d42;
        border-radius: 8px;
        padding: 12px 16px;
    }}
    div[data-testid="stMetric"] label {{
        color: #7a8ba3;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {ACCENT_CYAN};
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }}
    h1 {{
        font-family: 'JetBrains Mono', monospace !important;
        color: {ACCENT_CYAN} !important;
        letter-spacing: 2px;
    }}
    .stSlider > div > div > div > div {{
        background-color: {ACCENT_CYAN} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CACHED COMPUTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def cached_generate_aeps(n_elements: int, theta_res: float, phi_res: float):
    """Generate and cache AEP data for all elements."""
    aeps, _ = generate_all_element_aeps(n_elements, theta_res, phi_res)
    return [
        {
            'theta': aep.theta_deg, 'phi': aep.phi_deg,
            'magnitude': aep.magnitude, 'phase': aep.phase_deg,
            'index': aep.element_index,
        }
        for aep in aeps
    ]


@st.cache_data
def cached_s_matrix(n_elements: int, coupling_mag: float, isolation_db: float):
    return generate_mock_s_matrix(n_elements, coupling_mag, isolation_db)


@st.cache_data
def cached_3d_pattern(
    _w_real, _w_imag, _pos_tuple, n_elements, freq_hz, az_pts, el_pts,
    _c_real, _c_imag,
):
    """Cache-friendly 3D pattern evaluation with chunked compute."""
    w = np.array(_w_real) + 1j * np.array(_w_imag)
    positions = np.array(_pos_tuple).reshape(n_elements, 3)
    coupling = None
    if _c_real is not None:
        coupling = (np.array(_c_real) + 1j * np.array(_c_imag)).reshape(n_elements, n_elements)

    az_grid = np.linspace(0, 360, az_pts)
    el_grid = np.linspace(0, 90, el_pts)

    return evaluate_pattern_3d(
        w, positions, freq_hz, az_grid, el_grid,
        coupling_matrix=coupling,
    ), az_grid, el_grid


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def build_sidebar():
    """Build the sidebar controls and return full configuration."""

    st.sidebar.markdown('# 📡 CRPA NULL STEERING')
    st.sidebar.markdown('---')

    # ── Algorithm Selection ──
    st.sidebar.markdown('### Algorithm')
    algo_choice = st.sidebar.radio(
        'Null-Steering Method',
        ['Projection Matrix', 'MVDR / SMI'],
        index=0,
        help=(
            '**Projection Matrix** (paper-aligned): Deterministic, guaranteed clean '
            'quiescent beam with 0 jammers. Uses P⊥ = I − A_n(A_n^H A_n + εI)^{-1}A_n^H.\n\n'
            '**MVDR/SMI**: Adaptive covariance-based. Accounts for jammer power levels '
            'but requires diagonal loading for robustness.'
        ),
    )
    algorithm = (
        NullingAlgorithm.PROJECTION if algo_choice == 'Projection Matrix'
        else NullingAlgorithm.MVDR
    )

    st.sidebar.markdown('---')

    # ── Array Configuration ──
    st.sidebar.markdown('### Array Configuration')

    topo_choice = st.sidebar.selectbox(
        'Array Topology',
        ['Circular (N+1)', 'Conformal Cylinder', 'Conformal Sphere', 'Rectangular'],
        index=0,
    )
    topo_map = {
        'Circular (N+1)': ArrayTopology.CIRCULAR,
        'Conformal Cylinder': ArrayTopology.CONFORMAL_CYLINDER,
        'Conformal Sphere': ArrayTopology.CONFORMAL_SPHERE,
        'Rectangular': ArrayTopology.RECTANGULAR,
    }
    topology = topo_map[topo_choice]

    n_elements = st.sidebar.slider('Number of Elements', 4, 16, 7,
                                   help='Total elements including centre element')
    freq_mhz = st.sidebar.number_input('Carrier Frequency (MHz)', 500.0, 6000.0,
                                        1575.42, step=0.01, format='%.2f')
    radius_mm = st.sidebar.slider('Array Radius (mm)', 30, 300, 95,
                                  help='Ring/conformal radius for outer elements')

    array_cfg = ArrayConfig(
        n_elements=n_elements,
        radius_m=radius_mm / 1000.0,
        freq_hz=freq_mhz * 1e6,
        topology=topology,
    )

    st.sidebar.markdown('---')

    # ── Desired Signal ──
    st.sidebar.markdown('### Desired Signal')
    des_az = st.sidebar.slider('Azimuth (°)', 0, 359, 0, key='des_az')
    des_el = st.sidebar.slider('Elevation (°)', 1, 90, 45, key='des_el')
    sig_power = st.sidebar.number_input(
        'Signal Power (dBm)', -160.0, -100.0, -130.0, step=1.0,
        help='Typical GPS: -130 dBm',
    )

    st.sidebar.markdown('---')

    # ── Jammers ──
    st.sidebar.markdown('### Jammer Sources')
    n_jammers = st.sidebar.slider('Number of Jammers', 0, min(4, n_elements - 1), 0)

    jammers: List[JammerConfig] = []
    for i in range(n_jammers):
        with st.sidebar.expander(f'Jammer {i + 1}', expanded=(i == 0)):
            jaz = st.slider(f'J{i+1} Azimuth (°)', 0, 359,
                            [90, 220, 315, 45][i % 4], key=f'j{i}_az')
            jel = st.slider(f'J{i+1} Elevation (°)', 1, 90, 30, key=f'j{i}_el')
            jjsr = st.slider(f'J{i+1} JSR (dB)', 10, 80, 40, key=f'j{i}_jsr')
            jpol = st.selectbox(
                f'J{i+1} Polarisation',
                ['Linear-V', 'Linear-H', 'LHCP', 'RHCP'],
                index=0, key=f'j{i}_pol',
            )
            pol_map = {
                'Linear-V': Polarisation.LINEAR_V,
                'Linear-H': Polarisation.LINEAR_H,
                'LHCP': Polarisation.LHCP,
                'RHCP': Polarisation.RHCP,
            }
            jammers.append(JammerConfig(
                azimuth_deg=float(jaz),
                elevation_deg=float(jel),
                jsr_db=float(jjsr),
                polarisation=pol_map[jpol],
            ))

    st.sidebar.markdown('---')

    # ── Hardware Quantization ──
    st.sidebar.markdown('### Hardware Constraints')
    quant_enabled = st.sidebar.checkbox('Enable Quantization', value=True)
    phase_bits = st.sidebar.slider('Phase Shifter Bits', 3, 8, 6)
    amp_step = st.sidebar.select_slider('Amplitude Step (dB)',
                                        options=[0.25, 0.5, 1.0, 2.0], value=0.5)
    amp_range = st.sidebar.slider('Amplitude Range (dB)', 10.0, 63.0, 31.5, step=0.5)

    quant_cfg = QuantConfig(
        phase_bits=phase_bits,
        amp_step_db=float(amp_step),
        amp_range_db=amp_range,
        enabled=quant_enabled,
    )

    st.sidebar.markdown('---')

    # ── STAP Configuration ──
    st.sidebar.markdown('### STAP Processing')
    stap_enabled = st.sidebar.checkbox('Enable STAP', value=False,
                                       help='Space-Time Adaptive Processing for wideband nulling')
    stap_taps = st.sidebar.slider('FIR Taps per Element', 2, 16, 5,
                                   disabled=not stap_enabled)
    stap_delay = st.sidebar.slider('Tap Spacing (ns)', 10.0, 200.0, 50.0, step=10.0,
                                    disabled=not stap_enabled)

    stap_cfg = STAPConfig(
        enabled=stap_enabled,
        n_taps=stap_taps,
        tap_spacing_ns=stap_delay,
    )

    st.sidebar.markdown('---')

    # ── RF Environment ──
    st.sidebar.markdown('### RF Environment')
    noise_floor = st.sidebar.number_input(
        'Noise Floor (dBm)', -130.0, -90.0, -114.0, step=1.0,
    )
    coupling_mag = st.sidebar.slider('Mutual Coupling |S₂₁|', 0.01, 0.4, 0.15, step=0.01)
    isolation_db = st.sidebar.slider('Return Loss (dB)', -35.0, -10.0, -20.0, step=1.0)
    diag_loading = st.sidebar.slider('Diagonal Loading (dB)', 0.0, 20.0, 3.0, step=0.5,
                                     disabled=(algorithm == NullingAlgorithm.PROJECTION))

    # ── Capability Status ──
    st.sidebar.markdown('---')
    st.sidebar.markdown('### System Status')
    st.sidebar.markdown(
        f'🔧 Numba: **{"✅ Active" if HAS_NUMBA else "❌ Not installed"}**\n\n'
        f'🔧 scikit-rf: **{"✅ Active" if HAS_SKRF else "❌ Not installed"}**\n\n'
        f'🔧 Algorithm: **{algorithm.value}**'
    )

    cfg = ScenarioConfig(
        array=array_cfg,
        jammers=jammers,
        quant=quant_cfg,
        stap=stap_cfg,
        desired_az_deg=float(des_az),
        desired_el_deg=float(des_el),
        noise_floor_dbm=noise_floor,
        signal_power_dbm=sig_power,
        algorithm=algorithm,
    )

    return cfg, coupling_mag, isolation_db, diag_loading


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg, coupling_mag, isolation_db, diag_loading = build_sidebar()
    arr = cfg.array
    positions = arr.element_positions_m
    n = arr.n_elements
    freq = arr.freq_hz
    lam = wavelength(freq)

    # ── AEP Generation ──
    aep_dicts = cached_generate_aeps(n, 2.0, 2.0)
    aeps = [
        AEPData(
            theta_deg=d['theta'], phi_deg=d['phi'],
            magnitude=d['magnitude'], phase_deg=d['phase'],
            element_index=d['index'],
        )
        for d in aep_dicts
    ]

    # ── FIX: Replace build_aep_interpolator with RectBivariateSpline ──
    # The original function returned interpolators incompatible with the
    # grid=False call inside steering_vector_with_aep. RectBivariateSpline
    # supports this interface and works identically.
    interpolators = [
        RectBivariateSpline(
            aep.theta_deg, aep.phi_deg, aep.magnitude,
            kx=1, ky=1, s=0            # linear interpolation, no smoothing
        )
        for aep in aeps
    ]
    # ────────────────────────────────────────────────────────────────

    # ── Coupling Matrix ──
    S = cached_s_matrix(n, coupling_mag, isolation_db)
    C = s_to_coupling_matrix(S)

    # ══════════════════════════════════════════════════════════════
    #  WEIGHT COMPUTATION — Algorithm Selection
    # ══════════════════════════════════════════════════════════════
    #
    #  PROJECTION MATRIX (Bug Fix):
    #    - Weights computed in IDEAL (uncoupled) domain
    #    - With K=0: w = a_d / ‖a_d‖ → clean quiescent beam
    #    - Pattern eval: |w^H a(θ)|² — no coupling in eval
    #    - Coupling ONLY in pattern eval when MVDR is selected
    #
    #  MVDR / SMI:
    #    - Weights computed in COUPLED domain: a_d_coupled = C·a_d
    #    - Pattern eval: |w^H (C·A)|² — coupling in both domains
    #    - With K=0: w ∝ C·a_d → clean coupled beam (no ghost nulls
    #      because coupling applied consistently in both domains)
    # ══════════════════════════════════════════════════════════════

    if cfg.algorithm == NullingAlgorithm.PROJECTION:
        # Paper-aligned: weights in ideal domain, no coupling in weight calc
        w_ideal = projection_matrix_weights(
            positions, freq, cfg.desired_az_deg, cfg.desired_el_deg,
            cfg.jammers, coupling_matrix=None,  # No coupling in weights!
            aep_interpolators=interpolators,
        )
        # Pattern evaluation: also in ideal domain (no coupling)
        C_for_pattern = None
    else:
        # MVDR: weights and pattern both in coupled domain
        a_desired = steering_vector_with_aep(
            positions, freq, cfg.desired_az_deg, cfg.desired_el_deg, interpolators
        )
        a_desired_coupled = coupled_steering_vector(C, a_desired)
        R = build_covariance_matrix(
            positions, freq, cfg.jammers, cfg.noise_floor_dbm,
            cfg.signal_power_dbm, coupling_matrix=C,
            aep_interpolators=interpolators,
        )
        w_ideal = smi_weights(R, a_desired_coupled, diagonal_loading_db=diag_loading)
        C_for_pattern = C  # Coupling in both domains

    w_quant = quantize_weights(w_ideal, cfg.quant)

    # ── Pattern Evaluation ──
    az_sweep = np.linspace(0, 360, 720)
    pattern_ideal = evaluate_pattern_2d(
        w_ideal, positions, freq, az_sweep,
        el_deg=cfg.desired_el_deg, coupling_matrix=C_for_pattern,
        aep_interpolators=interpolators,
    )
    pattern_quant = evaluate_pattern_2d(
        w_quant, positions, freq, az_sweep,
        el_deg=cfg.desired_el_deg, coupling_matrix=C_for_pattern,
        aep_interpolators=interpolators,
    )

    # ═══════════ HEADER ═══════════
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 5px 0;">
        <h1 style="margin-bottom:2px; font-size:28px;">📡 CRPA NULL STEERING DASHBOARD</h1>
        <p style="color: #7a8ba3; font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 3px;">
            HARDWARE-CONSTRAINED RF BEAMFORMING SIMULATOR
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════ METRICS ROW ═══════════
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric('Elements', f'{n}')
    with col2:
        st.metric('Frequency', f'{freq / 1e6:.2f} MHz')
    with col3:
        st.metric('λ', f'{lam * 100:.1f} cm')
    with col4:
        quant_err = quantization_error_db(w_ideal, w_quant)
        st.metric('Quant Error', f'{quant_err:.1f} dB')
    with col5:
        if cfg.jammers:
            j0 = cfg.jammers[0]
            j0_idx = np.argmin(np.abs(az_sweep - j0.azimuth_deg))
            w_active = w_quant if cfg.quant.enabled else w_ideal
            null_d = pattern_quant[j0_idx] if cfg.quant.enabled else pattern_ideal[j0_idx]
            st.metric('J1 Null Depth', f'{null_d:.1f} dB')
        else:
            st.metric('Jammers', '0')

    st.markdown('---')

    # ═══════════ TABS ═══════════
    tab_layout, tab_2d, tab_3d, tab_squint, tab_hw, tab_aep, tab_pol = st.tabs([
        '🏗️ Array Layout',
        '📊 2D Pattern',
        '🌐 3D Pattern',
        '📉 Wideband Squint',
        '⚙️ Hardware',
        '📐 AEP',
        '🔄 Polarisation',
    ])

    # ════════ Tab 1: Array Layout ════════
    with tab_layout:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig_geom = plot_array_geometry(positions, cfg.desired_az_deg, cfg.jammers)
            st.pyplot(fig_geom, use_container_width=True)

        with c2:
            st.markdown('#### Coupling Matrix $|C|$ (dB)')
            fig_coup = plot_coupling_matrix(C)
            st.pyplot(fig_coup, use_container_width=True)

            d_spacing = 2 * arr.radius_m * np.sin(np.pi / max(n - 1, 1))
            metrics = {
                'Algorithm': cfg.algorithm.value,
                'Topology': arr.topology.value.replace('_', ' ').title(),
                'Array Radius': f'{arr.radius_m * 1000:.1f} mm',
                'Element Spacing': f'{d_spacing * 1000:.1f} mm ({d_spacing / lam:.3f}λ)',
                'Phase Shifter': f'{cfg.quant.phase_bits}-bit',
                'Noise Floor': f'{cfg.noise_floor_dbm:.0f} dBm',
                'Jammers': f'{len(cfg.jammers)}',
                'DoF (N−1)': f'{n - 1}',
                'STAP': f'{"ON (" + str(cfg.stap.n_taps) + " taps)" if cfg.stap.enabled else "OFF"}',
            }
            st.plotly_chart(create_metrics_table(metrics), use_container_width=True)

    # ════════ Tab 2: 2D Pattern Cuts ════════
    with tab_2d:
        dyn_range = st.slider('Dynamic Range (dB)', 20, 80, 60, step=5, key='dr_2d')
        c1, c2 = st.columns(2)
        with c1:
            fig_polar = plot_polar_pattern(
                az_sweep, pattern_ideal, cfg.desired_az_deg, cfg.jammers,
                title=f'{cfg.algorithm.value} Pattern (Polar)',
                pattern_db_quant=pattern_quant if cfg.quant.enabled else None,
                dynamic_range=dyn_range,
            )
            st.pyplot(fig_polar, use_container_width=True)

        with c2:
            fig_cart = plot_cartesian_pattern(
                az_sweep, pattern_ideal, cfg.desired_az_deg, cfg.jammers,
                pattern_db_quant=pattern_quant if cfg.quant.enabled else None,
                dynamic_range=dyn_range,
            )
            st.pyplot(fig_cart, use_container_width=True)

        if cfg.jammers:
            st.markdown('#### Null Performance Summary')
            rows = []
            for i, j in enumerate(cfg.jammers):
                j_idx = np.argmin(np.abs(az_sweep - j.azimuth_deg))
                nd_ideal = pattern_ideal[j_idx]
                nd_quant = pattern_quant[j_idx]
                pol_loss = polarisation_mismatch_loss_db(j.polarisation, Polarisation.RHCP)
                rows.append({
                    'Jammer': f'J{i+1}',
                    'Az (°)': f'{j.azimuth_deg:.1f}',
                    'El (°)': f'{j.elevation_deg:.1f}',
                    'JSR (dB)': f'{j.jsr_db:.0f}',
                    'Pol': j.polarisation.value,
                    'Pol Loss (dB)': f'{pol_loss:.1f}',
                    'Null Ideal (dB)': f'{nd_ideal:.1f}',
                    'Null Quant (dB)': f'{nd_quant:.1f}',
                    'Δ (dB)': f'{nd_quant - nd_ideal:.1f}',
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ════════ Tab 3: 3D Pattern (FIXED) ════════
    with tab_3d:
        st.markdown('#### Interactive 3D Radiation Pattern')
        st.markdown(
            '*Resolution is automatically capped at 60×60 vertices to prevent '
            'browser WebGL hangs.*'
        )
        res_3d = st.select_slider('3D Resolution', options=[30, 45, 60], value=45,
                                  help='Grid points per axis (auto-decimated in renderer)')

        w_for_3d = w_quant if cfg.quant.enabled else w_ideal
        _w_r = tuple(w_for_3d.real.tolist())
        _w_i = tuple(w_for_3d.imag.tolist())
        _p_t = tuple(positions.ravel().tolist())
        _c_r = tuple(C_for_pattern.real.ravel().tolist()) if C_for_pattern is not None else None
        _c_i = tuple(C_for_pattern.imag.ravel().tolist()) if C_for_pattern is not None else None

        result, az_3d, el_3d = cached_3d_pattern(
            _w_r, _w_i, _p_t, n, freq, res_3d, max(res_3d // 2, 15),
            _c_r, _c_i,
        )
        fig_3d = plot_3d_pattern(az_3d, el_3d, result, dynamic_range=40.0)
        st.plotly_chart(fig_3d, use_container_width=True)

    # ════════ Tab 4: Wideband Squint ════════
    with tab_squint:
        st.markdown('#### Wideband Null Dispersion Analysis')
        st.markdown(
            'Evaluates narrowband weights $w(f_c)$ across a frequency sweep. '
            'Null depth degrades at band edges due to **array squint**.'
        )

        bw_mhz = st.slider('Analysis Bandwidth (± MHz)', 1.0, 30.0, 15.0, step=1.0)
        n_freq_pts = st.slider('Frequency Points', 31, 201, 101, step=10)
        f_offsets = np.linspace(-bw_mhz * 1e6, bw_mhz * 1e6, n_freq_pts)

        if cfg.jammers:
            null_depths = {}
            stap_depths = None

            w_eval = w_quant if cfg.quant.enabled else w_ideal

            for i, j in enumerate(cfg.jammers):
                nd = wideband_null_response(
                    w_eval, positions, freq, f_offsets,
                    j.azimuth_deg, j.elevation_deg,
                    coupling_matrix_fc=C_for_pattern,
                    aep_interpolators=interpolators,
                )
                # Normalise to desired signal response
                a_des = steering_vector_with_aep(
                    positions, freq, cfg.desired_az_deg, cfg.desired_el_deg, interpolators
                )
                if C_for_pattern is not None:
                    a_des = coupled_steering_vector(C_for_pattern, a_des)
                p_des = np.abs(w_eval.conj() @ a_des) ** 2
                nd -= 10 * np.log10(max(p_des, 1e-30))
                null_depths[f'J{i+1} ({j.azimuth_deg:.0f}°, {j.jsr_db:.0f}dB)'] = nd

            # STAP comparison if enabled
            if cfg.stap.enabled and cfg.jammers:
                stap_depths = {}
                tap_s = cfg.stap.tap_spacing_ns * 1e-9
                w_st = stap_weights(
                    positions, freq, cfg, C if cfg.algorithm == NullingAlgorithm.MVDR else None,
                    interpolators, diag_loading,
                )
                for i, j in enumerate(cfg.jammers):
                    nd_stap = stap_null_response(
                        w_st, positions, freq, f_offsets,
                        j.azimuth_deg, j.elevation_deg,
                        cfg.stap.n_taps, tap_s, C_for_pattern, interpolators,
                    )
                    v_des = build_stap_steering_vector(
                        positions, freq, cfg.desired_az_deg, cfg.desired_el_deg,
                        cfg.stap.n_taps, tap_s, C_for_pattern, interpolators,
                    )
                    p_des_st = np.abs(w_st.conj() @ v_des) ** 2
                    nd_stap -= 10 * np.log10(max(p_des_st, 1e-30))
                    stap_depths[f'J{i+1} STAP'] = nd_stap

            fig_sq = plot_wideband_squint(f_offsets / 1e6, null_depths, stap_depths)
            st.plotly_chart(fig_sq, use_container_width=True)

            st.markdown('##### Null Bandwidth Summary')
            for label, nd in null_depths.items():
                mask_20 = nd < -20
                bw_20 = (f_offsets[mask_20][-1] - f_offsets[mask_20][0]) / 1e6 if mask_20.any() else 0.0
                mask_30 = nd < -30
                bw_30 = (f_offsets[mask_30][-1] - f_offsets[mask_30][0]) / 1e6 if mask_30.any() else 0.0
                st.markdown(
                    f'**{label}**: '
                    f'BW @ −20 dB = **{bw_20:.1f} MHz** · '
                    f'BW @ −30 dB = **{bw_30:.1f} MHz**'
                )
        else:
            st.info('Add at least one jammer to analyse wideband null dispersion.')

    # ════════ Tab 5: Hardware Limits ════════
    with tab_hw:
        st.markdown('#### Beamformer Weight Analysis')

        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig_w = plot_weights(w_ideal, w_quant if cfg.quant.enabled else None, cfg.quant)
            st.pyplot(fig_w, use_container_width=True)

        with c2:
            st.markdown('##### Quantization Impact')
            phase_lsb = 360.0 / 2 ** cfg.quant.phase_bits
            st.markdown(f"""
| Parameter | Value |
|:--|:--|
| Phase LSB | {phase_lsb:.2f}° |
| Phase States | {2**cfg.quant.phase_bits} |
| Amplitude LSB | {cfg.quant.amp_step_db} dB |
| Amp States | {int(cfg.quant.amp_range_db / cfg.quant.amp_step_db) + 1} |
| RMS Error | {quantization_error_db(w_ideal, w_quant):.2f} dB |
""")

            st.markdown('##### Per-Element Weights')
            rows = []
            for i in range(n):
                rows.append({
                    'Elem': f'{"★" if i == 0 else ""}E{i}',
                    '|w| Ideal': f'{np.abs(w_ideal[i]):.4f}',
                    '|w| Quant': f'{np.abs(w_quant[i]):.4f}',
                    '∠ Ideal': f'{np.rad2deg(np.angle(w_ideal[i])):.1f}°',
                    '∠ Quant': f'{np.rad2deg(np.angle(w_quant[i])):.1f}°',
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # STAP weight visualization
        if cfg.stap.enabled:
            st.markdown('#### STAP Weight Matrix')
            tap_s = cfg.stap.tap_spacing_ns * 1e-9
            w_st = stap_weights(
                positions, freq, cfg,
                C if cfg.algorithm == NullingAlgorithm.MVDR else None,
                interpolators, diag_loading,
            )
            fig_stap = plot_stap_weights(w_st, n, cfg.stap.n_taps)
            st.pyplot(fig_stap, use_container_width=True)

    # ════════ Tab 6: Element Patterns ════════
    with tab_aep:
        st.markdown('#### Active Element Patterns (AEP)')
        st.markdown(
            'Per-element patterns from a Huygens-source model with edge diffraction. '
            'These replace the naive `cos^1.5(θ)` approximation.'
        )

        el_select = st.selectbox(
            'Select Element', list(range(n)),
            format_func=lambda i: f'Element {i} {"(Centre)" if i == 0 else "(Ring)"}',
        )

        aep = aeps[el_select]
        fig_aep = plot_aep_pattern(
            aep.theta_deg, aep.phi_deg, aep.magnitude,
            element_index=el_select,
        )
        st.pyplot(fig_aep, use_container_width=True)

        import matplotlib.pyplot as plt
        c1, c2 = st.columns(2)
        with c1:
            fig_cut, ax_cut = plt.subplots(figsize=(6, 4))
            ax_cut.set_facecolor(PANEL_BG)
            fig_cut.patch.set_facecolor(DARK_BG)
            ax_cut.plot(aep.theta_deg, aep.magnitude[:, 0],
                       color=ACCENT_CYAN, linewidth=2)
            ax_cut.set_xlabel('Theta (°)', color=TEXT_COLOR)
            ax_cut.set_ylabel('Normalised Gain', color=TEXT_COLOR)
            ax_cut.set_title(f'E-Plane Cut (φ=0°) — Element {el_select}',
                           color=TEXT_COLOR, fontsize=12, fontweight='bold')
            ax_cut.tick_params(colors=TEXT_COLOR)
            ax_cut.grid(True, alpha=0.3, color='#1e2d42')
            ax_cut.set_xlim(0, 180)
            fig_cut.tight_layout()
            st.pyplot(fig_cut, use_container_width=True)

        with c2:
            fig_cut2, ax_cut2 = plt.subplots(figsize=(6, 4))
            ax_cut2.set_facecolor(PANEL_BG)
            fig_cut2.patch.set_facecolor(DARK_BG)
            phi_idx_90 = np.argmin(np.abs(aep.phi_deg - 90.0))
            ax_cut2.plot(aep.theta_deg, aep.magnitude[:, phi_idx_90],
                        color=ACCENT_GREEN, linewidth=2)
            ax_cut2.set_xlabel('Theta (°)', color=TEXT_COLOR)
            ax_cut2.set_ylabel('Normalised Gain', color=TEXT_COLOR)
            ax_cut2.set_title(f'H-Plane Cut (φ=90°) — Element {el_select}',
                            color=TEXT_COLOR, fontsize=12, fontweight='bold')
            ax_cut2.tick_params(colors=TEXT_COLOR)
            ax_cut2.grid(True, alpha=0.3, color='#1e2d42')
            ax_cut2.set_xlim(0, 180)
            fig_cut2.tight_layout()
            st.pyplot(fig_cut2, use_container_width=True)

    # ════════ Tab 7: Polarisation ════════
    with tab_pol:
        st.markdown('#### Polarisation Analysis')
        st.markdown(
            'GNSS signals are **Right-Hand Circularly Polarised (RHCP)**. '
            'This tab analyses the array\'s polarisation response and '
            'computes mismatch losses for different jammer polarisations.'
        )

        # Polarisation mismatch table
        st.markdown('##### Polarisation Mismatch Loss Matrix')
        pols = [Polarisation.RHCP, Polarisation.LHCP,
                Polarisation.LINEAR_H, Polarisation.LINEAR_V]
        pol_data = []
        for tx in pols:
            row = {'TX Polarisation': tx.value}
            for rx in pols:
                loss = polarisation_mismatch_loss_db(tx, rx)
                row[f'RX {rx.value}'] = f'{loss:.1f} dB'
            pol_data.append(row)
        st.dataframe(pd.DataFrame(pol_data), use_container_width=True, hide_index=True)

        # Jones vector analysis
        st.markdown('##### Jones Vector Properties')
        for pol in pols:
            jones = polarisation_jones_vector(pol)
            ar = axial_ratio_from_jones(jones)
            st.markdown(
                f'**{pol.value}**: Jones = [{jones[0]:.3f}, {jones[1]:.3f}] · '
                f'Axial Ratio = {ar:.1f} dB'
            )

        # Jammer polarisation impact
        if cfg.jammers:
            st.markdown('##### Jammer Polarisation Impact on Null Depth')
            st.markdown(
                'The effective JSR seen by the RHCP array is reduced by the '
                'polarisation mismatch loss. An LHCP jammer is significantly '
                'attenuated before any spatial nulling.'
            )
            for i, j in enumerate(cfg.jammers):
                pol_loss = polarisation_mismatch_loss_db(j.polarisation, Polarisation.RHCP)
                effective_jsr = j.jsr_db + pol_loss
                st.markdown(
                    f'**J{i+1}** ({j.polarisation.value}): '
                    f'JSR = {j.jsr_db:.0f} dB − Pol loss {abs(pol_loss):.1f} dB = '
                    f'**Effective JSR = {effective_jsr:.1f} dB**'
                )


if __name__ == '__main__':
    main()
