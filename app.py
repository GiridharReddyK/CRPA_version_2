"""
app.py — CRPA Null Steering Dashboard (Production-Grade)

A physically grounded, hardware-constrained RF engineering simulator for
Controlled Reception Pattern Antenna null steering analysis.

Features:
    1. Active Element Pattern (AEP) ingestion with RectBivariateSpline
    2. Mutual Coupling Matrix from S-parameters
    3. Hardware quantization (phase shifter + attenuator)
    4. Sample Matrix Inversion (SMI) covariance-based nulling
    5. Wideband null dispersion (squint) analysis

Author: RF Systems Engineering Division
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Optional

# Local modules
from rf_physics import (
    ArrayConfig, JammerConfig, QuantConfig, ScenarioConfig,
    wavelength, ideal_steering_vector, steering_vector_with_aep,
    coupled_steering_vector, generate_mock_s_matrix, s_to_coupling_matrix,
    build_covariance_matrix, smi_weights, quantize_weights,
    quantization_error_db, evaluate_pattern_2d, evaluate_pattern_3d,
    wideband_null_response, dbm_to_watts,
)
from data_ingestion import (
    generate_mock_aep, generate_all_element_aeps,
    build_aep_interpolator, AEPData,
)
from visualizations import (
    plot_array_geometry, plot_polar_pattern, plot_cartesian_pattern,
    plot_3d_pattern, plot_weights, plot_wideband_squint,
    plot_coupling_matrix, plot_aep_pattern, create_metrics_table,
    DARK_BG, ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    TEXT_COLOR, PANEL_BG,
)


# ---------------------------------------------------------------------------
# Page Config & Custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title='CRPA Null Steering Dashboard',
    page_icon='📡',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(f"""
<style>
    /* ---- Dark engineering dashboard theme ---- */
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
    .metric-card {{
        background: linear-gradient(135deg, {PANEL_BG}, #141e30);
        border: 1px solid #1e2d42;
        border-radius: 10px;
        padding: 16px;
        margin: 4px 0;
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


# ---------------------------------------------------------------------------
# Cached Computation Functions
# ---------------------------------------------------------------------------

@st.cache_data
def cached_generate_aeps(n_elements: int, theta_res: float, phi_res: float):
    """Generate and cache AEP data for all elements."""
    aeps, _ = generate_all_element_aeps(n_elements, theta_res, phi_res)
    # Convert to serialisable form for caching
    aep_dicts = []
    for aep in aeps:
        aep_dicts.append({
            'theta': aep.theta_deg,
            'phi': aep.phi_deg,
            'magnitude': aep.magnitude,
            'phase': aep.phase_deg,
            'index': aep.element_index,
        })
    return aep_dicts


@st.cache_data
def cached_s_matrix(n_elements: int, coupling_mag: float, isolation_db: float):
    """Generate and cache the S-parameter matrix."""
    return generate_mock_s_matrix(n_elements, coupling_mag, isolation_db)


@st.cache_data
def cached_3d_pattern(
    _w_real, _w_imag, _pos_tuple, n_elements, freq_hz, az_pts, el_pts,
    _c_real, _c_imag,
):
    """Cache-friendly 3D pattern evaluation.

    Complex arrays are split into real/imag tuples for hashability.
    Shape information is reconstructed from n_elements.
    """
    w = np.array(_w_real) + 1j * np.array(_w_imag)
    positions = np.array(_pos_tuple).reshape(n_elements, 3)
    coupling = (
        (np.array(_c_real) + 1j * np.array(_c_imag)).reshape(n_elements, n_elements)
        if _c_real is not None else None
    )

    az_grid = np.linspace(0, 360, az_pts)
    el_grid = np.linspace(0, 90, el_pts)

    return evaluate_pattern_3d(
        w, positions, freq_hz, az_grid, el_grid,
        coupling_matrix=coupling,
    ), az_grid, el_grid


# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------

def build_sidebar() -> ScenarioConfig:
    """Build the sidebar controls and return a ScenarioConfig."""

    st.sidebar.markdown('# 📡 CRPA NULL STEERING')
    st.sidebar.markdown('---')

    # --- Array Parameters ---
    st.sidebar.markdown('### Array Configuration')
    n_elements = st.sidebar.slider('Number of Elements', 4, 12, 7,
                                   help='Total elements including centre element')
    freq_mhz = st.sidebar.number_input('Carrier Frequency (MHz)', 500.0, 6000.0,
                                        1575.42, step=0.01, format='%.2f')
    radius_mm = st.sidebar.slider('Array Radius (mm)', 30, 300, 95,
                                  help='Ring radius for outer elements')

    array_cfg = ArrayConfig(
        n_elements=n_elements,
        radius_m=radius_mm / 1000.0,
        freq_hz=freq_mhz * 1e6,
    )

    st.sidebar.markdown('---')

    # --- Desired Signal ---
    st.sidebar.markdown('### Desired Signal')
    des_az = st.sidebar.slider('Azimuth (°)', 0, 359, 0, key='des_az')
    des_el = st.sidebar.slider('Elevation (°)', 1, 90, 45, key='des_el')
    sig_power = st.sidebar.number_input(
        'Signal Power (dBm)', -160.0, -100.0, -130.0, step=1.0,
        help='Typical GPS: -130 dBm',
    )

    st.sidebar.markdown('---')

    # --- Jammers ---
    st.sidebar.markdown('### Jammer Sources')
    n_jammers = st.sidebar.slider('Number of Jammers', 0, min(4, n_elements - 1), 2)

    jammers: List[JammerConfig] = []
    for i in range(n_jammers):
        with st.sidebar.expander(f'Jammer {i + 1}', expanded=(i == 0)):
            jaz = st.slider(f'J{i+1} Azimuth (°)', 0, 359,
                            [90, 220, 315, 45][i % 4], key=f'j{i}_az')
            jel = st.slider(f'J{i+1} Elevation (°)', 1, 90, 30, key=f'j{i}_el')
            jjsr = st.slider(f'J{i+1} JSR (dB)', 10, 80, 40, key=f'j{i}_jsr')
            jammers.append(JammerConfig(
                azimuth_deg=float(jaz),
                elevation_deg=float(jel),
                jsr_db=float(jjsr),
            ))

    st.sidebar.markdown('---')

    # --- Hardware Quantization ---
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

    # --- RF Environment ---
    st.sidebar.markdown('### RF Environment')
    noise_floor = st.sidebar.number_input(
        'Noise Floor (dBm)', -130.0, -90.0, -114.0, step=1.0,
        help='kTB for ~2 MHz GPS bandwidth',
    )
    coupling_mag = st.sidebar.slider('Mutual Coupling Level', 0.01, 0.4, 0.15,
                                     step=0.01, help='Adjacent element |S21|')
    isolation_db = st.sidebar.slider('Return Loss (dB)', -35.0, -10.0, -20.0, step=1.0)
    diag_loading = st.sidebar.slider('Diagonal Loading (dB)', 0.0, 20.0, 3.0, step=0.5,
                                     help='Robustness against steering errors')

    return ScenarioConfig(
        array=array_cfg,
        jammers=jammers,
        quant=quant_cfg,
        desired_az_deg=float(des_az),
        desired_el_deg=float(des_el),
        noise_floor_dbm=noise_floor,
        signal_power_dbm=sig_power,
    ), coupling_mag, isolation_db, diag_loading


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the CRPA dashboard."""

    # Build sidebar and get configuration
    cfg, coupling_mag, isolation_db, diag_loading = build_sidebar()
    arr = cfg.array
    positions = arr.element_positions_m
    n = arr.n_elements
    freq = arr.freq_hz
    lam = wavelength(freq)

    # --- Generate AEPs ---
    aep_dicts = cached_generate_aeps(n, 2.0, 2.0)
    aeps = [
        AEPData(
            theta_deg=d['theta'], phi_deg=d['phi'],
            magnitude=d['magnitude'], phase_deg=d['phase'],
            element_index=d['index'],
        )
        for d in aep_dicts
    ]
    interpolators = [build_aep_interpolator(aep) for aep in aeps]

    # --- Coupling Matrix ---
    S = cached_s_matrix(n, coupling_mag, isolation_db)
    C = s_to_coupling_matrix(S)

    # --- Steering Vectors ---
    a_desired_ideal = ideal_steering_vector(
        positions, freq, cfg.desired_az_deg, cfg.desired_el_deg
    )
    a_desired_aep = steering_vector_with_aep(
        positions, freq, cfg.desired_az_deg, cfg.desired_el_deg, interpolators
    )
    a_desired_coupled = coupled_steering_vector(C, a_desired_aep)

    # --- Covariance & Weights ---
    R = build_covariance_matrix(
        positions, freq, cfg.jammers, cfg.noise_floor_dbm,
        cfg.signal_power_dbm, coupling_matrix=C,
        aep_interpolators=interpolators,
    )
    w_ideal = smi_weights(R, a_desired_coupled, diagonal_loading_db=diag_loading)
    w_quant = quantize_weights(w_ideal, cfg.quant)

    # --- Pattern Evaluation ---
    az_sweep = np.linspace(0, 360, 720)
    pattern_ideal = evaluate_pattern_2d(
        w_ideal, positions, freq, az_sweep,
        el_deg=cfg.desired_el_deg, coupling_matrix=C,
        aep_interpolators=interpolators,
    )
    pattern_quant = evaluate_pattern_2d(
        w_quant, positions, freq, az_sweep,
        el_deg=cfg.desired_el_deg, coupling_matrix=C,
        aep_interpolators=interpolators,
    )

    # === HEADER ===
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 5px 0;">
        <h1 style="margin-bottom:2px; font-size:28px;">📡 CRPA NULL STEERING DASHBOARD</h1>
        <p style="color: #7a8ba3; font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 3px;">
            HARDWARE-CONSTRAINED RF BEAMFORMING SIMULATOR
        </p>
    </div>
    """, unsafe_allow_html=True)

    # === METRICS ROW ===
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
        # Null depth at first jammer
        if cfg.jammers:
            j0 = cfg.jammers[0]
            j0_idx = np.argmin(np.abs(az_sweep - j0.azimuth_deg))
            null_d = pattern_quant[j0_idx] if cfg.quant.enabled else pattern_ideal[j0_idx]
            st.metric('J1 Null Depth', f'{null_d:.1f} dB')
        else:
            st.metric('Jammers', '0')

    st.markdown('---')

    # === TABS ===
    tab_layout, tab_2d, tab_3d, tab_squint, tab_hw, tab_aep = st.tabs([
        '🏗️ Array Layout',
        '📊 2D Pattern Cuts',
        '🌐 3D Pattern',
        '📉 Wideband Squint',
        '⚙️ Hardware Limits',
        '📐 Element Patterns',
    ])

    # ---- Tab 1: Array Layout ----
    with tab_layout:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig_geom = plot_array_geometry(
                positions, cfg.desired_az_deg, cfg.jammers
            )
            st.pyplot(fig_geom, use_container_width=True)

        with c2:
            st.markdown('#### Coupling Matrix  $|C|$  (dB)')
            fig_coup = plot_coupling_matrix(C, title='Mutual Coupling Matrix |C|')
            st.pyplot(fig_coup, use_container_width=True)

            # Scenario summary
            metrics = {
                'Array Radius': f'{arr.radius_m * 1000:.1f} mm',
                'Inter-element spacing': f'{2 * arr.radius_m * np.sin(np.pi / max(n-1, 1)) * 1000:.1f} mm',
                'Spacing / λ': f'{2 * arr.radius_m * np.sin(np.pi / max(n-1, 1)) / lam:.3f}',
                'Phase Shifter': f'{cfg.quant.phase_bits}-bit ({360 / 2**cfg.quant.phase_bits:.2f}° LSB)',
                'Attenuator': f'{cfg.quant.amp_step_db} dB step / {cfg.quant.amp_range_db} dB range',
                'Noise Floor': f'{cfg.noise_floor_dbm:.0f} dBm',
                'Diagonal Loading': f'{diag_loading:.1f} dB',
                'Number of Jammers': f'{len(cfg.jammers)}',
                'DOF Available': f'{n - 1} (N-1)',
            }
            st.plotly_chart(create_metrics_table(metrics), use_container_width=True)

    # ---- Tab 2: 2D Pattern Cuts ----
    with tab_2d:
        dyn_range = st.slider('Dynamic Range (dB)', 20, 80, 60, step=5, key='dr_2d')
        c1, c2 = st.columns(2)
        with c1:
            fig_polar = plot_polar_pattern(
                az_sweep, pattern_ideal, cfg.desired_az_deg, cfg.jammers,
                title='Adapted Radiation Pattern (Polar)',
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

        # Null depth table
        if cfg.jammers:
            st.markdown('#### Null Performance Summary')
            rows = []
            for i, j in enumerate(cfg.jammers):
                j_idx = np.argmin(np.abs(az_sweep - j.azimuth_deg))
                nd_ideal = pattern_ideal[j_idx]
                nd_quant = pattern_quant[j_idx]
                rows.append({
                    'Jammer': f'J{i+1}',
                    'Azimuth (°)': f'{j.azimuth_deg:.1f}',
                    'Elevation (°)': f'{j.elevation_deg:.1f}',
                    'JSR (dB)': f'{j.jsr_db:.0f}',
                    'Null Depth Ideal (dB)': f'{nd_ideal:.1f}',
                    'Null Depth Quantized (dB)': f'{nd_quant:.1f}',
                    'Degradation (dB)': f'{nd_quant - nd_ideal:.1f}',
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # ---- Tab 3: 3D Pattern ----
    with tab_3d:
        st.markdown('#### Interactive 3D Radiation Pattern')
        res_3d = st.select_slider('3D Resolution', options=[30, 45, 60, 90, 120], value=60,
                                  help='Higher = smoother but slower')

        w_for_3d = w_quant if cfg.quant.enabled else w_ideal
        _w_r = tuple(w_for_3d.real.tolist())
        _w_i = tuple(w_for_3d.imag.tolist())
        _p_t = tuple(positions.ravel().tolist())
        _c_r = tuple(C.real.ravel().tolist())
        _c_i = tuple(C.imag.ravel().tolist())

        result, az_3d, el_3d = cached_3d_pattern(
            _w_r, _w_i, _p_t, n, freq, res_3d, res_3d // 2,
            _c_r, _c_i,
        )
        fig_3d = plot_3d_pattern(az_3d, el_3d, result, dynamic_range=40.0)
        st.plotly_chart(fig_3d, use_container_width=True)

    # ---- Tab 4: Wideband Squint ----
    with tab_squint:
        st.markdown('#### Wideband Null Dispersion Analysis')
        st.markdown(
            'Evaluates narrowband weights $w(f_c)$ across a frequency sweep to reveal '
            'how null depth degrades at band edges due to **array squint**.'
        )

        bw_mhz = st.slider('Analysis Bandwidth (± MHz)', 1.0, 30.0, 15.0, step=1.0)
        n_freq_pts = st.slider('Frequency Points', 31, 201, 101, step=10)

        f_offsets = np.linspace(-bw_mhz * 1e6, bw_mhz * 1e6, n_freq_pts)

        if cfg.jammers:
            null_depths = {}
            for i, j in enumerate(cfg.jammers):
                w_eval = w_quant if cfg.quant.enabled else w_ideal
                nd = wideband_null_response(
                    w_eval, positions, freq, f_offsets,
                    j.azimuth_deg, j.elevation_deg,
                    coupling_matrix_fc=C,
                    aep_interpolators=interpolators,
                )
                # Normalise to desired signal response at fc
                a_des_fc = steering_vector_with_aep(
                    positions, freq, cfg.desired_az_deg, cfg.desired_el_deg, interpolators
                )
                a_des_fc = coupled_steering_vector(C, a_des_fc)
                p_des = np.abs(w_eval.conj() @ a_des_fc) ** 2
                nd -= 10 * np.log10(max(p_des, 1e-30))
                null_depths[f'J{i+1} ({j.azimuth_deg:.0f}°, {j.jsr_db:.0f}dB JSR)'] = nd

            fig_sq = plot_wideband_squint(f_offsets / 1e6, null_depths)
            st.plotly_chart(fig_sq, use_container_width=True)

            # Summary
            st.markdown('##### Null Bandwidth Summary')
            for label, nd in null_depths.items():
                mask_20 = nd < -20
                if mask_20.any():
                    bw_20 = (f_offsets[mask_20][-1] - f_offsets[mask_20][0]) / 1e6
                else:
                    bw_20 = 0.0
                mask_30 = nd < -30
                if mask_30.any():
                    bw_30 = (f_offsets[mask_30][-1] - f_offsets[mask_30][0]) / 1e6
                else:
                    bw_30 = 0.0
                st.markdown(
                    f'**{label}**: '
                    f'BW @ -20 dB = **{bw_20:.1f} MHz** · '
                    f'BW @ -30 dB = **{bw_30:.1f} MHz**'
                )
        else:
            st.info('Add at least one jammer to analyse wideband null dispersion.')

    # ---- Tab 5: Hardware Limits ----
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
| Amplitude States | {int(cfg.quant.amp_range_db / cfg.quant.amp_step_db) + 1} |
| RMS Weight Error | {quantization_error_db(w_ideal, w_quant):.2f} dB |
""")

            # Phase comparison table
            st.markdown('##### Per-Element Weight Comparison')
            rows = []
            for i in range(n):
                rows.append({
                    'Element': f'E{i}',
                    '|w| Ideal': f'{np.abs(w_ideal[i]):.4f}',
                    '|w| Quant': f'{np.abs(w_quant[i]):.4f}',
                    '∠w Ideal (°)': f'{np.rad2deg(np.angle(w_ideal[i])):.2f}',
                    '∠w Quant (°)': f'{np.rad2deg(np.angle(w_quant[i])):.2f}',
                    'Δ∠ (°)': f'{np.rad2deg(np.angle(w_quant[i]) - np.angle(w_ideal[i])):.2f}',
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Bit-depth sweep
        st.markdown('#### Phase Quantization Sweep')
        st.markdown('Evaluates null depth degradation across different phase shifter bit depths.')

        if cfg.jammers:
            bit_range = list(range(3, 9))
            sweep_data = []
            for bits in bit_range:
                q_tmp = QuantConfig(
                    phase_bits=bits,
                    amp_step_db=cfg.quant.amp_step_db,
                    amp_range_db=cfg.quant.amp_range_db,
                    enabled=True,
                )
                w_tmp = quantize_weights(w_ideal, q_tmp)
                patt_tmp = evaluate_pattern_2d(
                    w_tmp, positions, freq, az_sweep,
                    el_deg=cfg.desired_el_deg, coupling_matrix=C,
                    aep_interpolators=interpolators,
                )
                for ji, j in enumerate(cfg.jammers):
                    j_idx = np.argmin(np.abs(az_sweep - j.azimuth_deg))
                    sweep_data.append({
                        'Bits': bits,
                        'Jammer': f'J{ji+1}',
                        'Null Depth (dB)': patt_tmp[j_idx],
                        'Quant Error (dB)': quantization_error_db(w_ideal, w_tmp),
                    })
            df_sweep = pd.DataFrame(sweep_data)
            st.dataframe(df_sweep, use_container_width=True, hide_index=True)

    # ---- Tab 6: Element Patterns ----
    with tab_aep:
        st.markdown('#### Active Element Patterns (AEP)')
        st.markdown(
            'Per-element radiation patterns generated from a mock Huygens-source model '
            'with ground-plane edge diffraction. These replace the naive `cos^1.5(θ)` '
            'approximation and can be swapped for real HFSS/CST imports.'
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

        # Theta cut at phi=0
        c1, c2 = st.columns(2)
        with c1:
            import matplotlib.pyplot as plt
            fig_cut, ax_cut = plt.subplots(figsize=(6, 4))
            ax_cut.set_facecolor(PANEL_BG)
            fig_cut.patch.set_facecolor(DARK_BG)
            phi_idx = 0
            ax_cut.plot(aep.theta_deg, aep.magnitude[:, phi_idx],
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

        # Data format info
        with st.expander('AEP Data Format Reference'):
            st.markdown("""
**Expected CSV format for HFSS/CST import:**

| Theta | Phi | Mag | Phase |
|:------|:----|:----|:------|
| 0.0   | 0.0 | -3.2 | 12.5 |
| 0.0   | 2.0 | -3.1 | 11.8 |
| ...   | ... | ...  | ...  |

- **Theta**: Zenith angle (0° = boresight, 180° = nadir)
- **Phi**: Azimuth angle (0°–360°)
- **Mag**: Gain in dB (converted to linear internally)
- **Phase**: Phase in degrees

Use `data_ingestion.parse_aep_dataframe(df)` or
`data_ingestion.parse_hfss_csv(filepath)` to ingest.
""")


if __name__ == '__main__':
    main()
