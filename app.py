"""
app.py — CRPA Null Steering Dashboard (Production-Grade v3)
==========================================================

Industry-standard Controlled Reception Pattern Antenna simulator
with full-featured RF engineering capabilities.

NEW IN THIS VERSION:
- Added "IEEE Paper Exports" tab to generate high-resolution, 
  side-by-side Matplotlib figures specifically formatted for 
  academic paper submission (Quantization Collapse & Wideband Squint).
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

# Assuming these are available in your local rf_physics.py and others
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
    st.sidebar.markdown('# 📡 CRPA NULL STEERING')
    st.sidebar.markdown('---')

    st.sidebar.markdown('### Algorithm')
    algo_choice = st.sidebar.radio(
        'Null-Steering Method',
        ['Projection Matrix', 'MVDR / SMI'],
        index=0,
    )
    algorithm = (
        NullingAlgorithm.PROJECTION if algo_choice == 'Projection Matrix'
        else NullingAlgorithm.MVDR
    )

    st.sidebar.markdown('---')
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

    n_elements = st.sidebar.slider('Number of Elements', 4, 16, 7)
    freq_mhz = st.sidebar.number_input('Carrier Frequency (MHz)', 500.0, 6000.0, 1575.42, step=0.01, format='%.2f')
    radius_mm = st.sidebar.slider('Array Radius (mm)', 30, 300, 95)

    array_cfg = ArrayConfig(
        n_elements=n_elements,
        radius_m=radius_mm / 1000.0,
        freq_hz=freq_mhz * 1e6,
        topology=topology,
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Desired Signal')
    des_az = st.sidebar.slider('Azimuth (°)', 0, 359, 0, key='des_az')
    des_el = st.sidebar.slider('Elevation (°)', 1, 90, 45, key='des_el')
    sig_power = st.sidebar.number_input('Signal Power (dBm)', -160.0, -100.0, -130.0, step=1.0)

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Jammer Sources')
    n_jammers = st.sidebar.slider('Number of Jammers', 0, min(4, n_elements - 1), 1)

    jammers: List[JammerConfig] = []
    for i in range(n_jammers):
        with st.sidebar.expander(f'Jammer {i + 1}', expanded=(i == 0)):
            jaz = st.slider(f'J{i+1} Azimuth (°)', 0, 359, [90, 220, 315, 45][i % 4], key=f'j{i}_az')
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
    st.sidebar.markdown('### Hardware Constraints')
    quant_enabled = st.sidebar.checkbox('Enable Quantization', value=True)
    phase_bits = st.sidebar.slider('Phase Shifter Bits', 3, 8, 6)
    amp_step = st.sidebar.select_slider('Amplitude Step (dB)', options=[0.25, 0.5, 1.0, 2.0], value=0.5)
    amp_range = st.sidebar.slider('Amplitude Range (dB)', 10.0, 63.0, 31.5, step=0.5)

    quant_cfg = QuantConfig(
        phase_bits=phase_bits,
        amp_step_db=float(amp_step),
        amp_range_db=amp_range,
        enabled=quant_enabled,
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown('### STAP Processing')
    stap_enabled = st.sidebar.checkbox('Enable STAP', value=False)
    stap_taps = st.sidebar.slider('FIR Taps per Element', 2, 16, 5, disabled=not stap_enabled)
    stap_delay = st.sidebar.slider('Tap Spacing (ns)', 10.0, 200.0, 50.0, step=10.0, disabled=not stap_enabled)

    stap_cfg = STAPConfig(
        enabled=stap_enabled,
        n_taps=stap_taps,
        tap_spacing_ns=stap_delay,
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown('### RF Environment')
    noise_floor = st.sidebar.number_input('Noise Floor (dBm)', -130.0, -90.0, -114.0, step=1.0)
    coupling_mag = st.sidebar.slider('Mutual Coupling |S₂₁|', 0.01, 0.4, 0.15, step=0.01)
    isolation_db = st.sidebar.slider('Return Loss (dB)', -35.0, -10.0, -20.0, step=1.0)
    diag_loading = st.sidebar.slider('Diagonal Loading (dB)', 0.0, 20.0, 3.0, step=0.5,
                                     disabled=(algorithm == NullingAlgorithm.PROJECTION))

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

    S = cached_s_matrix(n, coupling_mag, isolation_db)
    C = s_to_coupling_matrix(S)

    if cfg.algorithm == NullingAlgorithm.PROJECTION:
        w_ideal = projection_matrix_weights(
            positions, freq, cfg.desired_az_deg, cfg.desired_el_deg,
            cfg.jammers, coupling_matrix=None, 
            aep_interpolators=interpolators,
        )
        C_for_pattern = None
    else:
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
        C_for_pattern = C 

    w_quant = quantize_weights(w_ideal, cfg.quant)

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

    # ═══════════ HEADER & METRICS ═══════════
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 5px 0;">
        <h1 style="margin-bottom:2px; font-size:28px;">📡 CRPA NULL STEERING DASHBOARD</h1>
        <p style="color: #7a8ba3; font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 3px;">
            HARDWARE-CONSTRAINED RF BEAMFORMING SIMULATOR
        </p>
    </div>
    """, unsafe_allow_html=True)

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
            null_d = pattern_quant[j0_idx] if cfg.quant.enabled else pattern_ideal[j0_idx]
            st.metric('J1 Null Depth', f'{null_d:.1f} dB')
        else:
            st.metric('Jammers', '0')

    st.markdown('---')

    # ═══════════ TABS ═══════════
    tab_layout, tab_2d, tab_3d, tab_squint, tab_hw, tab_aep, tab_pol, tab_export = st.tabs([
        '🏗️ Layout',
        '📊 2D Pattern',
        '🌐 3D Pattern',
        '📉 Squint',
        '⚙️ Hardware',
        '📐 AEP',
        '🔄 Polarisation',
        '📄 IEEE Exports' # <-- NEW TAB FOR ACADEMIC PUBLICATION PLOTS
    ])

    # [Tabs 1 through 7 remain the same as your original code. I am truncating them here 
    # to highlight the newly added export tab, but they function identically in your local file]

    with tab_layout:
        st.markdown('#### See original code for Layout Tab')
    with tab_2d:
        st.markdown('#### See original code for 2D Pattern Tab')
    with tab_3d:
        st.markdown('#### See original code for 3D Pattern Tab')
    with tab_squint:
        st.markdown('#### See original code for Wideband Squint Tab')
    with tab_hw:
        st.markdown('#### See original code for Hardware Tab')
    with tab_aep:
        st.markdown('#### See original code for AEP Tab')
    with tab_pol:
        st.markdown('#### See original code for Polarisation Tab')

    # ════════ Tab 8: IEEE Paper Exports (NEW) ════════
    with tab_export:
        st.markdown('#### 📄 Publication-Ready Figures')
        st.markdown('Right-click and save the image below. It is formatted specifically for an IEEE two-column paper (e.g., placing it across the top of a page to demonstrate Production-Grade extensions).')
        
        if not cfg.jammers:
            st.warning("⚠️ Please add at least one jammer in the sidebar to generate the degradation and squint plots.")
        else:
            # Setup data for plotting
            j0 = cfg.jammers[0]
            bw_mhz = 15.0
            f_offsets = np.linspace(-bw_mhz * 1e6, bw_mhz * 1e6, 101)
            
            w_eval = w_quant if cfg.quant.enabled else w_ideal
            nd_squint = wideband_null_response(
                w_eval, positions, freq, f_offsets,
                j0.azimuth_deg, j0.elevation_deg,
                coupling_matrix_fc=C_for_pattern,
                aep_interpolators=interpolators,
            )
            
            # Normalize squint correctly
            a_des = steering_vector_with_aep(positions, freq, cfg.desired_az_deg, cfg.desired_el_deg, interpolators)
            if C_for_pattern is not None:
                a_des = coupled_steering_vector(C_for_pattern, a_des)
            p_des = np.abs(w_eval.conj() @ a_des) ** 2
            nd_squint -= 10 * np.log10(max(p_des, 1e-30))

            # --- PLOT STYLING FIX START ---
            # Using plt.style.context('default') ignores Streamlit's dark mode overrides
            with plt.style.context('default'):
                # Matplotlib Figure generation (White background for IEEE PDF)
                fig_ieee, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
                fig_ieee.patch.set_facecolor('white')

                # --- Plot A: Quantization Collapse ---
                ax1.set_facecolor('white')
                ax1.plot(az_sweep, pattern_ideal, label='Ideal Weights', color='blue', linestyle='--')
                if cfg.quant.enabled:
                    ax1.plot(az_sweep, pattern_quant, label=f'{cfg.quant.phase_bits}-bit Quantized', color='red')
                ax1.axvline(x=j0.azimuth_deg, color='black', linestyle=':', label='Jammer DOA')
                
                # Explicitly set all text, ticks, and borders to black
                ax1.set_title('(a) Null Depth Collapse via Quantization', fontsize=11, fontweight='bold', color='black')
                ax1.set_xlabel('Azimuth (°)', fontsize=10, color='black')
                ax1.set_ylabel('Normalized Gain (dB)', fontsize=10, color='black')
                ax1.tick_params(colors='black')
                for spine in ax1.spines.values():
                    spine.set_edgecolor('black')
                
                ax1.set_xlim(0, 360)
                ax1.set_ylim(-65, 5)
                ax1.grid(True, linestyle=':', alpha=0.6, color='gray')
                
                # Enforce black text on legends
                leg1 = ax1.legend(loc='lower right', fontsize=9, facecolor='white', edgecolor='black')
                for text in leg1.get_texts():
                    text.set_color('black')

                # --- Plot B: Wideband Squint ---
                ax2.set_facecolor('white')
                ax2.plot(f_offsets / 1e6, nd_squint, color='purple', linewidth=2)
                ax2.axhline(y=-30, color='red', linestyle='--', alpha=0.5, label='-30 dB Threshold')
                
                # Explicitly set all text, ticks, and borders to black
                ax2.set_title(f'(b) Wideband Squint at Az={j0.azimuth_deg}°', fontsize=11, fontweight='bold', color='black')
                ax2.set_xlabel('Frequency Offset (MHz)', fontsize=10, color='black')
                ax2.set_ylabel('Null Depth (dB)', fontsize=10, color='black')
                ax2.tick_params(colors='black')
                for spine in ax2.spines.values():
                    spine.set_edgecolor('black')
                    
                ax2.set_xlim(-bw_mhz, bw_mhz)
                ax2.set_ylim(-65, 0)
                ax2.grid(True, linestyle=':', alpha=0.6, color='gray')
                
                # Enforce black text on legends
                leg2 = ax2.legend(loc='lower left', fontsize=9, facecolor='white', edgecolor='black')
                for text in leg2.get_texts():
                    text.set_color('black')

                fig_ieee.tight_layout()
                # --- PLOT STYLING FIX END ---
                
                # Display it cleanly in Streamlit
                st.pyplot(fig_ieee)

if __name__ == '__main__':
    main()
