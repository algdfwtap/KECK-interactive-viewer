import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import fnmatch


def find_telemetry_files(night_path, patterns):
    """
    Recursively search for files matching patterns (case-insensitive),
    following symbolic links. Accepts a single pattern string or a list of patterns.
    """
    # If a single string is passed, convert it to a list so we can loop over it
    if isinstance(patterns, str):
        patterns = [patterns]
        
    matches = []
    for root, dirs, files in os.walk(night_path, followlinks=True):
        for filename in files:
            fn_lower = filename.lower()
            # Loop through each pattern in our list
            for pattern in patterns:
                if fnmatch.fnmatch(fn_lower, pattern.lower()):
                    matches.append(Path(root) / filename)
                    break # Stop checking patterns if we already found a match
                    
    return sorted(list(set(matches)))

def find_fits_directories(night_path):
    """Recursively find all .fits directories inside an observing night folder."""
    fits_dirs = []
    for root, dirs, files in os.walk(night_path, followlinks=True):
        for d in dirs:
            if d.endswith('.fits'):
                fits_dirs.append(Path(root) / d)
    return sorted(list(set(fits_dirs)))


# 1. Page Configuration
st.set_page_config(page_title="Keck AO Telemetry Viewer", layout="wide")
st.title("Keck AO Telemetry Viewer")

# 2. Base Data Path & Discovery
base_dir = Path('/g/lu/data/kapa')

if not base_dir.exists():
    st.error(f"Directory not found: `{base_dir}`")
    st.stop()

# Discover observing night folders (filtering out metadata/trash directories)
ignored_folders = {'source_list', 'trash', 'lbwfs_images', 'logs'}
night_dirs = sorted([
    d for d in base_dir.iterdir() 
    if d.is_dir() and not d.name.startswith('.') and d.name not in ignored_folders
], reverse=True)

if not night_dirs:
    st.warning("No night directories found in `/g/lu/data/kapa`")
    st.stop()

# Sidebar directory selector
selected_night = st.sidebar.selectbox(
    "Select Observing Night", 
    [d.name for d in night_dirs]
)
selected_night_path = base_dir / selected_night

# 3. Main Page UI Controls
col1, col2 = st.columns(2)

# Find all FITS exposure runs for the selected night
fits_directories = find_fits_directories(selected_night_path)

with col1:
    if fits_directories:
        selected_fits_dir = st.selectbox(
            "Select FITS Exposure Folder:",
            fits_directories,
            format_func=lambda p: f"{p.parent.name}/{p.name}" if p.parent.name != 'IMAG' else p.name
        )
    else:
        st.warning("No `.fits` directories found for this night.")
        selected_fits_dir = selected_night_path

with col2:
    graph_options = [
        'Up Tip Tilt', 
        'Down Tip Tilt', 
        'Down-TT Strain Gauge', 
        'Up-TT Strain Gauge', 
        'Shack Hartmann WFS Data'
    ]
    selected_graph = st.selectbox("Select Graph Type", graph_options)

st.divider()


# 4. Dashboard Logic
selected_night_path = base_dir / selected_night

if selected_graph == 'Down-TT Strain Gauge':
    # Load BOTH files needed for calibration
    sg_files = find_telemetry_files(selected_fits_dir, ['*clmpstraingauge*.npz', '*straingauge*.npz'])
    dtt_files = find_telemetry_files(selected_fits_dir, ['*downtiptilt*.npz', '*dtt*.npz'])
    
    if not sg_files or not dtt_files:
        st.warning(f"Missing strain gauge or command `.npz` file in `{selected_fits_dir.name}`")
    else:
        try:
            
            # LOAD BOTH DATASETS
            sg_data = np.load(sg_files[0])
            dtt_data = np.load(dtt_files[0])
            st.info(f"Loaded Strain Gauge: `{sg_files[0].name}` | Command: `{dtt_files[0].name}`")

            # ---> ADD THIS WIDGET HERE <---
            selected_subgraphs = st.multiselect(
                "Select Calibration Graphs to View:",
                ['Multi-Axis Calibration (Actual vs Expected)', 'Percent Realized vs Expected Strain'],
                default=['Multi-Axis Calibration (Actual vs Expected)', 'Percent Realized vs Expected Strain']
            )

            # 2. CALIBRATION GRAPHS (Matplotlib)
            if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs or 'Percent Realized vs Expected Strain' in selected_subgraphs:
                st.write("---")
                st.write("### Calibration Analysis")
                
                sg_time = sg_data['rawtimestamp']
                cmd_time = dtt_data['rawtimestamp']
                dtt_strain = sg_data['DTTStrainGauge']
                dtt_cmd = dtt_data['DTTMirrorCommand']
                
                dtt_cmd_aligned = np.zeros((len(sg_time), dtt_cmd.shape[1]))
                for ch in range(dtt_cmd.shape[1]):
                    dtt_cmd_aligned[:, ch] = np.interp(sg_time, cmd_time, dtt_cmd[:, ch])
                    
                dtt_micro = (dtt_strain - dtt_strain[0, :]) * 1e6
                cmd_delta_2d = dtt_cmd_aligned - dtt_cmd_aligned[0, :]
                
                coeff_text = ""
                expected_strains = []
                
                for i in range(dtt_strain.shape[1]):
                    actual_strain = dtt_micro[:, i]
                    X = np.c_[cmd_delta_2d, np.ones(cmd_delta_2d.shape[0])]
                    coeffs, _, _, _ = np.linalg.lstsq(X, actual_strain, rcond=None)
                    expected_strains.append(X @ coeffs)
                    coeff_text += f"**Channel {i}:** Tip Sensitivity: `{coeffs[0]:.4f}`, Tilt Sensitivity: `{coeffs[1]:.4f}`, Intercept: `{coeffs[2]:.4f}`\n\n"
                
                with st.expander("View Regression Coefficients (με / arcsec)"):
                    st.markdown(coeff_text)
                
                cols = st.columns(2)
                
                if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs:
                    with cols[0]:
                        fig1, ax1 = plt.subplots(figsize=(7, 5))
                        axis_limits = [np.min(dtt_micro), np.max(dtt_micro)]
                        ax1.plot(axis_limits, axis_limits, 'k--', linewidth=1.5, label='Ideal 1:1 Response')
                        
                        for i in range(dtt_strain.shape[1]):
                            ax1.scatter(expected_strains[i], dtt_micro[:, i], s=2, alpha=0.15, label=f'Ch {i} Data')
                            
                        ax1.set_xlabel(r'Expected Strain (From Tip/Tilt mapping) [$\mu\epsilon$]')
                        ax1.set_ylabel(r'Actual Measured Strain [$\mu\epsilon$]')
                        ax1.set_title('Multi-Axis Calibration (DTT)', fontweight='bold')
                        ax1.legend(loc='upper left')
                        ax1.grid(True, alpha=0.3)
                        st.pyplot(fig1)
                        
                if 'Percent Realized vs Expected Strain' in selected_subgraphs:
                    col_idx = 1 if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs else 0
                    with cols[col_idx]:
                        fig2, ax2 = plt.subplots(figsize=(7, 5))
                        ax2.axhline(100, color='r', linestyle='--', linewidth=1.5, label='100% Target')
                        
                        for i in range(dtt_strain.shape[1]):
                            actual_strain = dtt_micro[:, i]
                            expected_strain = expected_strains[i]
                            
                            threshold = np.max(np.abs(expected_strain)) * 0.05
                            valid_mask = np.abs(expected_strain) > threshold
                            
                            pct_moved = np.full_like(actual_strain, np.nan)
                            pct_moved[valid_mask] = (actual_strain[valid_mask] / expected_strain[valid_mask]) * 100.0
                            
                            ax2.scatter(expected_strain[valid_mask], pct_moved[valid_mask], s=3, alpha=0.2, label=f'DTT Ch {i}')
                            
                        ax2.set_xlabel(r'Expected Strain [$\mu\epsilon$]')
                        ax2.set_ylabel('Percent Moved (% Realized)')
                        ax2.set_title('Percent Moved vs. Expected Strain (DTT)', fontweight='bold')
                        ax2.set_ylim(50, 150)
                        ax2.legend(loc='upper right')
                        ax2.grid(True, alpha=0.3)
                        st.pyplot(fig2)
                        
        except Exception as e:
            st.error(f"Error reading or plotting data: {e}")
            
            

elif selected_graph == 'Up-TT Strain Gauge':
    # Mirroring the exact same logic for Up Tip-Tilt using our robust file finder
    sg_files = find_telemetry_files(selected_fits_dir, ['*clmpstraingauge*.npz', '*straingauge*.npz'])
    utt_files = find_telemetry_files(selected_fits_dir, ['*uptiptilt*.npz', '*utt*.npz'])
    
    if not sg_files or not utt_files:
        st.warning(f"Missing strain gauge or Up Tip-Tilt `.npz` file in `{selected_fits_dir.name}`")
    else:
        try:
            # LOAD BOTH DATASETS
            sg_data = np.load(sg_files[0])
            utt_data = np.load(utt_files[0])
            st.info(f"Loaded Strain Gauge: `{sg_files[0].name}` | Command: `{utt_files[0].name}`") 
            
            st.write("### Select Graphs to Display")
            graph_choices = [
                'Strain Gauge Sequence (By Sample)',
                'Multi-Axis Calibration (Actual vs Expected)',
                'Percent Realized vs Expected Strain'
            ]
            selected_subgraphs = st.multiselect("Check the graphs you want to view:", graph_choices, default=graph_choices)
            
            if not selected_subgraphs:
                st.warning("Please select at least one graph.")
            else:
                if 'Strain Gauge Sequence (By Sample)' in selected_subgraphs:
                    utt_strain = sg_data['UTTMainStrainGauge']
                    x_axis = np.arange(utt_strain.shape[0])
                    
                    st.write("*(Use the slider below the graph to scrub and zoom)*")
                    fig = go.Figure()
                    for i in range(utt_strain.shape[1]):
                        fig.add_trace(go.Scatter(x=x_axis, y=utt_strain[:, i], mode='lines', name=f'Channel {i}', line=dict(width=1), opacity=0.7))
                    
                    fig.update_layout(
                        title=dict(text="<b>Up-TT Strain Gauge Sequence</b>", x=0.5, font=dict(size=16)),
                        xaxis_title="Sample Number", yaxis_title="Strain", height=400,
                        margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig, use_container_width=True)

                if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs or 'Percent Realized vs Expected Strain' in selected_subgraphs:
                    st.write("---")
                    st.write("### Calibration Analysis")
                    
                    sg_time = sg_data['rawtimestamp']
                    cmd_time = utt_data['rawtimestamp']
                    utt_strain = sg_data['UTTMainStrainGauge']
                    utt_cmd = utt_data['UTTMainMirrorCommand']
                    
                    utt_cmd_aligned = np.zeros((len(sg_time), utt_cmd.shape[1]))
                    for ch in range(utt_cmd.shape[1]):
                        utt_cmd_aligned[:, ch] = np.interp(sg_time, cmd_time, utt_cmd[:, ch])
                        
                    utt_micro = (utt_strain - utt_strain[0, :]) * 1e6
                    cmd_delta_2d = utt_cmd_aligned - utt_cmd_aligned[0, :]
                    
                    coeff_text = ""
                    expected_strains = []
                    
                    for i in range(utt_strain.shape[1]):
                        actual_strain = utt_micro[:, i]
                        X = np.c_[cmd_delta_2d, np.ones(cmd_delta_2d.shape[0])]
                        coeffs, _, _, _ = np.linalg.lstsq(X, actual_strain, rcond=None)
                        expected_strains.append(X @ coeffs)
                        coeff_text += f"**Channel {i}:** Tip Sensitivity: `{coeffs[0]:.4f}`, Tilt Sensitivity: `{coeffs[1]:.4f}`, Intercept: `{coeffs[2]:.4f}`\n\n"
                    
                    with st.expander("View Regression Coefficients (με / arcsec)"):
                        st.markdown(coeff_text)
                    
                    cols = st.columns(2)
                    
                    if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs:
                        with cols[0]:
                            fig1, ax1 = plt.subplots(figsize=(7, 5))
                            axis_limits = [np.min(utt_micro), np.max(utt_micro)]
                            ax1.plot(axis_limits, axis_limits, 'k--', linewidth=1.5, label='Ideal 1:1 Response')
                            
                            for i in range(utt_strain.shape[1]):
                                ax1.scatter(expected_strains[i], utt_micro[:, i], s=2, alpha=0.15, label=f'Ch {i} Data')
                                
                            ax1.set_xlabel(r'Expected Strain (From Tip/Tilt mapping) [$\mu\epsilon$]')
                            ax1.set_ylabel(r'Actual Measured Strain [$\mu\epsilon$]')
                            ax1.set_title('Multi-Axis Calibration (UTT)', fontweight='bold')
                            ax1.legend(loc='upper left')
                            ax1.grid(True, alpha=0.3)
                            st.pyplot(fig1)
                            
                    if 'Percent Realized vs Expected Strain' in selected_subgraphs:
                        col_idx = 1 if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs else 0
                        with cols[col_idx]:
                            fig2, ax2 = plt.subplots(figsize=(7, 5))
                            ax2.axhline(100, color='r', linestyle='--', linewidth=1.5, label='100% Target')
                            
                            for i in range(utt_strain.shape[1]):
                                actual_strain = utt_micro[:, i]
                                expected_strain = expected_strains[i]
                                
                                threshold = np.max(np.abs(expected_strain)) * 0.05
                                valid_mask = np.abs(expected_strain) > threshold
                                
                                pct_moved = np.full_like(actual_strain, np.nan)
                                pct_moved[valid_mask] = (actual_strain[valid_mask] / expected_strain[valid_mask]) * 100.0
                                
                                ax2.scatter(expected_strain[valid_mask], pct_moved[valid_mask], s=3, alpha=0.2, label=f'UTT Ch {i}')
                                
                            ax2.set_xlabel(r'Expected Strain [$\mu\epsilon$]')
                            ax2.set_ylabel('Percent Moved (% Realized)')
                            ax2.set_title('Percent Moved vs. Expected Strain (UTT)', fontweight='bold')
                            ax2.set_ylim(50, 150)
                            ax2.legend(loc='upper right')
                            ax2.grid(True, alpha=0.3)
                            st.pyplot(fig2)
                            
        except Exception as e:
            st.error(f"Error reading or plotting data: {e}")

            
            
            
elif selected_graph == 'Up Tip Tilt':
    # Use the robust file finder to handle symlinks correctly
    utt_files = find_telemetry_files(selected_fits_dir, ['*uptiptilt*.npz', '*utt*.npz'])
    
    if not utt_files:
        st.warning(f"No up tip-tilt `.npz` file found in `{selected_fits_dir.name}`")
    else:
        file_path = utt_files[0]
        st.info(f"Loaded file: `{file_path.name}`")
        
        try:
            utt_data = np.load(file_path)
            
            # --- DYNAMIC KEY FINDER ---
            # Define variations of the array names you might encounter
            possible_slope_keys = ['SHWFS1TT', 'SHWFSTT', 'TTslope', 'TTslope1']
            possible_mirror_keys = ['UTTMainMirrorCommand', 'UTTMirrorCommand', 'MirrorCommand']
            
            # Find the first matching key in the file
            slope_key = next((key for key in possible_slope_keys if key in utt_data.files), None)
            mirror_key = next((key for key in possible_mirror_keys if key in utt_data.files), None)
            
            if not slope_key or not mirror_key:
                st.write("Available keys in this file:", utt_data.files)
                raise KeyError("Could not find expected array names for slopes or mirror commands.")
                
            # Extract data using the dynamically found names
            tt_slopes = utt_data[slope_key]  
            mirror_commands = utt_data[mirror_key]  
            
            tip = tt_slopes[:, 0]
            tilt = tt_slopes[:, 1]
            mirror_tip = mirror_commands[:, 0]
            mirror_tilt = mirror_commands[:, 1]
            total_samples = len(tip)
            x_axis = np.arange(total_samples)
            
            
            # --- DYNAMIC GRAPH SELECTOR ---
            st.write("### Select Graphs to Display")
            graph_choices = [
                'Tip Distribution (Histogram)',
                'Tilt Distribution (Histogram)',
                'Tip Sequence (By Sample)',
                'Tilt Sequence (By Sample)',
                'Mirror Correction: Tip',
                'Mirror Correction: Tilt'
            ]
            
            selected_subgraphs = st.multiselect("Check the graphs you want to view:", graph_choices, default=graph_choices)
            
            if not selected_subgraphs:
                st.warning("Please select at least one graph from the dropdown menu above.")
            else:
                st.write("*(Tip: Sequence graphs are interactive—click and drag to zoom/pan. Histograms are static.)*"
)
                
                # --- HYBRID PLOTTING (Matplotlib + Plotly) ---
                for i in range(0, len(selected_subgraphs), 2):
                    cols = st.columns(2)
                    
                    for j in range(2):
                        if i + j < len(selected_subgraphs):
                            graph_name = selected_subgraphs[i + j]
                            
                            with cols[j]:
                                
                                # 1. MATPLOTLIB FOR HISTOGRAMS
                                if 'Histogram' in graph_name:
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    
                                    if graph_name == 'Tip Distribution (Histogram)':
                                        ax.hist(tip, bins=50, edgecolor='black', alpha=0.7)
                                        ax.set_xlabel('Tip Value (arcseconds)')
                                        ax.set_ylabel('Count')
                                        ax.set_title('Tip Distribution', fontweight='bold')
                                    elif graph_name == 'Tilt Distribution (Histogram)':
                                        ax.hist(tilt, bins=50, edgecolor='black', alpha=0.7, color='red')
                                        ax.set_xlabel('Tilt Value (arcseconds)')
                                        ax.set_ylabel('Count')
                                        ax.set_title('Tilt Distribution', fontweight='bold')
                                        
                                    ax.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # 2. PLOTLY FOR SEQUENCE GRAPHS
                                else:
                                    fig = go.Figure()
                                    
                                    if graph_name == 'Tip Sequence (By Sample)':
                                        fig.add_trace(go.Scatter(x=x_axis, y=tip, mode='lines', line=dict(color='#1f77b4', width=1)))
                                        title = 'Tip Sequence (By Sample)'
                                    elif graph_name == 'Tilt Sequence (By Sample)':
                                        fig.add_trace(go.Scatter(x=x_axis, y=tilt, mode='lines', line=dict(color='#d62728', width=1)))
                                        title = 'Tilt Sequence (By Sample)'
                                    elif graph_name == 'Mirror Correction: Tip':
                                        fig.add_trace(go.Scatter(x=x_axis, y=mirror_tip, mode='lines', line=dict(color='#1f77b4', width=1)))
                                        title = 'Mirror Correction: Tip'
                                    elif graph_name == 'Mirror Correction: Tilt':
                                        fig.add_trace(go.Scatter(x=x_axis, y=mirror_tilt, mode='lines', line=dict(color='#d62728', width=1)))
                                        title = 'Mirror Correction: Tilt'
                                        
                                    fig.update_layout(
                                        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14)),
                                        xaxis_title="Sample Number",
                                        yaxis_title="Command (arcseconds)",
                                        height=350,
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        showlegend=False,
                                        hovermode="x unified"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
        except KeyError as e:
            st.error(f"Data format error: Missing expected array {e} in {file_path.name}")
        except Exception as e:
            st.error(f"Error reading or plotting data: {e}")
            
            
            
    
    
elif selected_graph == 'Down Tip Tilt':
    # Use the robust file finder to handle symlinks correctly
    dtt_files = find_telemetry_files(selected_fits_dir, ['*downtiptilt*.npz', '*dtt*.npz'])
    
    if not dtt_files:
        st.warning(f"No down tip-tilt `.npz` file found in `{selected_fits_dir.name}`")
    else:
        file_path = dtt_files[0]
        st.info(f"Loaded file: `{file_path.name}`")
        
        try:
            dtt_data = np.load(file_path)
            
            # --- DYNAMIC KEY FINDER ---
            possible_slope_keys = ['TTslope1', 'TTslope', 'TTSLOPE1', 'TTSLOPE']
            slope_key = next((key for key in possible_slope_keys if key in dtt_data.files), None)
            
            if not slope_key:
                raise KeyError(f"No valid slope key found. Available keys are: {dtt_data.files}")
                
            tt_slopes = dtt_data[slope_key]  
            mirror_commands = dtt_data['DTTMirrorCommand']  
            
            tip = tt_slopes[:, 0]
            tilt = tt_slopes[:, 1]
            mirror_tip = mirror_commands[:, 0]
            mirror_tilt = mirror_commands[:, 1]
            total_samples = len(tip)
            x_axis = np.arange(total_samples)
            
            # --- DYNAMIC GRAPH SELECTOR ---
            st.write("### Select Graphs to Display")
            graph_choices = [
                'Tip Distribution (Histogram)',
                'Tilt Distribution (Histogram)',
                'Tip Sequence (By Sample)',
                'Tilt Sequence (By Sample)',
                'Mirror Correction: Tip',
                'Mirror Correction: Tilt'
            ]
            
            selected_subgraphs = st.multiselect("Check the graphs you want to view:", graph_choices, default=graph_choices)
            
            if not selected_subgraphs:
                st.warning("Please select at least one graph from the dropdown menu above.")
            else:
                st.write("*(Tip: Sequence graphs are interactive—click and drag to zoom/pan. Histograms are static.)*")
                
                # --- HYBRID PLOTTING (Matplotlib + Plotly) ---
                # We build the layout row by row, 2 columns per row
                for i in range(0, len(selected_subgraphs), 2):
                    cols = st.columns(2)
                    
                    for j in range(2):
                        if i + j < len(selected_subgraphs):
                            graph_name = selected_subgraphs[i + j]
                            
                            # Place the plot inside the current column
                            with cols[j]:
                                
                                # 1. MATPLOTLIB FOR HISTOGRAMS
                                if 'Histogram' in graph_name:
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    
                                    if graph_name == 'Tip Distribution (Histogram)':
                                        ax.hist(tip, bins=50, edgecolor='black', alpha=0.7)
                                        ax.set_xlabel('Tip Value (arcseconds)')
                                        ax.set_ylabel('Count')
                                        ax.set_title('Tip Distribution', fontweight='bold')
                                    elif graph_name == 'Tilt Distribution (Histogram)':
                                        ax.hist(tilt, bins=50, edgecolor='black', alpha=0.7, color='red')
                                        ax.set_xlabel('Tilt Value (arcseconds)')
                                        ax.set_ylabel('Count')
                                        ax.set_title('Tilt Distribution', fontweight='bold')
                                        
                                    ax.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # 2. PLOTLY FOR SEQUENCE GRAPHS
                                else:
                                    fig = go.Figure()
                                    
                                    if graph_name == 'Tip Sequence (By Sample)':
                                        fig.add_trace(go.Scatter(x=x_axis, y=tip, mode='lines', line=dict(color='#1f77b4', width=1)))
                                        title = 'Tip Sequence (By Sample)'
                                    elif graph_name == 'Tilt Sequence (By Sample)':
                                        fig.add_trace(go.Scatter(x=x_axis, y=tilt, mode='lines', line=dict(color='#d62728', width=1)))
                                        title = 'Tilt Sequence (By Sample)'
                                    elif graph_name == 'Mirror Correction: Tip':
                                        fig.add_trace(go.Scatter(x=x_axis, y=mirror_tip, mode='lines', line=dict(color='#1f77b4', width=1)))
                                        title = 'Mirror Correction: Tip'
                                    elif graph_name == 'Mirror Correction: Tilt':
                                        fig.add_trace(go.Scatter(x=x_axis, y=mirror_tilt, mode='lines', line=dict(color='#d62728', width=1)))
                                        title = 'Mirror Correction: Tilt'
                                        
                                    fig.update_layout(
                                        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14)),
                                        xaxis_title="Sample Number",
                                        yaxis_title="Command (arcseconds)",
                                        height=350, # Matches rough height of matplotlib figures
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        showlegend=False,
                                        hovermode="x unified"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
        except Exception as e:
            st.error(f"Error reading or plotting data: {e}")
            
            
            
            
            
elif selected_graph in ['Shack Hartmann WFS Data', 'SHWFS']:
    # Search for both SHWFS and ocam2k file naming conventions
    shwfs_files = find_telemetry_files(selected_fits_dir, ['*SHWFS*.npz', '*shwfs*.npz', '*ocam2k*.npz', '*OCAM2K*.npz'])
    
    if not shwfs_files:
        st.warning(f"No SHWFS or ocam2k `.npz` file found in `{selected_fits_dir.name}`")
    else:
        wfs_file = shwfs_files[0]
        st.info(f"Loaded: `{wfs_file.name}`")
        
        try:
            wfs_data = np.load(wfs_file)
            
            # --- SMART KEY DETECTION ---
            # 1. Look for image arrays
            img_key = next((k for k in wfs_data.files if 'image' in k.lower() or 'img' in k.lower()), None)
            if not img_key:
                # Fallback: look for 2D/3D arrays that aren't timestamps, centroids, or subapertures
                for k in wfs_data.files:
                    if wfs_data[k].ndim >= 2 and not any(x in k.lower() for x in ['timestamp', 'centroid', 'subaper']):
                        img_key = k
                        break
                        
            # 2. Look for telemetry arrays (Centroids and Subaperture Flux)
            centroid_key = next((k for k in wfs_data.files if 'centroid' in k.lower()), None)
            subap_key = next((k for k in wfs_data.files if 'subaper' in k.lower()), None)
            
            # --- DYNAMIC GRAPH MENU ---
            # Build the available options based on exactly what data is in this file
            graph_choices = []
            if img_key:
                graph_choices.extend([
                    'Average Intensity Over Time (Image)',
                    'Single Frame Viewer (Spot Pattern)',
                    'Multi-Frame Sample Grid (2x3)'
                ])
            if subap_key:
                graph_choices.append('Average Subaperture Flux Over Time')
            if centroid_key:
                graph_choices.extend([
                    'Centroid Distribution (Histogram)',
                ])
                
            if not graph_choices:
                st.warning("Could not find recognizable image, centroid, or subaperture arrays in this file.")
                st.write("Available keys:", wfs_data.files)
            else:
                st.write("### Select graphs")
                selected_subgraphs = st.multiselect("Select Graphs to display:", graph_choices, default=graph_choices)
                
                if not selected_subgraphs:
                    st.warning("Please select at least one graph.")
                else:
                    
                    # ==========================================
                    # 1. RAW IMAGE PLOTTING BLOCK
                    # ==========================================
                    if img_key and any('Image' in g or 'Frame' in g for g in selected_subgraphs):
                        raw_images = wfs_data[img_key]
                        n_frames = raw_images.shape[0]
                        
                        # Dynamic Reshape Logic for non-square pixel counts
                        if raw_images.ndim == 3:
                            images_reshaped = raw_images
                        else:
                            n_pixels = raw_images.shape[1]
                            best_h, best_w = 1, n_pixels
                            root = int(np.sqrt(n_pixels))
                            if root * root == n_pixels:
                                best_h, best_w = root, root
                            else:
                                for i in range(root, 0, -1):
                                    if n_pixels % i == 0:
                                        best_h = i
                                        best_w = n_pixels // i
                                        break
                            images_reshaped = raw_images.reshape(n_frames, best_h, best_w)
                        
                        abs_min = float(np.min(images_reshaped))
                        abs_max = float(np.max(images_reshaped))
                        default_p_min = float(np.percentile(images_reshaped, 0.5))
                        default_p_max = float(np.percentile(images_reshaped, 99.5))
                        
                        st.write("---")
                        with st.expander("Image Color Scaling & Thresholds", expanded=True):
                            v_min, v_max = st.slider(
                                "Colorbar Limits (Counts)", min_value=abs_min, max_value=abs_max,
                                value=(default_p_min, default_p_max),
                                step=(abs_max - abs_min) / 500.0 if abs_max > abs_min else 1.0
                            )
                        
                        if 'Average Intensity Over Time (Image)' in selected_subgraphs:
                            st.write("---")
                            mean_intensity = images_reshaped.mean(axis=(1, 2))
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(mean_intensity, color='#1f77b4', linewidth=1.5)
                            ax.set_xlabel('Frame Number')
                            ax.set_ylabel('Mean Intensity (counts)')
                            ax.set_title(f'Average Intensity Over Time ({img_key})', fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                        if 'Single Frame Viewer (Spot Pattern)' in selected_subgraphs:
                            st.write("---")
                            st.write("### Single SHWFS Frame")
                            frame_idx = st.slider("Scrub through frames:", 0, n_frames - 1, 0)
                            fig, ax = plt.subplots(figsize=(7, 7))
                            im = ax.imshow(images_reshaped[frame_idx], cmap='hot', vmin=v_min, vmax=v_max)
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity (counts)")
                            h, w = images_reshaped.shape[1], images_reshaped.shape[2]
                            ax.set_title(f'SHWFS Frame {frame_idx} [{h}x{w}]', fontweight='bold')
                            st.pyplot(fig)
                            
                        if 'Multi-Frame Sample Grid (2x3)' in selected_subgraphs:
                            st.write("---")
                            st.write("### SHWFS Frame Sequence Grid")
                            step = max(1, (n_frames - 1) // 5)
                            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                            for i, ax in enumerate(axes.flat):
                                idx = min(i * step, n_frames - 1)
                                ax.imshow(images_reshaped[idx], cmap='hot', vmin=v_min, vmax=v_max)
                                ax.set_title(f'Frame {idx}')
                                ax.axis('off')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                    # ==========================================
                    # 2. TELEMETRY PLOTTING BLOCK
                    # ==========================================
                    if subap_key and 'Average Subaperture Flux Over Time' in selected_subgraphs:
                        st.write("---")
                        subap_data = wfs_data[subap_key]
                        mean_flux = subap_data.mean(axis=1)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(mean_flux, color='#2ca02c', linewidth=1.5)
                        ax.set_xlabel('Frame Number')
                        ax.set_ylabel('Mean Flux')
                        ax.set_title(f'Average Subaperture Flux ({subap_key})', fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                    if centroid_key and any('Centroid' in g for g in selected_subgraphs):
                        st.write("---")
                        centroid_data = wfs_data[centroid_key]
                        n_frames = centroid_data.shape[0]
                        x_axis = np.arange(n_frames)
                        cols = st.columns(2)
                        
                        if 'Centroid Distribution (Histogram)' in selected_subgraphs:
                            with cols[0]:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.hist(centroid_data.flatten(), bins=50, edgecolor='black', alpha=0.7, color='purple')
                                ax.set_xlabel('Centroid Value')
                                ax.set_ylabel('Count')
                                ax.set_title('Overall Centroid Distribution', fontweight='bold')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                
                       
        except Exception as e:
            st.error(f"Error processing SHWFS file: {e}")