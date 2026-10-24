import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import fnmatch
from astropy.io import fits



def find_telemetry_files(search_target, patterns):
    """
    Recursively search for files matching patterns (case-insensitive),
    following symbolic links. 
    'search_target' can be a directory path (str/Path) OR a list of file Paths.
    Accepts a single pattern string or a list of patterns.
    """
    # If a single string is passed, convert it to a list so we can loop over it
    if isinstance(patterns, str):
        patterns = [patterns]
        
    matches = []
    
    # BRANCH 1: If we passed in a pre-filtered list of files
    if isinstance(search_target, list):
        for filepath in search_target:
            fn_lower = filepath.name.lower()
            for pattern in patterns:
                if fnmatch.fnmatch(fn_lower, pattern.lower()):
                    matches.append(filepath)
                    break
                    
    # BRANCH 2: If we passed in a directory path, do the original os.walk
    else:
        for root, dirs, files in os.walk(search_target, followlinks=True):
            for filename in files:
                fn_lower = filename.lower()
                for pattern in patterns:
                    if fnmatch.fnmatch(fn_lower, pattern.lower()):
                        matches.append(Path(root) / filename)
                        break
                        
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
 


    
    
# 2. Select NGS or LGS
mode_filter = st.radio(
    "Filter by Guide Star Mode:", 
    ["All", "LGS", "NGS"], 
    horizontal=True
)    
    
# Pre-filter the FITS directories so empty ones don't show up in the dropdown
all_fits_directories = find_fits_directories(selected_night_path)
filtered_dirs = []

if mode_filter == "All":
    filtered_dirs = all_fits_directories
else:
    for d in all_fits_directories:
        # Keep directory only if it contains at least one matching .npz file
        if any(f.name.upper().startswith(mode_filter) for f in d.rglob('*.npz')):
            filtered_dirs.append(d)

st.divider()

# 3 & 4. Select Filtered Folder and Graph Type
col1, col2 = st.columns(2)

with col1:
    if not filtered_dirs:
        st.warning(f"No folders containing {mode_filter} data found for this night.")
        selected_fits_dir = selected_night_path # Fallback to avoid breaking downstream code
    else:
        selected_fits_dir = st.selectbox(
            "Select FITS Exposure Folder:",
            filtered_dirs,
            format_func=lambda p: f"{p.parent.name}/{p.name}" if p.parent.name != 'IMAG' else p.name
        )

with col2:
    # Base list of all possible graphs
    graph_options = [
        'Up Tip Tilt', 
        'Down Tip Tilt', 
        'Down-TT Strain Gauge', 
        'Up-TT Strain Gauge', 
        'Shack Hartmann WFS Data'
    ]
    
    # Dynamically remove Up-TT graphs if NGS is selected
    if mode_filter == "NGS":
        graph_options.remove('Up Tip Tilt')
        graph_options.remove('Up-TT Strain Gauge')

    selected_graph = st.selectbox("Select Graph Type:", graph_options)

# Generate the filtered file list for the chosen folder to feed into your graph blocks
if selected_fits_dir.exists():
    all_files_in_dir = list(selected_fits_dir.rglob('*.npz'))
    if mode_filter == "All":
        filtered_files = all_files_in_dir
    else:
        filtered_files = [f for f in all_files_in_dir if f.name.upper().startswith(mode_filter)]
else:
    filtered_files = []
    
    
    


# 4. Dashboard Logic
selected_night_path = base_dir / selected_night

#print keys
st.write("---")
with st.expander("View Available Data Keys for this Night", expanded=False):
    # Find all npz files in the selected night directory (including subdirectories)
    all_npz_files = list(selected_fits_dir.rglob('*.npz'))
    
    if not all_npz_files:
        st.info(f"No `.npz` files found in `{selected_fits_dir.name}`")
    else:
        st.write(f"Found {len(all_npz_files)} `.npz` files. Expand to see available arrays:")
        
        # Loop through each file and quickly grab its keys
        for fpath in all_npz_files:
            try:
                # np.load on a zip file reads the header/keys instantly
                temp_data = np.load(fpath)
                
                # Display the filename in bold, followed by its keys
                st.markdown(f"**`{fpath.name}`**")
                st.write(temp_data.files)
                
                # Check for DTT array and print its length and NaN count
                if 'DTTStrainGauge' in temp_data.files:
                    dtt_array = temp_data['DTTStrainGauge']
                    dtt_length = len(dtt_array)
                    nan_count = np.isnan(dtt_array).sum()
                    st.markdown(f"**`DTTStrainGauge` length:** `{dtt_length}` | **NaN Count:** `{nan_count}`")
                
                if 'rawtimestamp' in temp_data.files:
                    time_length = len(temp_data['rawtimestamp'])
                    st.markdown(f"**`time` length:** `{time_length}`")
                
                # Close the file object to prevent memory leaks
                temp_data.close() 
            except Exception as e:
                st.error(f"**`{fpath.name}`**: Could not read file ({e})")
st.write("---")




if selected_graph == 'Down-TT Strain Gauge':
    # Load BOTH files needed for calibration
    sg_files = find_telemetry_files(filtered_files, ['*clmpstraingauge*.npz', '*straingauge*.npz'])
    dtt_files = find_telemetry_files(filtered_files, ['*downtiptilt*.npz', '*dtt*.npz'])
    
    if not sg_files or not dtt_files:
        st.warning(f"Missing strain gauge or command `.npz` file in `{selected_fits_dir.name}`")
    else:
        try:
    
            
            # LOAD BOTH DATASETS
            sg_data = np.load(sg_files[0])
            dtt_data = np.load(dtt_files[0])
            
    
            
            
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
                
                fig, ax = plt.subplots(figsize=(8, 4))
              

                #WHY ARE MY DTT STRAIN VERTICAL?
                ax.plot(dtt_cmd)
                ax.set_title(f"Raw DTT Mirror Commands: {dtt_files[0].name}")
                ax.set_xlabel("Index")
                ax.set_ylabel("Command Value")
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(dtt_strain)
                ax.set_title(f"Raw DTT Strain: {dtt_files[0].name}")
                ax.set_xlabel("Index")
                ax.set_ylabel("Command Value")
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

                
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
    sg_files = find_telemetry_files(filtered_files, ['*clmpstraingauge*.npz', '*straingauge*.npz'])
    utt_files = find_telemetry_files(filtered_files, ['*uptiptilt*.npz', '*utt*.npz', 'UTTMainStrainGauge'])
    
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
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(utt_cmd)
                    ax.set_title(f"Raw UTT Mirror Commands: {utt_files[0].name}")
                    ax.set_xlabel("Index")
                    ax.set_ylabel("Command Value")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                    
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(utt_strain)
                    ax.set_title(f"Raw UTT Strain: {utt_files[0].name}")
                    ax.set_xlabel("Index")
                    ax.set_ylabel("Command Value")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                    
                    
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
    utt_files = find_telemetry_files(filtered_files, ['*uptiptilt*.npz', '*utt*.npz'])
    
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
    dtt_files = find_telemetry_files(filtered_files, ['*downtiptilt*.npz', '*dtt*.npz'])
    
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
    # 1. Search for both files separately (Case-Insensitive)
    raw_files = [f for f in filtered_files if 'raw' in f.name.lower() and 'ocam2k' in f.name.lower()]
    proc_files = [f for f in filtered_files if 'ocam2k' in f.name.lower() and 'raw' not in f.name.lower()]
    
    if not raw_files and not proc_files:
        st.warning(f"No SHWFS `.npz` files found in `{selected_fits_dir.name}`")
    else:
        try:
            # Load whichever files exist
            raw_data = np.load(raw_files[0]) if raw_files else None
            proc_data = np.load(proc_files[0]) if proc_files else None
            
            # --- ROUTE THE IMAGE SOURCE ---
            image_source_data = raw_data if raw_data is not None else proc_data
            
            raw_name = raw_files[0].name if raw_files else (proc_files[0].name if proc_files else 'None')
            proc_name = proc_files[0].name if proc_files else 'None'
            
# ==========================================
            # 1. GLOBAL SUBAPERTURE FLUX MAP
            # ==========================================
            keys_to_plot = ["SHWFS1SubAper", "SHWFS2SubAper", "SHWFS3SubAper", "SHWFS4SubAper"]
            valid_keys = [k for k in keys_to_plot if proc_data and k in proc_data.files]
            
            if valid_keys:
                st.write("---")
                
                # --- NEW: DISPLAY MODE TOGGLE ---
                if len(valid_keys) > 1:
                    display_mode = st.radio(
                        "Flux Map Display Mode:", 
                        ["Single Sensor (SHWFS1)", "All Valid Sensors (Synchronized last 1000 frames)"],
                        horizontal=True
                    )
                    active_keys = [valid_keys[0]] if "Single Sensor" in display_mode else valid_keys
                    sync_last_1000 = "All Valid Sensors" in display_mode
                else:
                    active_keys = valid_keys
                    sync_last_1000 = False
                    
                st.write(f"### Subaperture Flux Map ({len(active_keys)} Sensor{'s' if len(active_keys)>1 else ''})")
                
                # --- ALIGNMENT LOGIC ---
                if sync_last_1000:
                    # Find the minimum available length across sensors to prevent out-of-bounds
                    min_len = min([proc_data[k].shape[0] for k in active_keys])
                    n_telemetry_frames = min(1000, min_len)
                    st.info(f"Viewing the final {n_telemetry_frames} synchronized frames.")
                else:
                    n_telemetry_frames = proc_data[active_keys[0]].shape[0]

                frame_idx_slider = st.slider("Select Frame:", 0, n_telemetry_frames - 1, 0, key="global_subap_slider")
                
                # --- GLOBAL COLORBAR LIMITS ---
                # Gather only the data that will actually be displayed for limit calculation
                if sync_last_1000:
                    all_sensor_data = np.concatenate([proc_data[k][-n_telemetry_frames:] for k in active_keys])
                else:
                    all_sensor_data = proc_data[active_keys[0]]
                    
                positive_data = all_sensor_data[all_sensor_data > 0]
                
                if len(positive_data) > 0:
                    abs_min_sub = float(np.min(positive_data))
                    abs_max_sub = float(np.max(positive_data))
                    default_min_sub = float(np.percentile(positive_data, 5))
                    default_max_sub = float(np.percentile(positive_data, 99.5))
                else:
                    abs_min_sub, abs_max_sub, default_min_sub, default_max_sub = 1.0, 1000.0, 1.0, 1000.0
                    
                with st.expander("Subaperture Color Scaling & Thresholds", expanded=False):
                    v_min_sub, v_max_sub = st.slider(
                        "Flux Map Colorbar Limits (Log Scale)", 
                        min_value=abs_min_sub, max_value=abs_max_sub,
                        value=(default_min_sub, default_max_sub),
                        step=(abs_max_sub - abs_min_sub) / 500.0 if abs_max_sub > abs_min_sub else 1.0
                    )

                maps_with_data = []
                map_path = "/u/bdigia/code/python/paarti/paarti/utils/sub_ap_map.txt"
                
                try:
                    subap_map = np.loadtxt(map_path)
                    
                    for key in active_keys:
                        # If synced, calculate the true index by counting backwards from the end of THIS specific array
                        if sync_last_1000:
                            actual_idx = proc_data[key].shape[0] - n_telemetry_frames + frame_idx_slider
                        else:
                            actual_idx = frame_idx_slider
                            
                        copy_of_aperture_data = np.copy(proc_data[key][actual_idx]).tolist()
                        copy_of_subap_map = np.copy(subap_map) 

                        for i in range(subap_map.shape[0]):
                            for j in range(subap_map.shape[1]):
                                if subap_map[i, j] == 1:
                                    if copy_of_aperture_data:
                                        copy_of_subap_map[i, j] = copy_of_aperture_data.pop(0)

                        map_with_data = np.where(copy_of_subap_map == 0, np.nan, copy_of_subap_map)
                        maps_with_data.append(map_with_data)

                    if maps_with_data:
                        from matplotlib.colors import LogNorm
                        cmap = plt.cm.viridis.copy()
                        cmap.set_bad(color="black")
                        
                        if len(active_keys) == 1:
                            fig, ax = plt.subplots(figsize=(6, 5))
                            im = ax.imshow(maps_with_data[0], norm=LogNorm(vmin=v_min_sub, vmax=v_max_sub), cmap=cmap)
                            ax.set_title(active_keys[0], fontweight='bold')
                            ax.set_xlabel("x index")
                            ax.set_ylabel("y index")
                            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                            cbar.set_label("Subaperture intensity (ADU)", rotation=270, labelpad=15)
                            st.pyplot(fig)
                        else:
                            fig, axes = plt.subplots(2, 2, figsize=(10, 9))
                            axes = axes.flatten()
                            
                            # --- CUSTOM 2x2 LAYOUT MAPPING ---
                            # Top-Left (0): SHWFS4, Top-Right (1): SHWFS1
                            # Bottom-Left (2): SHWFS3, Bottom-Right (3): SHWFS2
                            target_layout = {
                                "SHWFS4SubAper": 0,
                                "SHWFS1SubAper": 1,
                                "SHWFS3SubAper": 2,
                                "SHWFS2SubAper": 3
                            }
                            
                            # Turn off axes initially to hide empty quadrants if a sensor is missing
                            for ax in axes: 
                                ax.axis('off')
                                
                            for i, key in enumerate(active_keys):
                                ax_idx = target_layout.get(key, i)
                                ax = axes[ax_idx]
                                ax.axis('on') # Turn the active quadrant back on
                                im = ax.imshow(maps_with_data[i], norm=LogNorm(vmin=v_min_sub, vmax=v_max_sub), cmap=cmap)
                                ax.set_title(key, fontweight='bold')
                                ax.set_xlabel("x index")
                                ax.set_ylabel("y index")
                                
                            cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
                            cbar.set_label("Subaperture intensity (ADU)", rotation=270, labelpad=15)
                            st.pyplot(fig)
                            
                
                except FileNotFoundError:
                    st.error(f"Could not find the pupil mask text file at: `{map_path}`")
                except Exception as e:
                    st.error(f"Error loading the pupil mask: {e}")
                    
            # ==========================================
            # 2. TELEMETRY TIME-SERIES & DISTRIBUTIONS
            # ==========================================
            st.write("---")
            st.write("### Telemetry Analysis")
            
            # Extract the base sensor names (e.g., 'SHWFS1') from the valid_keys we found earlier
            valid_sensors = [k.replace('SubAper', '') for k in valid_keys]
            
            col1, col2 = st.columns(2)
            
            # --- LEFT COLUMN: ALL SENSORS OVERLAY ---
            with col1:
                st.markdown("**All Sensors (Overlay)**")
                all_graph_options = ['None', 'Average Subaperture Flux Over Time', 'Centroid Distribution (Histogram)']
                sel_all_graph = st.selectbox("Select Combined Graph:", all_graph_options, key="all_sensors_graph")
                
                if sel_all_graph != 'None' and valid_sensors:
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    
                    if sel_all_graph == 'Average Subaperture Flux Over Time':
                        # Find minimum length to ensure aligned X-axes if arrays differ in size
                        min_len = min([proc_data[f"{s}SubAper"].shape[0] for s in valid_sensors if f"{s}SubAper" in proc_data.files])
                        
                        for sensor in valid_sensors:
                            sub_key = f"{sensor}SubAper"
                            if sub_key in proc_data.files:
                                subap_data = proc_data[sub_key][-min_len:] # Sync to the end
                                mean_subap_flux = subap_data.mean(axis=1) if subap_data.ndim > 1 else subap_data
                                ax1.plot(mean_subap_flux, label=sensor, alpha=0.8, linewidth=1.2)
                                
                        ax1.set_xlabel('Synchronized Frame Number (counted from end)')
                        ax1.set_ylabel('Mean Flux (ADU)')
                        ax1.set_title('Average Subaperture Flux (All Sensors)', fontweight='bold')
                        ax1.grid(True, alpha=0.3)
                        ax1.legend()
                        st.pyplot(fig1)
                        
                    elif sel_all_graph == 'Centroid Distribution (Histogram)':
                        for sensor in valid_sensors:
                            cent_key = f"{sensor}Centroid"
                            if cent_key in proc_data.files:
                                centroid_data = proc_data[cent_key].flatten()
                                ax1.hist(centroid_data, bins=100, alpha=0.5, label=sensor)
                                
                        ax1.set_xlabel('Centroid Value')
                        ax1.set_ylabel('Frequency')
                        ax1.set_title('Centroid Distribution (All Sensors)', fontweight='bold')
                        ax1.grid(True, alpha=0.3)
                        ax1.legend()
                        st.pyplot(fig1)

            # --- RIGHT COLUMN: SINGLE SENSOR ISOLATION ---
            with col2:
                st.markdown("**Single Sensor**")
                if valid_sensors:
                    sel_single_sensor = st.selectbox("Select WFS Sensor:", valid_sensors, key="single_sensor_sel")
                    single_graph_options = ['None', 'Average Subaperture Flux Over Time', 'Centroid Distribution (Histogram)']
                    sel_single_graph = st.selectbox("Select Isolated Graph:", single_graph_options, key="single_sensor_graph")
                    
                    if sel_single_graph != 'None':
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        
                        if sel_single_graph == 'Average Subaperture Flux Over Time':
                            sub_key = f"{sel_single_sensor}SubAper"
                            if sub_key in proc_data.files:
                                subap_data = proc_data[sub_key]
                                mean_subap_flux = subap_data.mean(axis=1) if subap_data.ndim > 1 else subap_data
                                ax2.plot(mean_subap_flux, color='#ff7f0e', linewidth=1.5)
                                ax2.set_xlabel('Absolute Frame Number')
                                ax2.set_ylabel('Mean Flux (ADU)')
                                ax2.set_title(f'Average Subaperture Flux ({sel_single_sensor})', fontweight='bold')
                                ax2.grid(True, alpha=0.3)
                                st.pyplot(fig2)
                            else:
                                st.warning(f"No SubAper data for {sel_single_sensor}")
                                
                        elif sel_single_graph == 'Centroid Distribution (Histogram)':
                            cent_key = f"{sel_single_sensor}Centroid"
                            if cent_key in proc_data.files:
                                centroid_data = proc_data[cent_key].flatten()
                                ax2.hist(centroid_data, bins=100, color='#2ca02c', alpha=0.75)
                                ax2.set_xlabel('Centroid Value')
                                ax2.set_ylabel('Frequency')
                                ax2.set_title(f'Centroid Distribution ({sel_single_sensor})', fontweight='bold')
                                ax2.grid(True, alpha=0.3)
                                st.pyplot(fig2)
                            else:
                                st.warning(f"No Centroid data for {sel_single_sensor}")
                                
     
                            
        except Exception as e:
            st.error(f"Error processing SHWFS files: {e}")
        finally:
            if 'raw_data' in locals() and raw_data is not None: raw_data.close()
            if 'proc_data' in locals() and proc_data is not None: proc_data.close()