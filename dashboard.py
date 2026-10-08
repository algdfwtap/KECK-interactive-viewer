import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Page Configuration
st.set_page_config(page_title="Keck AO Telemetry Viewer", layout="wide")
st.title("Keck AO Telemetry Viewer")

# 2. Get the viewing nights from the directory
base_dir = Path("/g3/data/keck_telemetry/")

if base_dir.exists():
    nights = [folder.name for folder in base_dir.iterdir() if folder.is_dir()]
    nights.sort(reverse=True)
else:
    nights = []

if not nights:
    st.error(f"No data directories found in {base_dir}")
    st.stop()

# 3. Build the UI Controls side-by-side
col1, col2 = st.columns(2)

with col1:
    selected_night = st.selectbox("Select Viewing Night", nights)

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
    # We need BOTH files to run the calibration
    sg_files = list(selected_night_path.glob('*clmpstraingauge*.npz'))
    dtt_files = list(selected_night_path.glob('*downtiptilt*.npz'))
    
    if not sg_files:
        st.warning(f"No strain gauge file found in {selected_night_path}")
    elif not dtt_files:
        st.warning(f"No Down Tip-Tilt file found to cross-reference in {selected_night_path}")
    else:
        sg_file, dtt_file = sg_files[0], dtt_files[0]
        st.info(f"Loaded: `{sg_file.name}` and `{dtt_file.name}`") 
        
        try:
            sg_data = np.load(sg_file)
            dtt_data = np.load(dtt_file)
            
            st.write("### 📊 Select Graphs to Display")
            graph_choices = [
                'Strain Gauge Sequence (By Sample)',
                'Multi-Axis Calibration (Actual vs Expected)',
                'Percent Realized vs Expected Strain'
            ]
            selected_subgraphs = st.multiselect("Check the graphs you want to view:", graph_choices, default=graph_choices)
            
            if not selected_subgraphs:
                st.warning("Please select at least one graph.")
            else:
                # 1. SEQUENCE GRAPH (Plotly Interactive)
                if 'Strain Gauge Sequence (By Sample)' in selected_subgraphs:
                    dtt_strain = sg_data['DTTStrainGauge']
                    x_axis = np.arange(dtt_strain.shape[0])
                    
                    st.write("*(💡 **Tip:** Use the slider below the graph to scrub and zoom!)*")
                    fig = go.Figure()
                    for i in range(dtt_strain.shape[1]):
                        fig.add_trace(go.Scatter(x=x_axis, y=dtt_strain[:, i], mode='lines', name=f'Channel {i}', line=dict(width=1), opacity=0.7))
                    
                    fig.update_layout(
                        title=dict(text="<b>Down-TT Strain Gauge Sequence</b>", x=0.5, font=dict(size=16)),
                        xaxis_title="Sample Number", yaxis_title="Strain", height=400,
                        margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig.update_xaxes(rangeslider_visible=True)
                    st.plotly_chart(fig, use_container_width=True)

                # 2. CALIBRATION GRAPHS (Matplotlib)
                if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs or 'Percent Realized vs Expected Strain' in selected_subgraphs:
                    st.write("---")
                    st.write("### 🧮 Calibration Analysis")
                    
                    # Extract and interpolate
                    sg_time = sg_data['rawtimestamp']
                    cmd_time = dtt_data['rawtimestamp']
                    dtt_strain = sg_data['DTTStrainGauge']
                    dtt_cmd = dtt_data['DTTMirrorCommand']
                    
                    dtt_cmd_aligned = np.zeros((len(sg_time), dtt_cmd.shape[1]))
                    for ch in range(dtt_cmd.shape[1]):
                        dtt_cmd_aligned[:, ch] = np.interp(sg_time, cmd_time, dtt_cmd[:, ch])
                        
                    dtt_micro = (dtt_strain - dtt_strain[0, :]) * 1e6
                    cmd_delta_2d = dtt_cmd_aligned - dtt_cmd_aligned[0, :]
                    
                    # Compute coefficients for UI display
                    coeff_text = ""
                    expected_strains = [] # Cache expected strain to avoid re-calculating
                    
                    for i in range(dtt_strain.shape[1]):
                        actual_strain = dtt_micro[:, i]
                        X = np.c_[cmd_delta_2d, np.ones(cmd_delta_2d.shape[0])]
                        coeffs, _, _, _ = np.linalg.lstsq(X, actual_strain, rcond=None)
                        expected_strains.append(X @ coeffs)
                        
                        coeff_text += f"**Channel {i}:** Tip Sensitivity: `{coeffs[0]:.4f}`, Tilt Sensitivity: `{coeffs[1]:.4f}`, Intercept: `{coeffs[2]:.4f}`\n\n"
                    
                    with st.expander("View Regression Coefficients (με / arcsec)"):
                        st.markdown(coeff_text)
                    
                    # Layout for the plots
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
    # Mirroring the exact same logic for Up Tip-Tilt
    sg_files = list(selected_night_path.glob('*clmpstraingauge*.npz'))
    utt_files = list(selected_night_path.glob('*uptiptilt*.npz'))
    
    if not sg_files:
        st.warning(f"No strain gauge file found in {selected_night_path}")
    elif not utt_files:
        st.warning(f"No Up Tip-Tilt file found to cross-reference in {selected_night_path}")
    else:
        sg_file, utt_file = sg_files[0], utt_files[0]
        st.info(f"Loaded: `{sg_file.name}` and `{utt_file.name}`") 
        
        try:
            sg_data = np.load(sg_file)
            utt_data = np.load(utt_file)
            
            st.write("### 📊 Select Graphs to Display")
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
                    
                    st.write("*(💡 **Tip:** Use the slider below the graph to scrub and zoom!)*")
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
                    st.write("### 🧮 Calibration Analysis")
                    
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
    # Search for the up tip-tilt file
    utt_files = list(selected_night_path.glob('*uptiptilt*.npz'))
    
    if not utt_files:
        st.warning(f"No up tip-tilt file found in {selected_night_path}. (Check the file naming convention if this is unexpected.)")
    else:
        file_path = utt_files[0]
        st.info(f"Loaded file: `{file_path.name}`")
        
        try:
            utt_data = np.load(file_path)
            
            # Extract data using the Up Tip Tilt specific array names
            tt_slopes = utt_data['SHWFS1TT']  
            mirror_commands = utt_data['UTTMainMirrorCommand']  
            
            tip = tt_slopes[:, 0]
            tilt = tt_slopes[:, 1]
            mirror_tip = mirror_commands[:, 0]
            mirror_tilt = mirror_commands[:, 1]
            total_samples = len(tip)
            x_axis = np.arange(total_samples)
            
            # --- DYNAMIC GRAPH SELECTOR ---
            st.write("### 📊 Select Graphs to Display")
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
                st.write("*(💡 **Tip:** Sequence graphs are interactive—click and drag to zoom/pan. Histograms are static.)*")
                
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
    dtt_files = list(selected_night_path.glob('*downtiptilt*.npz'))
    
    if not dtt_files:
        st.warning(f"No down tip-tilt file found in {selected_night_path}")
    else:
        file_path = dtt_files[0]
        st.info(f"Loaded file: `{file_path.name}`")
        
        try:
            dtt_data = np.load(file_path)
            tt_slopes = dtt_data['TTslope']  
            mirror_commands = dtt_data['DTTMirrorCommand']  
            
            tip = tt_slopes[:, 0]
            tilt = tt_slopes[:, 1]
            mirror_tip = mirror_commands[:, 0]
            mirror_tilt = mirror_commands[:, 1]
            total_samples = len(tip)
            x_axis = np.arange(total_samples)
            
            # --- DYNAMIC GRAPH SELECTOR ---
            st.write("### 📊 Select Graphs to Display")
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
                st.write("*(💡 **Tip:** Sequence graphs are interactive—click and drag to zoom/pan. Histograms are static.)*")
                
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

            
            
            
            
            
            
elif selected_graph == 'Shack Hartmann WFS Data':
    st.write("WFS plotting logic goes here.")
