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


# ==========================================
# SPATIAL PSD HELPERS
# ==========================================
def build_actuator_mask():
    """21x21 boolean map of the 349 active Keck xinetics DM actuators (row-major)."""
    widths = [7, 11, 13, 15, 17, 19, 19, 21, 21, 21, 21, 21, 21, 21, 19, 19, 17, 15, 13, 11, 7]
    m = np.zeros((21, 21), dtype=bool)
    for i, w in enumerate(widths):
        start = (21 - w) // 2
        m[i, start:start + w] = True
    assert m.sum() == 349
    return m

ACT_MASK = build_actuator_mask()


@st.cache_resource(show_spinner="Loading xinetics telemetry...")
def load_xinetics(path_str):
    """Load residual / DM command (n x 349, volts) and timestamps. Cached; do not mutate."""
    with np.load(path_str) as d:
        res = d['residualWaveFront'].astype(np.float64)
        cmd = d['DMCommand'].astype(np.float64) if 'DMCommand' in d.files else None
        if 'rawtimestamp' in d.files:
            raw = d['rawtimestamp']
            t = (raw - raw[0]).astype(np.float64) * 1e-9      # subtract in native dtype first (precision)
        else:
            t = None
    return res, cmd, t


def vector_to_grid(v, mask=ACT_MASK):
    """Place a 349-vector on the 21x21 grid, NaN outside the pupil."""
    g = np.full(mask.shape, np.nan)
    g[mask] = v
    return g


def get_sample_rate(data):
    """
    Sampling rate (Hz) for a loaded .npz. Returns (fs, source).

    1. `rawtimestamp`: integer nanoseconds since epoch (documented in the telemetry table).
    2. `timestamp`: units are not documented, so they are inferred. A numeric array is tried as ns, us, ms and s,
       and the unit that gives a plausible loop rate (100 Hz to 20 kHz) is used; string or datetime arrays are
       parsed as datetimes.
    3. Otherwise 1000 Hz is assumed.

    Differences are taken in the stored integer dtype BEFORE converting to float, because epoch-nanosecond values
    (~1.7e18) lose ~256 ns of precision as float64. The median interval is used so dropped frames or timing
    jitter do not bias the rate.
    """
    def rate_from_ns_or_unit(dt, scale):
        interval = float(np.median(dt)) * scale
        return 1.0 / interval if interval > 0 else None

    if 'rawtimestamp' in data.files:
        ts = data['rawtimestamp']
        if ts.dtype.kind in 'iuf' and ts.size > 2:
            dt = np.diff(ts)
            dt = dt[dt > 0]
            if len(dt):
                fs = rate_from_ns_or_unit(dt, 1e-9)
                if fs is not None and 1.0 <= fs <= 1e5:
                    return fs, 'rawtimestamp'

    if 'timestamp' in data.files:
        ts = data['timestamp']
        if ts.size > 2:
            dt = None
            if ts.dtype.kind in 'iuf':
                dt = np.diff(ts)
                dt = dt[dt > 0]
                if len(dt):
                    for unit, scale in (('ns', 1e-9), ('us', 1e-6), ('ms', 1e-3), ('s', 1.0)):
                        fs = rate_from_ns_or_unit(dt, scale)
                        if fs is not None and 100.0 <= fs <= 2e4:
                            return fs, f'timestamp (inferred {unit})'
            else:
                try:
                    dts = np.diff(np.asarray(ts).astype('datetime64[ns]').astype(np.int64))
                    dts = dts[dts > 0]
                    if len(dts):
                        fs = rate_from_ns_or_unit(dts, 1e-9)
                        if fs is not None and 1.0 <= fs <= 1e5:
                            return fs, 'timestamp (parsed datetime)'
                except Exception:
                    pass
    return 1000.0, 'assumed 1000 Hz (no usable timestamp)'


def estimate_fs(t):
    """Sample rate from timestamps (seconds); falls back to 1 kHz."""
    if t is not None and len(t) > 2:
        dt = np.diff(t)
        dt = dt[dt > 0]
        if len(dt):
            return 1.0 / np.median(dt)
    return 1000.0


@st.cache_data(show_spinner="Computing temporal PSDs...")
def compute_temporal_psd(path_str, lag, nperseg, exclude_edge, sign=1):
    """
    Welch PSDs of residual and POL, averaged over actuators, plus two rejection estimates:
    ratio = sqrt(P_res / P_pol) and cross-spectral |P_res,pol / P_pol,pol|.
    """
    from scipy.signal import welch, csd
    from scipy.ndimage import binary_erosion
    res, cmd, t = load_xinetics(path_str)
    fs = estimate_fs(t)
    good = binary_erosion(ACT_MASK) if exclude_edge else ACT_MASK
    gv = good[ACT_MASK]
    r, pol, _ = make_pol(res, cmd, lag, sign)
    r, pol = r[:, gv], pol[:, gv]
    nperseg = int(min(nperseg, r.shape[0]))
    f, Pr = welch(r, fs=fs, nperseg=nperseg, axis=0)
    _, Pp = welch(pol, fs=fs, nperseg=nperseg, axis=0)
    _, Prp = csd(pol, r, fs=fs, nperseg=nperseg, axis=0)
    Pr, Pp, Prp = Pr.mean(axis=1), Pp.mean(axis=1), Prp.mean(axis=1)
    return f, Pr, Pp, np.sqrt(Pr / Pp), np.abs(Prp / Pp), fs


@st.cache_data(show_spinner="Checking controller timing...")
def controller_timing(path_str, leak=0.99, max_m=8, exclude_edge=True):
    """
    Correlation between the command update u[k] = cmd[k] - leak*cmd[k-1] and the residual m frames
    earlier. The peak m is how the telemetry indexes the controller (0 = cmd[k] computed from res[k]);
    the sign of the peak is the loop's sign convention. Also least-squares fits
    cmd[k] = a*cmd[k-1] + b*res[k-m] at the peak, so a ~ leak and b ~ -/+ gain.
    This describes how the stored arrays line up, NOT the physical WFS-to-mirror delay.
    Returns ms, corr, a, b.
    """
    from scipy.ndimage import binary_erosion
    res, cmd, _ = load_xinetics(path_str)
    good = binary_erosion(ACT_MASK) if exclude_edge else ACT_MASK
    gv = good[ACT_MASK]
    u = (cmd[1:] - leak * cmd[:-1])[:, gv]               # u[j] belongs to frame j+1
    r = res[1:][:, gv]
    ms = np.arange(0, max_m + 1)
    corr = []
    for m in ms:
        a_ = r[:len(r) - m]; b_ = u[m:]
        a_ = a_ - a_.mean(axis=0); b_ = b_ - b_.mean(axis=0)
        corr.append((a_ * b_).sum() / np.sqrt((a_ * a_).sum() * (b_ * b_).sum()))
    corr = np.array(corr)
    m0 = int(ms[np.argmax(np.abs(corr))])
    n = cmd.shape[0]
    ks = np.arange(max(1, m0), n)
    y = cmd[ks][:, gv].ravel()
    X = np.column_stack([cmd[ks - 1][:, gv].ravel(), res[ks - m0][:, gv].ravel()])
    (a_fit, b_fit), *_ = np.linalg.lstsq(X, y, rcond=None)
    return ms, corr, float(a_fit), float(b_fit)


def detect_pol_sign(path_str):
    """-1 if the command update is negatively correlated with the residual (POL = res - cmd), else +1."""
    ms, corr, _, _ = controller_timing(path_str, 0.99, 8, True)
    return -1 if corr[int(np.argmax(np.abs(corr)))] < 0 else 1


def leaky_integrator_rtf(f, fs, gain, leak, delay):
    """Rejection transfer function 1/(1+G), G = g z^-d / (1 - k z^-1), delay d in frames (may be fractional)."""
    w = 2 * np.pi * f / fs
    z1 = np.exp(-1j * w)
    G = gain * np.exp(-1j * w * delay) / (1 - leak * z1)
    return np.abs(1 / (1 + G))


def first_valid_index(lag):
    """First frame index k for which a (possibly fractional) lag has data behind it."""
    return int(np.floor(lag)) + (1 if lag - np.floor(lag) > 1e-9 else 0)


def delayed_cmd(cmd, ks, lag):
    """
    DM command as it was on the mirror `lag` frames before frame ks (lag may be fractional).
    Linear blend of the two neighboring stored commands, e.g. lag=1.7 ->
    0.3*cmd[k-1] + 0.7*cmd[k-2]. (The DM holds each command for a frame, so a blend is a
    reasonable model of a non-integer delay; it slightly low-passes the command at high frequency.)
    """
    L0 = int(np.floor(lag))
    f = lag - L0
    out = (1 - f) * cmd[ks - L0]
    if f > 1e-9:
        out = out + f * cmd[ks - L0 - 1]
    return out


def make_pol(res, cmd, lag, sign=1):
    """Pseudo-open-loop = residual[k] + sign * (command on the mirror at frame k).
    sign = -1 when the stored command update is negatively correlated with the residual
    (cmd[k] = leak*cmd[k-1] - g*res[k]); see detect_pol_sign. Returns (res_trim, pol, start)."""
    start = first_valid_index(lag)
    ks = np.arange(start, res.shape[0])
    return res[ks], res[ks] + sign * delayed_cmd(cmd, ks, lag), start


@st.cache_data(show_spinner="Computing spatial PSD...")
def compute_spatial_psd(path_str, lag, n_use, exclude_edge, sign=1, pad=64):
    """
    Average of per-frame 2D power spectra (NOT the power of the average frame),
    radially averaged in 1 cycle/pupil bins up to Nyquist.
    Returns freqs (cycles/pupil), {name: profile}, effective pupil width N.
    """
    from scipy.ndimage import binary_erosion
    res, cmd, _ = load_xinetics(path_str)
    good = binary_erosion(ACT_MASK) if exclude_edge else ACT_MASK
    gv = good[ACT_MASK]                      # which of the 349 are used
    gi = np.nonzero(good)                    # same row-major order as v[:, gv]
    n_good = int(good.sum())
    N = int(good.any(axis=0).sum())          # effective pupil width in actuators

    ks = np.arange(first_valid_index(lag), res.shape[0])
    if len(ks) > n_use:
        ks = ks[np.linspace(0, len(ks) - 1, n_use).astype(int)]

    names = ['Residual'] + (['POL'] if cmd is not None else [])
    profiles = {}

    y, x = np.indices((pad, pad))
    r = np.hypot(x - pad // 2, y - pad // 2) * N / pad     # cycles/pupil
    idx = np.rint(r).astype(int)
    nbins = N // 2                                          # up to Nyquist
    freqs = np.arange(1, nbins + 1)

    for name in names:
        acc = np.zeros((pad, pad))
        for chunk in np.array_split(ks, max(1, int(np.ceil(len(ks) / 200)))):
            v = res[chunk] if name == 'Residual' else res[chunk] + sign * delayed_cmd(cmd, chunk, lag)
            v = v[:, gv]
            v = v - v.mean(axis=1, keepdims=True)            # remove piston
            g = np.zeros((len(chunk), pad, pad))
            g[:, gi[0], gi[1]] = v
            F = np.fft.fftshift(np.fft.fft2(g), axes=(-2, -1))
            acc += (np.abs(F) ** 2).sum(axis=0)
        psd2d = acc / len(ks) / n_good
        profiles[name] = np.array([psd2d[idx == k].mean() for k in freqs])
    return freqs, profiles, N


# Keck telemetry documentation (K1 xinetics DM, volts): 10 V = 4 um of wavefront (OPD) = 2 um of mirror stroke,
# i.e. 0.4 um (400 nm) of wavefront per volt. (van Kooten et al. 2022 use 0.6 um/V for their PyWFS configuration.)
XIN_NM_PER_V = 400.0


def good_actuator_vector(exclude_edge):
    """Boolean over the 349-vector: True for the actuators used (inner ones if exclude_edge)."""
    from scipy.ndimage import binary_erosion
    good = binary_erosion(ACT_MASK) if exclude_edge else ACT_MASK
    return good[ACT_MASK]


@st.cache_data(show_spinner="Computing RMS...")
def compute_rms_summary(path_str, lag, exclude_edge, sign):
    """Overall spatial RMS (piston removed per frame, over time and actuators), in volts, for residual and POL."""
    res, cmd, _ = load_xinetics(path_str)
    gv = good_actuator_vector(exclude_edge)
    r, pol, _ = make_pol(res, cmd, lag, sign)

    def rms(a):
        a = a[:, gv]
        a = a - a.mean(axis=1, keepdims=True)
        return float(np.sqrt(np.mean(a ** 2)))
    return rms(r), rms(pol)


def unit_controls(key_prefix):
    """Units selector. Returns (scale, unit): multiply volts by `scale` to get `unit`."""
    u1, u2 = st.columns(2)
    with u1:
        choice = st.radio("Units", ["Volts", "nm of wavefront (OPD)"], horizontal=True, key=f'{key_prefix}_units')
    if choice.startswith('nm'):
        with u2:
            nm_per_v = st.number_input(
                "nm of wavefront per volt", 1.0, 2000.0, XIN_NM_PER_V, 10.0, key=f'{key_prefix}_nmperv',
                help="Keck telemetry documentation: 10 V = 4 um of wavefront, so 400 nm/V. The van Kooten et al. "
                     "2022 paper uses 600 nm/V for its PyWFS configuration; confirm which applies to your data.")
        return float(nm_per_v), 'nm'
    return 1.0, 'V'


# ==========================================
# TURBULENCE PARAMETERS (tau0, r0)
# ==========================================
KOLMOGOROV_SF = 6.88          # D_phi(r) = 6.88 (r/r0)^(5/3)  [rad^2]


def fit_five_thirds(x, y, with_offset=True):
    """Least-squares fit y = c + A * x**(5/3). Returns (A, c). With with_offset=False, c = 0."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cols = [x ** (5.0 / 3.0)]
    if with_offset:
        cols.append(np.ones_like(x))
    coef = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]
    return float(coef[0]), (float(coef[1]) if with_offset else 0.0)


def volts_to_rad2(nm_per_v, wl_nm):
    """Multiply a variance in V^2 by this to get rad^2 of phase at wavelength wl_nm."""
    return (2.0 * np.pi * nm_per_v / wl_nm) ** 2


def temporal_sf_from_psd(f, P, taus):
    """D(tau) = 2 * integral P(f) (1 - cos(2 pi f tau)) df for a one-sided PSD P(f)."""
    df = f[1] - f[0]
    return np.array([2.0 * np.sum(P * (1.0 - np.cos(2.0 * np.pi * f * t))) * df for t in taus])


def temporal_sf_direct(a, max_lag, stride=1):
    """Time-domain structure function <(a[t+k]-a[t])^2>, k = 1..max_lag frames (mean over time and columns)."""
    a = a[::stride] if stride > 1 else a
    return np.array([np.mean((a[k:] - a[:-k]) ** 2) for k in range(1, max_lag + 1)])


def tau0_estimates(f, P, a, fs, nm_per_v, wl_nm, fit_ms, f_noise_hz, max_ms=40.0):
    """
    Coherence time tau0 (s) from a POL time series `a` (n x m volts, piston removed per frame) and its
    mean PSD `P` (V^2/Hz). Defined by D_phi(tau0) = 1 rad^2 at wl_nm with D_phi = A tau^(5/3).
      PSD route : subtract a flat noise floor (median of P above f_noise_hz), integrate to D(tau), fit A tau^(5/3).
      Direct    : time-domain structure function, fit c + A tau^(5/3) (c absorbs the white-noise offset 2*sigma_n^2).
    """
    k = volts_to_rad2(nm_per_v, wl_nm)
    n_max = max(3, int(max_ms * 1e-3 * fs))
    lags = np.arange(1, n_max + 1)
    taus = lags / fs
    floor = float(np.median(P[f >= f_noise_hz])) if np.any(f >= f_noise_hz) else 0.0
    D_psd = temporal_sf_from_psd(f, np.clip(P - floor, 0, None), taus) * k
    D_dir = temporal_sf_direct(a, n_max, stride=max(1, a.shape[0] // 20000)) * k
    sel = (taus >= fit_ms[0] * 1e-3) & (taus <= fit_ms[1] * 1e-3)
    out = dict(taus=taus, D_psd=D_psd, D_dir=D_dir, floor=floor, sel=sel)
    if sel.sum() < 3:
        out.update(tau0_psd=np.nan, tau0_dir=np.nan, A_psd=np.nan, A_dir=np.nan, c_dir=np.nan)
        return out
    A_p, _ = fit_five_thirds(taus[sel], D_psd[sel], with_offset=False)
    A_d, c_d = fit_five_thirds(taus[sel], D_dir[sel], with_offset=True)
    out.update(A_psd=A_p, A_dir=A_d, c_dir=c_d,
               tau0_psd=(1.0 / A_p) ** 0.6 if A_p > 0 else np.nan,
               tau0_dir=(1.0 / A_d) ** 0.6 if A_d > 0 else np.nan)
    return out


def spatial_sf(a, mask21, max_sep=5):
    """
    Spatial structure function of a (n x 349) time series on the 21x21 actuator grid, in V^2.
    The time mean of every actuator is removed first (static pattern), then <(a_j - a_i)^2> is accumulated
    over all actuator pairs whose both ends are in `mask21`, grouped by separation (in actuator pitches).
    Insensitive to piston by construction. Returns (separations in pitches, D in V^2, pair counts).
    """
    n = a.shape[0]
    cube = np.full((n, 21, 21), np.nan)
    cube[:, ACT_MASK] = a - a.mean(axis=0, keepdims=True)
    cube[:, ~mask21] = np.nan
    acc = {}
    for dy in range(0, max_sep + 1):
        for dx in range(-max_sep, max_sep + 1):
            if (dy == 0 and dx <= 0) or dx * dx + dy * dy > max_sep * max_sep:
                continue
            c0 = slice(max(0, -dx), 21 - max(0, dx))
            c1 = slice(max(0, dx), 21 - max(0, -dx))
            d = (cube[:, dy:, c1] - cube[:, :21 - dy, c0]) ** 2
            ok = np.isfinite(d)
            key = dx * dx + dy * dy
            s, c = acc.get(key, (0.0, 0))
            acc[key] = (s + float(np.sum(d[ok])), c + int(ok.sum()))
    keys = sorted(acc)
    r = np.sqrt(np.array(keys, float))
    D = np.array([acc[k][0] / max(acc[k][1], 1) for k in keys])
    N = np.array([acc[k][1] for k in keys])
    return r, D, N


def r0_from_sf(r_pitch, D_V2, pitch_m, nm_per_v, wl_nm, fit_pitches, with_offset=True):
    """r0 (m, at wl_nm) from D(r) = c + 6.88 (r/r0)^(5/3), r = separation in metres, D in rad^2."""
    k = volts_to_rad2(nm_per_v, wl_nm)
    rm = r_pitch * pitch_m
    sel = (r_pitch >= fit_pitches[0]) & (r_pitch <= fit_pitches[1])
    D = D_V2 * k
    if sel.sum() < (3 if with_offset else 2):
        return np.nan, np.nan, D, sel
    A, c = fit_five_thirds(rm[sel], D[sel], with_offset)
    r0 = (KOLMOGOROV_SF / A) ** 0.6 if A > 0 else np.nan
    return r0, c, D, sel


@st.cache_data(show_spinner="Estimating tau0 and r0 from the POL...")
def compute_turbulence(path_str, lag, exclude_edge, sign, nperseg, nm_per_v, wl_nm, fit_ms, f_noise,
                       pitch_m, fit_pitches, with_offset, max_sep=6):
    """tau0 from the temporal behaviour and r0 from the spatial structure function of the POL (pseudo-open-loop)."""
    from scipy.signal import welch
    from scipy.ndimage import binary_erosion
    res, cmd, t = load_xinetics(path_str)
    fs = estimate_fs(t)
    good = binary_erosion(ACT_MASK) if exclude_edge else ACT_MASK
    gv = good[ACT_MASK]
    _, pol, _ = make_pol(res, cmd, lag, sign)
    pol = pol - pol[:, gv].mean(axis=1, keepdims=True)            # piston is not measured: remove per frame
    pg = pol[:, gv]
    f, P = welch(pg, fs=fs, nperseg=int(min(nperseg, pg.shape[0])), axis=0)
    P = P.mean(axis=1)
    out = tau0_estimates(f, P, pg, fs, nm_per_v, wl_nm, fit_ms, f_noise, max_ms=max(40.0, 1.2 * fit_ms[1]))
    idx = np.linspace(0, pol.shape[0] - 1, min(pol.shape[0], 3000)).astype(int)
    r_p, D_V2, _ = spatial_sf(pol[idx], good, max_sep)
    r0, c_sp, D_rad, sel_sp = r0_from_sf(r_p, D_V2, pitch_m, nm_per_v, wl_nm, fit_pitches, with_offset)
    out.update(fs=fs, f=f, P=P, r_pitch=r_p, D_sp=D_rad, sel_sp=sel_sp, r0=r0, c_sp=c_sp)
    return out


def render_turbulence(xin_path, xin_name):
    """Derived turbulence parameters (tau0, r0, theta0) from the pseudo-open-loop wavefront."""
    res, cmd, t = load_xinetics(xin_path)
    if res.ndim != 2 or res.shape[1] != 349:
        st.error(f"`residualWaveFront` has shape {res.shape}, expected (n, 349).")
        return
    if cmd is None or cmd.shape != res.shape:
        st.error("`DMCommand` is missing or has the wrong shape, so POL cannot be built.")
        return
    fs = estimate_fs(t)
    st.metric("Sample rate", f"{fs:.1f} Hz")

    st.write("**POL settings**")
    s1, s2, s3 = st.columns(3)
    with s1:
        lag = float(st.number_input("POL lag (frames)", 0.0, 5.0, 1.7, 0.1, format="%.2f", key='tb_lag',
                                    help="POL[k] = residual[k] - the command on the mirror `lag` frames earlier."))
    with s2:
        exclude_edge = st.checkbox("Exclude outer actuator ring", value=True, key='tb_edge')
    with s3:
        nperseg = st.select_slider("Welch segment length", [256, 512, 1024, 2048, 4096], value=1024, key='tb_nperseg',
                                   help="Number of samples in each chunk Welch's method transforms before averaging the spectra. Frequency resolution = sample rate / segment length, and the lowest frequency you can resolve is about the same number. Longer segments resolve narrow peaks and slow behavior but leave fewer segments to average, so the curve is noisier. Shorter segments are smoother but blur peaks and cut off low frequencies. Also used by the rejection curve and by tau0 (PSD route).")
        st.caption(f"Resolution ~ {fs / nperseg:.2f} Hz; about {max(1, 2 * res.shape[0] // int(nperseg) - 1)} "
                   f"averaged segments (50% overlap).")
    sign = -1   # POL = residual - delayed command (see Performance Plots)

    st.markdown("#### Coherence time tau0 and Fried parameter r0")
    st.write("Both come from the **POL** (the open-loop turbulence estimate), never the closed-loop residual, "
             "which the loop has already suppressed. tau0 is read from how fast the POL wavefront decorrelates "
             "in time; r0 from how fast it decorrelates across the 21x21 actuator grid. Neither needs the "
             "other, and together they give an effective wind speed v = 0.314 r0 / tau0.")
    st.warning("Treat these as **trend / order-of-magnitude** numbers, not calibrated measurements. On simulated "
               "Kolmogorov frozen-flow data with known r0 and wind speed I found r0 came out about 1.1-1.4x too "
               "high and tau0 about 1.5-2x too high. Two reasons are physical: the WFS averages over a "
               "subaperture (~0.56 m), which smooths the wavefront, and the DM never sees low-order or outer-scale "
               "turbulence perfectly. Real data adds the lag/sign choice, vibration, noise and the nm/V factor.")
    u1, u2, u3, u4 = st.columns(4)
    with u1:
        wl_t = st.number_input("Wavelength (nm)", 300.0, 3000.0, 500.0, 50.0, key='pp_turb_wl')
    with u2:
        nmv_t = st.number_input("nm of wavefront per volt", 1.0, 2000.0, XIN_NM_PER_V, 10.0, key='pp_turb_nmv',
                                help="Squares into every number here: a 470 vs 400 nm/V choice changes r0 by "
                                     "(470/400)^(6/5) = 1.2x.")
    with u3:
        pitch_t = st.number_input("Actuator pitch on primary (m)", 0.2, 1.0, 0.5625, 0.005, format="%.4f",
                                  key='pp_turb_pitch',
                                  help="Please confirm for your configuration. Sets the length scale of the r0 fit.")
    with u4:
        airmass_t = st.number_input("Airmass", 1.0, 3.0, 1.0, 0.05, key='pp_turb_x',
                                    help="Only used to refer r0 to zenith: r0_zenith = r0_obs * airmass^(3/5).")
    v1, v2, v3 = st.columns(3)
    with v1:
        fit_ms = st.slider("tau fit window (ms)", 5, 400, (30, 150), 5, key='pp_turb_fit',
                           help="Use lags where the structure function follows tau^(5/3). Below ~ pitch/wind "
                                "(0.56 m / 10 m/s = 56 ms) the smoothed wavefront gives ~tau^2 instead, so a "
                                "window that is too short biases tau0 high. Check the slope readout.")
    with v2:
        fit_sep = st.slider("Separation fit window (pitches)", 1.0, 6.0, (1.0, 4.0), 0.5, key='pp_turb_sep')
    with v3:
        f_noise_t = st.number_input("Noise floor from f >= (Hz)", 10.0, float(fs / 2 - 1), float(0.8 * fs / 2), 10.0,
                                    key='pp_turb_fn',
                                    help="The median PSD above this frequency is taken as white WFS noise and "
                                         "removed before integrating the temporal structure function.")
    with_off = st.checkbox("Fit a constant offset in the spatial structure function (absorbs white noise)",
                           value=True, key='pp_turb_off')
    h_bar = st.number_input("Effective turbulence height h-bar (m), ASSUMED", 500.0, 20000.0, 5000.0, 500.0,
                            key='pp_turb_h',
                            help="theta0 needs the turbulence height profile, which one DM's telemetry cannot give. "
                                 "This is a placeholder: replace it with a value from a Cn2 profile (e.g. site "
                                 "monitor MASS/DIMM data) for your night.")

    if st.checkbox("Compute tau0 and r0 (a few seconds)", value=False, key='pp_turb_run'):
        tb = compute_turbulence(xin_path, lag, exclude_edge, sign, int(nperseg), float(nmv_t), float(wl_t),
                                (float(fit_ms[0]), float(fit_ms[1])), float(f_noise_t), float(pitch_t),
                                (float(fit_sep[0]), float(fit_sep[1])), bool(with_off))
        wl_f = float(wl_t)
        taus_ms = tb['taus'] * 1e3
        to_500 = (500.0 / wl_f) ** (6.0 / 5.0)
        r0w = tb['r0']
        tau_use = tb['tau0_psd']
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"tau0 at {wl_f:.0f} nm, PSD route", "n/a" if not np.isfinite(tb['tau0_psd']) else f"{tb['tau0_psd'] * 1e3:.2f} ms")
        m2.metric("tau0, direct route", "n/a" if not np.isfinite(tb['tau0_dir']) else f"{tb['tau0_dir'] * 1e3:.2f} ms",
                  help="Time-domain structure function with a free noise offset. Should agree with the PSD route.")
        m3.metric(f"r0 at {wl_f:.0f} nm", "n/a" if not np.isfinite(r0w) else f"{r0w * 100:.1f} cm")
        if np.isfinite(r0w):
            r0_500 = r0w * to_500
            r0_zen = r0_500 * float(airmass_t) ** 0.6
            m4.metric("r0 at 500 nm, zenith", f"{r0_zen * 100:.1f} cm",
                      help="Scaled with lambda^(6/5) and airmass^(3/5).")
            seeing = 0.98 * 500e-9 / r0_500 * 206265.0
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Seeing FWHM, 500 nm", f"{seeing:.2f} arcsec", help="0.98 lambda / r0 at 500 nm, observed airmass.")
            if np.isfinite(tau_use) and tau_use > 0:
                d2.metric("Effective wind speed", f"{0.314 * r0w / tau_use:.1f} m/s",
                          help="v = 0.314 r0 / tau0 with both at the same wavelength. Inherits both biases.")
                d3.metric("Greenwood frequency", f"{0.134 / tau_use:.1f} Hz", help="f_G = 0.134 / tau0.")
                d4.metric(f"tau0 at 500 nm", f"{tau_use * to_500 * 1e3:.2f} ms",
                          help="tau0 scales as lambda^(6/5).")
            e1, _e2, _e3, _e4 = st.columns(4)
            e1.metric(f"theta0 at {wl_f:.0f} nm (assumed h-bar)", f"{0.314 * r0w / h_bar * 206265.0:.2f} arcsec",
                      help="theta0 = 0.314 r0 / h-bar for a single effective layer. It is only as good as r0 "
                           "and the assumed h-bar; it is not measured from this telemetry.")

        # ---- temporal structure function ----
        sel = tb['sel']
        fig_tau = go.Figure()
        fig_tau.add_trace(go.Scatter(x=taus_ms, y=tb['D_psd'], name='From PSD (noise floor removed)', mode='lines'))
        fig_tau.add_trace(go.Scatter(x=taus_ms, y=tb['D_dir'], name='Direct (time domain)', mode='lines'))
        if np.isfinite(tb['tau0_psd']):
            fig_tau.add_trace(go.Scatter(x=taus_ms, y=tb['A_psd'] * tb['taus'] ** (5 / 3),
                                         name='A tau^(5/3) fit (PSD)', mode='lines', line=dict(dash='dash')))
        fig_tau.add_hline(y=1, line=dict(color='gray', width=1), annotation_text='1 rad^2')
        fig_tau.add_vrect(x0=fit_ms[0], x1=fit_ms[1], fillcolor='gray', opacity=0.12, line_width=0)
        fig_tau.update_xaxes(type='log', title_text='Lag tau (ms)')
        fig_tau.update_yaxes(type='log', title_text=f'Phase structure function D(tau) (rad^2 at {wl_f:.0f} nm)')
        fig_tau.update_layout(height=380, title='Temporal structure function of the POL', margin=dict(t=40))
        st.plotly_chart(fig_tau, use_container_width=True)
        st.caption("**How to read:** the curve shows how much the POL wavefront changes (in rad² of phase) over a "
                   "time lag. For frozen-flow Kolmogorov turbulence it rises as a straight line of slope 5/3 on these "
                   "log-log axes. The dashed line is that fit over the gray window, and tau0 is the lag where it "
                   "reaches 1 rad² (gray horizontal line). The blue (PSD route) and orange (direct) curves should "
                   "agree. Flattening at short lags is the noise floor or pupil-scale smoothing; flattening at long "
                   "lags is the finite pupil and outer scale. A lower curve means longer coherence time.")
        if sel.sum() >= 3 and np.all(tb['D_psd'][sel] > 0):
            slope = np.polyfit(np.log(tb['taus'][sel]), np.log(tb['D_psd'][sel]), 1)[0]
            msg = (f"Log-log slope inside the fit window: {slope:.2f} (frozen-flow Kolmogorov: 1.67; "
                   f"about 2 means the window is inside the pitch-scale smoothing, so move it to longer lags).")
            (st.caption if 1.4 <= slope <= 1.95 else st.warning)(msg)
        st.caption(f"Noise floor removed from the PSD: {tb['floor']:.3g} V^2/Hz. Direct-route noise offset "
                   f"c = {tb['c_dir']:.3g} rad^2 (2 sigma_n^2).")

        # ---- spatial structure function ----
        rp = tb['r_pitch']
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=rp * pitch_t, y=tb['D_sp'], mode='markers', name='Measured'))
        if np.isfinite(r0w):
            xs = np.linspace(rp.min(), rp.max(), 100) * pitch_t
            fig_r.add_trace(go.Scatter(x=xs, y=KOLMOGOROV_SF * (xs / r0w) ** (5 / 3) + (tb['c_sp'] if with_off else 0.0),
                                       name=f'6.88 (r/r0)^(5/3), r0 = {r0w * 100:.1f} cm', mode='lines', line=dict(dash='dash')))
        fig_r.add_vrect(x0=fit_sep[0] * pitch_t, x1=fit_sep[1] * pitch_t, fillcolor='gray', opacity=0.12, line_width=0)
        fig_r.update_xaxes(type='log', title_text='Separation r (m)')
        fig_r.update_yaxes(type='log', title_text=f'D(r) (rad^2 at {wl_f:.0f} nm)')
        fig_r.update_layout(height=380, title='Spatial structure function of the POL', margin=dict(t=40))
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption("**How to read:** each point is how much two actuators differ in phase (rad²) as a function "
                   "of their separation. The dashed line is the Kolmogorov law 6.88 (r/r0)^(5/3) with the fitted "
                   "r0, drawn over the gray fit window. Points that follow the dashed line there support the r0 "
                   "value. A lower curve means a larger r0, which is better seeing.")
        st.caption("The static (time-mean) pattern of each actuator is removed first. Piston cancels in a "
                   "difference, so it does not matter here. Separations beyond a few pitches are pulled down by "
                   "the outer scale and the finite pupil; separations near one pitch are pulled down by the WFS "
                   "subaperture averaging. If the points do not follow a straight 5/3 line inside the window, "
                   "do not trust r0.")
        with st.expander("How these are defined"):
            st.markdown(
                "- **Structure function:** D(tau) = < [phi(t+tau) - phi(t)]^2 >, phi = POL phase in rad at the "
                "chosen wavelength = 2 pi (nm/V x volts) / lambda.\n"
                "- **tau0:** for frozen flow D(tau) = 6.88 (v tau / r0)^(5/3), so D(tau0) = 1 rad^2 gives "
                "tau0 = 0.314 r0 / v. It is found by fitting A tau^(5/3) in the chosen window and solving "
                "A tau0^(5/3) = 1. The PSD route integrates D(tau) = 2 integral P(f)(1 - cos 2 pi f tau) df "
                "after subtracting the white-noise floor.\n"
                "- **r0:** fit D(r) = c + 6.88 (r/r0)^(5/3) over the chosen separations.\n"
                "- **Not included:** r0 from the DM tip/tilt. The tip/tilt stage takes most of the tilt out of the DM "
                "command, so tilt-based r0 should use the tip/tilt slopes (the Tip/Tilt Temporal PSD option, whose r0 formula is now corrected).")
    else:
        st.info("Tick the box above to compute. Results are cached for the chosen settings.")


def render_performance_plots(xin_path, xin_name):
    """Closed-loop (residual) vs open-loop (POL) temporal and spatial PSDs (with actuator maps), diagnostics."""
    res, cmd, t = load_xinetics(xin_path)
    if res.ndim != 2 or res.shape[1] != 349:
        st.error(f"`residualWaveFront` has shape {res.shape}, expected (n, 349).")
        return
    if cmd is None or cmd.shape != res.shape:
        st.error("`DMCommand` is missing or has the wrong shape, so POL cannot be built.")
        return

    n_frames = res.shape[0]
    fs = estimate_fs(t)
    dropped = repeated = None
    try:
        with np.load(xin_path) as dd:
            if 'frameCounter' in dd.files:
                step = np.diff(dd['frameCounter'].astype(np.int64))
                repeated = int(np.sum(step == 0))                    # same frame counter twice in a row
                dropped = int(np.sum(step[step > 1] - 1))            # frames missing from the sequence
    except Exception:
        pass
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Sample rate", f"{fs:.1f} Hz")
    h2.metric("Frames", f"{n_frames:,}")
    h3.metric("Dropped frames", "n/a" if dropped is None else dropped,
              help="Frames missing from the frameCounter sequence (steps of 2 or more).")
    h4.metric("Repeated frames", "n/a" if repeated is None else repeated,
              help="Consecutive samples with the same frameCounter (steps of 0).")

    # ---------- Shared settings ----------
    st.write("**Settings (shared by all plots)**")
    c1, c2 = st.columns(2)
    with c1:
        lag = st.number_input(
            "POL lag (frames)", 0.0, 5.0, 1.7, 0.1, format="%.2f", key='pp_lag',
            help="POL[k] = residual[k] - the command on the mirror `lag` frames earlier. Fractional values "
                 "blend the two neighboring commands. 1.7 is the PyWFS delay from the literature; the lag "
                 "cannot be measured from telemetry alone, so confirm it with the instrument team.")
    with c2:
        exclude_edge = st.checkbox(
            "Exclude outer actuator ring", value=True, key='pp_edge',
            help="Edge actuators are poorly constrained by the WFS and often have huge alternating values.")
    
    # Hardcoded: POL = residual - delayed command. Verified on this system: the measured rejection curve matches
    # the leaky-integrator model only with this sign (the stored command obeys cmd[k] = leak*cmd[k-1] - g*res[k]).
    sign = -1
    lag = float(lag)
    
    _, _, a_fit, b_fit = controller_timing(xin_path, 0.99, 8, exclude_edge)
    scale, unit = unit_controls('pp')
    rms_res, rms_pol = compute_rms_summary(xin_path, lag, exclude_edge, sign)
    
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Residual RMS, closed loop ({unit})", f"{rms_res * scale:.3g}")
    r2.metric(f"POL RMS, open-loop est. ({unit})", f"{rms_pol * scale:.3g}")
    r3.metric("Residual / POL RMS", f"{rms_res / rms_pol:.2f}")

    tab_t, tab_s, tab_d = st.tabs(["Temporal PSD", "Spatial PSD and actuator maps", "Diagnostics"])
    
    # ================= Temporal =================
    with tab_t:
        st.info("**DM-actuator temporal PSD.** Built from `residualWaveFront` and the POL of the 349 xinetics "
                "actuators (volts, or nm), averaged over actuators. It is **not** the *Tip/Tilt Temporal PSD* "
                "option, which uses the tip/tilt slope arrays (UTT / DTT, arcseconds) from the tip/tilt files. The "
                "two measure different quantities on different hardware, so their levels are not directly "
                "comparable without projecting the DM onto tip/tilt.")
        t1, t2, t3 = st.columns(3)
        with t1:
            nperseg = st.select_slider("Welch segment length", [256, 512, 1024, 2048, 4096], value=1024,
                                       key='pp_nperseg',
                                       help="Number of samples in each chunk Welch's method transforms before averaging the spectra. Frequency resolution = sample rate / segment length, and the lowest frequency you can resolve is about the same number. Longer segments resolve narrow peaks and slow behavior but leave fewer segments to average, so the curve is noisier. Shorter segments are smoother but blur peaks and cut off low frequencies. Also used by the rejection curve and by tau0 (PSD route).")
            st.caption(f"Resolution ~ {fs / nperseg:.2f} Hz; about {max(1, 2 * n_frames // int(nperseg) - 1)} "
                       f"averaged segments (50% overlap).")
        with t2:
            rej_method = st.radio("Rejection estimate", ["sqrt(Residual/POL)", "Cross-spectral"], key='pp_rej',
                                  help="The ratio is biased by WFS noise above the loop bandwidth. The "
                                       "cross-spectral estimate is less sensitive to it.")
        with t3:
            show_model = st.checkbox("Overlay leaky-integrator model", value=True, key='pp_model')
        gain = leak = delay = None
        if show_model:
            m1, m2, m3 = st.columns(3)
            with m1:
                gain = st.number_input("Model gain g", 0.0, 2.0, float(np.clip(abs(b_fit), 0, 2)), 0.01,
                                       format="%.3f", key=f'pp_gain_{xin_name}',
                                       help="Defaults to |b| from the controller timing fit.")
            with m2:
                leak = st.number_input("Model leak k", 0.0, 1.0, float(np.clip(a_fit, 0, 1)), 0.001,
                                       format="%.3f", key=f'pp_leak_{xin_name}')
            with m3:
                delay = st.number_input("Model delay (frames)", 0.0, 6.0, lag, 0.1, format="%.2f",
                                        key=f'pp_delay_{xin_name}_{lag}')

        f, Pr, Pp, rej_ratio, rej_cross, fs_ = compute_temporal_psd(xin_path, lag, int(nperseg), exclude_edge, sign)
        rej = rej_ratio if rej_method.startswith('sqrt') else rej_cross
        sl = f > 0

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Temporal PSD", "Rejection transfer function"])
        fig.add_trace(go.Scatter(x=f[sl], y=Pp[sl] * scale ** 2, name='Open loop (POL)',
                                 line=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=f[sl], y=Pr[sl] * scale ** 2, name='Closed loop (residual)',
                                 line=dict(color='#ff7f0e')), row=1, col=1)
        fig.add_trace(go.Scatter(x=f[sl], y=rej[sl], name=f'Measured: {rej_method}',
                                 line=dict(color='green')), row=1, col=2)
        if show_model:
            fig.add_trace(go.Scatter(x=f[sl], y=leaky_integrator_rtf(f[sl], fs_, gain, leak, delay),
                                     name=f'Model (g={gain:.3g}, k={leak:.3g}, d={delay:.3g})',
                                     line=dict(color='black', dash='dash')), row=1, col=2)
        fig.add_hline(y=1, line=dict(color='gray', width=1), row=1, col=2)
        for c in (1, 2):
            fig.update_xaxes(type='log', title_text='Frequency (Hz)', row=1, col=c)
        fig.update_yaxes(type='log', title_text=f'PSD ({unit}²/Hz)', row=1, col=1)
        fig.update_yaxes(type='log', title_text='|Rejection|', row=1, col=2)
        fig.update_layout(height=480, legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

        above = np.nonzero(sl & (rej >= 1))[0]
        st.caption(
            "**How to read:** *Left:* blue (POL) is the turbulence the loop had to correct and orange (residual) is "
            "what is left; the gap between them is the correction. *Right:* the rejection transfer function. Below 1 "
            "the loop attenuates that frequency, and the frequency where the curve crosses 1 is the rejection "
            "bandwidth. If the measured curve follows the dashed model, the loop behaves as modeled and the POL "
            "sign and lag are consistent.")
        st.caption(
            f"Welch PSDs averaged over {'inner' if exclude_edge else 'all'} actuators, units {unit}²/Hz"
            + (f" ({scale:g} nm of wavefront per volt). " if unit == 'nm' else " (no volts-to-nm conversion applied). ")
            + (f"Measured 0 dB crossover ≈ {f[above[0]]:.0f} Hz." if len(above) else ""))

        with st.expander("What is the rejection transfer function, the waterbed effect and the leaky integrator?"):
            st.markdown(
                "**Rejection transfer function (RTF).** At each frequency it is the amplitude ratio "
                "residual / POL, equal to 1/(1+G) where G is the loop's open-loop gain. A value of 0.1 means the "
                "residual keeps one tenth of the turbulence amplitude there. Below 1 the loop helps; where the "
                "curve crosses 1 (the rejection bandwidth) it stops helping.\n\n"
                "**Waterbed effect.** A feedback loop with delay cannot reject at every frequency. Pushing the "
                "curve well below 1 at low frequency forces it above 1 somewhere higher, because the area below 1 "
                "and the area above 1 on a log-frequency axis must balance. So the bump just beyond the "
                "bandwidth, where the loop slightly amplifies some frequencies, is normal.\n\n"
                "**Leaky integrator (the controller).** Each frame the command is updated as "
                "`cmd[k] = leak * cmd[k-1] - g * res[k]`. A pure integrator (leak = 1) accumulates every residual "
                "forever, which gives perfect static rejection but lets noise and offsets pile up. A leak slightly "
                "below 1 (here ~0.99) makes the command slowly forget its past. That caps the DC gain at "
                "g / (1 - leak), so the best low-frequency rejection is 1 / (1 + g / (1 - leak)). For g = 0.1 and "
                "leak = 0.99 that is about 0.09, which is the plateau at the low-frequency end of the curve.\n\n"
                "**The dashed model** is this controller plus the pipeline delay, using the gain, leak and delay "
                "set above it. It uses one gain for all modes, whereas Keck runs different high- and low-order "
                "gains, so the actuator-averaged curve will not match exactly.\n\n"
                "**Why the curve need not return exactly to 1 at Nyquist.** There the controller gain is only "
                "about g / (1 + leak), so the loop barely acts, but the delay adds phase and the curve settles "
                "near 1/(1+G) rather than exactly 1. Compare against the model rather than against the 1 line.")

        with st.expander("Which rejection estimate should I use: sqrt(Residual/POL) or cross-spectral?"):
            st.markdown(
                "**sqrt(Residual/POL)** uses power only: the square root of the residual PSD divided by the POL "
                "PSD. It is simple, but anything in the residual that is not turbulence, mainly WFS noise, is "
                "counted as unrejected turbulence. This inflates the curve at high frequency, where noise "
                "dominates.\n\n"
                "**Cross-spectral** uses |P(res,POL) / P(POL,POL)|, the cross-spectrum of residual and POL divided "
                "by the POL power. This is the standard linear-system estimate of how the output (residual) "
                "follows the input (POL). Noise in the residual that is uncorrelated with the POL averages out of "
                "the cross-spectrum, so it is less biased by noise.\n\n"
                "**In practice.** Neither is perfect, because noise also enters the POL (it is built from the "
                "residual). The two agree where turbulence dominates, below the bandwidth. If they separate at "
                "high frequency, that is the noise-contaminated region and the cross-spectral curve is the more "
                "trustworthy one there.")

    # ================= Spatial =================
    with tab_s:
        s1, s2 = st.columns(2)
        with s1:
            n_use = st.number_input("Frames to average", 50, max(50, n_frames), min(1000, n_frames), 50,
                                    key='pp_nuse', help="Evenly spaced frames across the file.")
        with s2:
            show_ratio = st.checkbox("Show Residual / POL ratio", value=True, key='pp_ratio')

        freqs, profiles, N_eff = compute_spatial_psd(xin_path, lag, int(n_use), exclude_edge, sign)
        ncols = 2 if show_ratio else 1
        fig_s = make_subplots(rows=1, cols=ncols,
                              subplot_titles=(["Spatial PSD", "Residual / POL"][:ncols]))
        names = {'POL': 'Open loop (POL)', 'Residual': 'Closed loop (residual)'}
        colors = {'Residual': '#ff7f0e', 'POL': '#1f77b4'}
        for key in ('POL', 'Residual'):
            fig_s.add_trace(go.Scatter(x=freqs, y=profiles[key] * scale ** 2, mode='lines+markers', name=names[key],
                                       line=dict(color=colors[key])), row=1, col=1)
        fig_s.update_xaxes(type='log', title_text='Spatial frequency (cycles/pupil)', row=1, col=1)
        fig_s.update_yaxes(type='log', title_text=f'Power ({unit}² per actuator, per bin)', row=1, col=1)
        if show_ratio:
            fig_s.add_trace(go.Scatter(x=freqs, y=np.sqrt(profiles['Residual'] / profiles['POL']),
                                       mode='lines+markers', name='sqrt(Residual / POL)',
                                       line=dict(color='green')), row=1, col=2)
            fig_s.add_hline(y=1, line=dict(color='black', width=1), row=1, col=2)
            fig_s.update_xaxes(type='log', title_text='Spatial frequency (cycles/pupil)', row=1, col=2)
            fig_s.update_yaxes(type='log', title_text='Amplitude ratio', row=1, col=2)
        fig_s.update_layout(height=450, legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption(
            "**How to read:** the horizontal axis is spatial scale in cycles per pupil (1 = pupil-sized "
            "structure, larger = finer detail, up to the DM's limit). *Left:* blue (POL) is the turbulence "
            "the DM has to follow and normally falls steeply toward fine scales; orange (residual) is what the loop "
            "leaves. *Right:* sqrt(residual/POL) is the spatial rejection. Small values mean that scale is well "
            "corrected; the ratio rises toward 1 near the DM's correction limit, where the loop can no longer help. "
            "A ratio above 1 means the loop adds error at that scale.")
        st.caption(
            f"Average of {min(int(n_use), n_frames - first_valid_index(lag))} per-frame 2D power spectra (piston "
            f"removed, no window, zero-padded), radially averaged to Nyquist (~{N_eff / 2:.1f} cycles/pupil, "
            f"effective pupil {N_eff} actuators). Units are {unit}²"
            + (f" ({scale:g} nm of wavefront per volt). " if unit == 'nm' else " (no volts-to-nm conversion applied). ") +
            "Residual is what the loop failed to correct, not the turbulence itself.")

    # ================= Actuator maps (inside the Spatial PSD tab) =================
    with tab_s:
        st.markdown("---")
        st.markdown("#### Actuator maps")
        k_min = first_valid_index(lag)
        t_rel = (t - t[0]) if (t is not None and len(t) == n_frames) else None
        k = st.slider("Frame", k_min, n_frames - 1, k_min, 1, key='pp_frame')
        if t_rel is not None:
            st.caption(f"Frame {k}  |  t = {t_rel[k]:.3f} s")
        subtract_piston = st.checkbox("Subtract piston for display", value=True, key='pp_piston')

        cmd_k = delayed_cmd(cmd, np.array([k]), lag)[0]
        panels = [('Closed loop: residual', res[k]), ('DM command', cmd_k),
                  ('Open loop: POL', res[k] + sign * cmd_k)]
        fig_m = make_subplots(rows=1, cols=len(panels), subplot_titles=[p_[0] for p_ in panels],
                              horizontal_spacing=0.09)
        for i, (title, v) in enumerate(panels, start=1):
            v = v - v.mean() if subtract_piston else v
            grid = vector_to_grid(v * scale)
            vmax = np.nanmax(np.abs(grid)) or 1.0
            cax = 'coloraxis' if i == 1 else f'coloraxis{i}'
            fig_m.add_trace(go.Heatmap(z=grid, coloraxis=cax,
                                       hovertemplate=f'row %{{y}}, col %{{x}}<br>%{{z:.3f}} {unit}<extra></extra>'),
                            row=1, col=i)
            fig_m.update_layout(**{cax: dict(
                colorscale='RdBu_r', cmin=-vmax, cmax=vmax,
                colorbar=dict(title=unit, len=0.8, thickness=12,
                              x=(i / len(panels)) - 0.015 if i < len(panels) else 1.0))})
            fig_m.update_yaxes(autorange='reversed', showticklabels=False, row=1, col=i,
                               scaleanchor=f'x{i if i > 1 else ""}', scaleratio=1)
            fig_m.update_xaxes(showticklabels=False, row=1, col=i)
        fig_m.update_layout(height=420, margin=dict(t=40, b=10))
        st.plotly_chart(fig_m, use_container_width=True)
        st.caption("**How to read:** each map is one frame on the 21x21 actuator grid. *Residual* is the error "
                   "left after correction and should be small and unstructured. *DM command* is the correction "
                   "being applied. *POL* (residual minus command) is the estimated turbulence. When the loop works "
                   "well the command and the POL look like mirror images of each other (opposite sign) and the "
                   "residual is much weaker than both. Each panel has its own symmetric color scale; blank = "
                   "outside the pupil. A smooth, coherent interior indicates the actuator ordering is correct. "
                   "Edge actuators often show large alternating values.")

        if st.checkbox("Show whole-file residual maps (mean and RMS per actuator)", value=False, key='pp_avgmaps'):
            fig_a = make_subplots(rows=1, cols=2, subplot_titles=["Mean residual (static error)",
                                                                  "Residual RMS per actuator"],
                                  horizontal_spacing=0.12)
            for i, (arr, cax) in enumerate([(res.mean(axis=0), 'coloraxis'), (res.std(axis=0), 'coloraxis2')], 1):
                fig_a.add_trace(go.Heatmap(z=vector_to_grid(arr * scale), coloraxis=cax), row=1, col=i)
                fig_a.update_yaxes(autorange='reversed', showticklabels=False, row=1, col=i)
                fig_a.update_xaxes(showticklabels=False, row=1, col=i)
            fig_a.update_layout(coloraxis=dict(colorscale='RdBu_r', colorbar=dict(title=unit, x=0.44, len=0.8)),
                                coloraxis2=dict(colorscale='Viridis', colorbar=dict(title=unit, len=0.8)),
                                height=420, margin=dict(t=40, b=10))
            st.plotly_chart(fig_a, use_container_width=True)
            st.caption("**How to read:** *Left* is the time-averaged residual on each actuator, a persistent static "
                       "error (offsets, non-common-path or WFS bias) that should be near zero. *Right* is how much "
                       "each actuator's residual varies over the file; bright regions are poorly corrected, noisy "
                       "or low-flux parts of the pupil. Edge actuators are often brightest.")

    # ================= Diagnostics =================
    with tab_d:
        st.markdown("#### Lag sensitivity")
        st.write("The POL lag is a physical delay (WFS exposure, readout, compute, DM) that the stored arrays do "
                 "not reveal, so treat it as an input and check how much it matters. Results that change "
                 "dramatically across plausible lags should not be trusted.")
        chosen = st.multiselect("Lags to compare (frames):", [0.0, 1.0, 1.5, 1.7, 2.0, 2.2, 2.5, 3.0],
                                default=[1.0, 1.7, 2.0], key='pp_lags')
        if chosen:
            fig_l = go.Figure()
            for lg in chosen:
                fl, _, _, rr, rc, _ = compute_temporal_psd(xin_path, float(lg), int(nperseg), exclude_edge, sign)
                yy = rr if rej_method.startswith('sqrt') else rc
                fig_l.add_trace(go.Scatter(x=fl[fl > 0], y=yy[fl > 0], name=f'lag = {lg}'))
            fig_l.add_hline(y=1, line=dict(color='gray', width=1))
            fig_l.update_xaxes(type='log', title_text='Frequency (Hz)')
            fig_l.update_yaxes(type='log', title_text='|Rejection|')
            fig_l.update_layout(height=380)
            st.plotly_chart(fig_l, use_container_width=True)
            st.caption("**How to read:** each line is the rejection curve computed with a different assumed POL lag. "
                       "If the lines nearly overlap, your conclusions do not depend on the lag; the farther apart "
                       "they are, the more the lag matters. A plausible lag gives a smooth curve that rises, "
                       "overshoots 1 slightly and returns, without a sharp spike or dip.")

        st.markdown("#### Controller timing check")
        st.write("Correlates the command update, cmd[k] - leak*cmd[k-1], with the residual m frames earlier. "
                 "A sharp peak at m = 0 means cmd[k] was computed from res[k]; m = 1 means the stored command is "
                 "one frame behind the residual it responds to. This describes how the stored arrays line up, "
                 "not the physical delay.")
        leak_t = st.number_input("Controller leak", 0.0, 1.0, 0.99, 0.001, format="%.3f", key='pp_timing_leak')
        ms_, corr_, a_t, b_t = controller_timing(xin_path, float(leak_t), 8, exclude_edge)
        fig_t = go.Figure(go.Bar(x=ms_, y=np.abs(corr_)))
        fig_t.update_layout(height=300, xaxis_title='m (frames)', yaxis_title='|correlation|')
        st.plotly_chart(fig_t, use_container_width=True)
        st.caption("**How to read:** each bar is the correlation between the command update and the residual m "
                   "frames earlier. One dominant bar at m = 0 means each command was computed from the residual of "
                   "the same frame; a peak at m = 1 means the stored command is one frame behind. Several similar "
                   "bars means the alignment is ambiguous.")
        pk = int(np.argmax(np.abs(corr_)))
        st.caption(f"Peak at m = {int(ms_[pk])}, correlation sign {'+' if corr_[pk] > 0 else '-'}. "
                   f"Least-squares fit cmd[k] = a*cmd[k-1] + b*res[k-m]: a = {a_t:.3f} (leak), "
                   f"b = {b_t:.3f} (effective gain with sign). A negative b means POL = residual - command.")



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

# 1. Sidebar directory selector (Night)
selected_night = st.sidebar.selectbox(
    "Select Observing Night", 
    [d.name for d in night_dirs]
)
selected_night_path = base_dir / selected_night

# 2. Sidebar Guide Star Mode Filter
mode_filter = st.sidebar.radio(
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

# 3. Sidebar FITS Directory Selector
if not filtered_dirs:
    st.sidebar.warning(f"No FITS directories found for mode '{mode_filter}'.")
    st.stop()
else:
    selected_fits_dir_name = st.sidebar.selectbox(
        "Select FITS Directory",
        [d.name for d in filtered_dirs]
    )
    # Recover the full Path object based on the user's selection
    selected_fits_dir = next(d for d in filtered_dirs if d.name == selected_fits_dir_name)

st.divider()

# 3 & 4. Select Filtered Folder and Graph Type
col1, col2 = st.columns(2)


with col2:
    # Base list of all possible graphs
    graph_options = [
        'Up Tip Tilt', 
        'Down Tip Tilt', 
        'Down-TT Strain Gauge', 
        'Up-TT Strain Gauge', 
        'Shack Hartmann WFS Data',
        'Tip/Tilt Temporal PSD',
        'Performance Plots',
        #'Turbulence Parameters (tau0, r0, theta0)'#
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
                
                if 'TTslope1' in temp_data.files:
                    time_length = len(temp_data['TTslope1'])
                    st.markdown(f"**`TT` length:** `{time_length}`")
                
                # Close the file object to prevent memory leaks
                temp_data.close() 
            except Exception as e:
                st.error(f"**`{fpath.name}`**: Could not read file ({e})")
st.write("---")




# ---- Plain-language descriptions shown under the legacy plots ----
TT_CAPTIONS = {
    'Tip Distribution (Histogram)': "**How to read:** how often each tip (X) slope value occurred in the file. With the tip/tilt loop closed these are the residual slopes, so a narrow peak near zero means the guide star is held steady, and the width is the leftover jitter. A peak away from zero is a static offset, and several peaks or long tails suggest steps, drifts or saturation.",
    'Tilt Distribution (Histogram)': "**How to read:** how often each tilt (Y) slope value occurred in the file. With the tip/tilt loop closed these are the residual slopes, so a narrow peak near zero means the guide star is held steady, and the width is the leftover jitter. A peak away from zero is a static offset, and several peaks or long tails suggest steps, drifts or saturation.",
    'Tip Sequence (By Sample)': "**How to read:** the tip slope measured at every sample, in time order. With the loop closed this is the small residual the mirror did not remove, so it looks like noise around zero; slopes much larger than the mirror's correction would mean the loop is open. Look for steps, drifts and regular oscillations (vibration). Drag to zoom and hover for exact values.",
    'Tilt Sequence (By Sample)': "**How to read:** the tilt slope measured at every sample, in time order. With the loop closed this is the small residual the mirror did not remove, so it looks like noise around zero; slopes much larger than the mirror's correction would mean the loop is open. Look for steps, drifts and regular oscillations (vibration). Drag to zoom and hover for exact values.",
    'Mirror Correction: Tip': "**How to read:** the tip command sent to the tip/tilt mirror at every sample. It follows the tilt the mirror is cancelling (atmosphere plus telescope motion), so it wanders by much more than the residual slopes. It is not a sample-by-sample mirror image of the slopes: the controller accumulates them, so for an integrator the change in command from one sample to the next is proportional to minus the slope. A flat line at a limit would mean the mirror has run out of range.",
    'Mirror Correction: Tilt': "**How to read:** the tilt command sent to the tip/tilt mirror at every sample. It follows the tilt the mirror is cancelling (atmosphere plus telescope motion), so it wanders by much more than the residual slopes. It is not a sample-by-sample mirror image of the slopes: the controller accumulates them, so for an integrator the change in command from one sample to the next is proportional to minus the slope. A flat line at a limit would mean the mirror has run out of range.",
}
SHWFS_CAPTIONS = {
    'Average Subaperture Flux Over Time': "**How to read:** the mean intensity over all subapertures for each frame. A steady line means a stable guide star. Dips or drops point to clouds, loss of lock or a star leaving the field, and a slow trend points to changing transparency or airmass.",
    'Centroid Distribution (Histogram)': "**How to read:** how often each centroid value occurred across all subapertures and frames. A narrow distribution centered where you expect means well-centered spots. Wide tails, double peaks or a pile-up at an edge suggest spots running off their windows, saturation or a misaligned sensor.",
}
STRAIN_CAPTIONS = {
    'cmd': "**How to read:** the command sent to the mirror at every sample, one line per channel. It shows the motion pattern that was requested (for example the steps or ramps used for calibration). Hover for the value at any sample.",
    'strain': "**How to read:** the raw strain gauge reading at every sample, one line per channel. Each channel should follow the commanded pattern above; a flat, noisy or drifting channel suggests a dead gauge or a bad connection. The calibration below multiplies these by 1e6 to give microstrain. Hover for the value at any sample.",
    'multi': "**How to read:** each point compares the strain a channel actually measured (vertical) with what a linear fit to the commanded tip/tilt predicts (horizontal). Points on the dashed 1:1 line mean the gauge responds linearly to the command. Because the expected strain is a least-squares fit, the useful information is the scatter and curvature around the line (hysteresis, nonlinearity, cross-talk), not the slope.",
    'pct': "**How to read:** measured strain as a percentage of the predicted strain, against the predicted strain. 100% (red dashed line) means the gauge saw exactly the motion the fit expects. Only points where the prediction exceeds 5% of its maximum are shown, since the ratio is meaningless near zero. A band spread wide around 100% indicates hysteresis or noise, and a systematic offset indicates a calibration error.",
}


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
                
                raw_cols = st.columns(2)
                x_idx = np.arange(dtt_cmd.shape[0])
                with raw_cols[0]:
                    fig_c = go.Figure()
                    for ch in range(dtt_cmd.shape[1] if dtt_cmd.ndim > 1 else 1):
                        y_c = dtt_cmd[:, ch] if dtt_cmd.ndim > 1 else dtt_cmd
                        fig_c.add_trace(go.Scattergl(x=x_idx, y=y_c, mode='lines', name=f'Ch {ch}', line=dict(width=1)))
                    fig_c.update_layout(title=dict(text=f"<b>Raw DTT Mirror Commands</b>", x=0.5, font=dict(size=14)),
                                        xaxis_title="Sample index", yaxis_title="Command value", height=350,
                                        margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified",
                                        legend=dict(orientation="h", y=-0.25))
                    st.plotly_chart(fig_c, use_container_width=True)
                    st.caption(STRAIN_CAPTIONS['cmd'])
                with raw_cols[1]:
                    fig_s = go.Figure()
                    for ch in range(dtt_strain.shape[1] if dtt_strain.ndim > 1 else 1):
                        y_s = dtt_strain[:, ch] if dtt_strain.ndim > 1 else dtt_strain
                        fig_s.add_trace(go.Scattergl(x=np.arange(len(y_s)), y=y_s, mode='lines',
                                                     name=f'Ch {ch}', line=dict(width=1)))
                    fig_s.update_layout(title=dict(text=f"<b>Raw DTT Strain</b>", x=0.5, font=dict(size=14)),
                                        xaxis_title="Sample index", yaxis_title="Strain", height=350,
                                        margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified",
                                        legend=dict(orientation="h", y=-0.25))
                    st.plotly_chart(fig_s, use_container_width=True)
                    st.caption(STRAIN_CAPTIONS['strain'])

                
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
                        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
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
                        st.caption(STRAIN_CAPTIONS['multi'])
                        
                if 'Percent Realized vs Expected Strain' in selected_subgraphs:
                    col_idx = 1 if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs else 0
                    with cols[col_idx]:
                        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
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
                        st.caption(STRAIN_CAPTIONS['pct'])
                        
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
                    
                    raw_cols = st.columns(2)
                    x_idx = np.arange(utt_cmd.shape[0])
                    with raw_cols[0]:
                        fig_c = go.Figure()
                        for ch in range(utt_cmd.shape[1] if utt_cmd.ndim > 1 else 1):
                            y_c = utt_cmd[:, ch] if utt_cmd.ndim > 1 else utt_cmd
                            fig_c.add_trace(go.Scattergl(x=x_idx, y=y_c, mode='lines', name=f'Ch {ch}', line=dict(width=1)))
                        fig_c.update_layout(title=dict(text=f"<b>Raw UTT Mirror Commands</b>", x=0.5, font=dict(size=14)),
                                            xaxis_title="Sample index", yaxis_title="Command value", height=350,
                                            margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified",
                                            legend=dict(orientation="h", y=-0.25))
                        st.plotly_chart(fig_c, use_container_width=True)
                        st.caption(STRAIN_CAPTIONS['cmd'])
                    with raw_cols[1]:
                        fig_s = go.Figure()
                        for ch in range(utt_strain.shape[1] if utt_strain.ndim > 1 else 1):
                            y_s = utt_strain[:, ch] if utt_strain.ndim > 1 else utt_strain
                            fig_s.add_trace(go.Scattergl(x=np.arange(len(y_s)), y=y_s, mode='lines',
                                                         name=f'Ch {ch}', line=dict(width=1)))
                        fig_s.update_layout(title=dict(text=f"<b>Raw UTT Strain</b>", x=0.5, font=dict(size=14)),
                                            xaxis_title="Sample index", yaxis_title="Strain", height=350,
                                            margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified",
                                            legend=dict(orientation="h", y=-0.25))
                        st.plotly_chart(fig_s, use_container_width=True)
                        st.caption(STRAIN_CAPTIONS['strain'])
                    
                    
                    with st.expander("View Regression Coefficients (με / arcsec)"):
                        st.markdown(coeff_text)
                    
                    cols = st.columns(2)
                    
                    
                    
                    if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs:
                        with cols[0]:
                            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
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
                            st.caption(STRAIN_CAPTIONS['multi'])
                            
                    if 'Percent Realized vs Expected Strain' in selected_subgraphs:
                        col_idx = 1 if 'Multi-Axis Calibration (Actual vs Expected)' in selected_subgraphs else 0
                        with cols[col_idx]:
                            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
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
                            st.caption(STRAIN_CAPTIONS['pct'])
                            
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
            
            if not slope_key and not mirror_key:
                st.write("Available keys in this file:", utt_data.files)
                raise KeyError("Could not find expected array names for slopes or mirror commands.")

            # Either array may be missing: plot whichever one(s) this file has.
            tip = tilt = mirror_tip = mirror_tilt = None
            x_axis = x_mirror = None
            if slope_key:
                tt_slopes = utt_data[slope_key]
                tip = tt_slopes[:, 0]
                tilt = tt_slopes[:, 1]
                x_axis = np.arange(len(tip))
            else:
                st.info("No tip/tilt slope array in this file, so only the mirror command plots are available.")
            if mirror_key:
                mirror_commands = utt_data[mirror_key]
                mirror_tip = mirror_commands[:, 0]
                mirror_tilt = mirror_commands[:, 1]
                x_mirror = np.arange(len(mirror_tip))
            else:
                st.info("No mirror command array in this file, so only the slope plots are available.")
            
            
            # --- DYNAMIC GRAPH SELECTOR ---
            st.write("### Select Graphs to Display")
            graph_choices = []
            if slope_key:
                graph_choices += ['Tip Distribution (Histogram)', 'Tilt Distribution (Histogram)',
                                  'Tip Sequence (By Sample)', 'Tilt Sequence (By Sample)']
            if mirror_key:
                graph_choices += ['Mirror Correction: Tip', 'Mirror Correction: Tilt']
            
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
                                    st.caption(TT_CAPTIONS[graph_name])
                                
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
                                        fig.add_trace(go.Scatter(x=x_mirror, y=mirror_tip, mode='lines', line=dict(color='#1f77b4', width=1)))
                                        title = 'Mirror Correction: Tip'
                                    elif graph_name == 'Mirror Correction: Tilt':
                                        fig.add_trace(go.Scatter(x=x_mirror, y=mirror_tilt, mode='lines', line=dict(color='#d62728', width=1)))
                                        title = 'Mirror Correction: Tilt'
                                        
                                    fig.update_layout(
                                        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14)),
                                        xaxis_title="Sample Number",
                                        yaxis_title=("Command (arcseconds)" if graph_name.startswith('Mirror') else "Slope (arcseconds)"),
                                        height=350,
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        showlegend=False,
                                        hovermode="x unified"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.caption(TT_CAPTIONS[graph_name])
                                    
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
            
            possible_mirror_keys = ['DTTMirrorCommand', 'DTTMainMirrorCommand', 'MirrorCommand']
            mirror_key = next((key for key in possible_mirror_keys if key in dtt_data.files), None)

            if not slope_key and not mirror_key:
                st.write("Available keys in this file:", dtt_data.files)
                raise KeyError("Could not find expected array names for slopes or mirror commands.")

            # Either array may be missing: plot whichever one(s) this file has.
            tip = tilt = mirror_tip = mirror_tilt = None
            x_axis = x_mirror = None
            if slope_key:
                tt_slopes = dtt_data[slope_key]
                tip = tt_slopes[:, 0]
                tilt = tt_slopes[:, 1]
                x_axis = np.arange(len(tip))
            else:
                st.info("No tip/tilt slope array in this file, so only the mirror command plots are available.")
            if mirror_key:
                mirror_commands = dtt_data[mirror_key]
                mirror_tip = mirror_commands[:, 0]
                mirror_tilt = mirror_commands[:, 1]
                x_mirror = np.arange(len(mirror_tip))
            else:
                st.info("No mirror command array in this file, so only the slope plots are available.")
            
            # --- DYNAMIC GRAPH SELECTOR ---
            st.write("### Select Graphs to Display")
            graph_choices = []
            if slope_key:
                graph_choices += ['Tip Distribution (Histogram)', 'Tilt Distribution (Histogram)',
                                  'Tip Sequence (By Sample)', 'Tilt Sequence (By Sample)']
            if mirror_key:
                graph_choices += ['Mirror Correction: Tip', 'Mirror Correction: Tilt']
            
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
                                    st.caption(TT_CAPTIONS[graph_name])
                                
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
                                        fig.add_trace(go.Scatter(x=x_mirror, y=mirror_tip, mode='lines', line=dict(color='#1f77b4', width=1)))
                                        title = 'Mirror Correction: Tip'
                                    elif graph_name == 'Mirror Correction: Tilt':
                                        fig.add_trace(go.Scatter(x=x_mirror, y=mirror_tilt, mode='lines', line=dict(color='#d62728', width=1)))
                                        title = 'Mirror Correction: Tilt'
                                        
                                    fig.update_layout(
                                        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14)),
                                        xaxis_title="Sample Number",
                                        yaxis_title=("Command (arcseconds)" if graph_name.startswith('Mirror') else "Slope (arcseconds)"),
                                        height=350, # Matches rough height of matplotlib figures
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        showlegend=False,
                                        hovermode="x unified"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.caption(TT_CAPTIONS[graph_name])
                                    
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
                            _pad_l, _map_col, _pad_r = st.columns([1, 2, 1])
                            fig, ax = plt.subplots(figsize=(5, 4.2))
                            im = ax.imshow(maps_with_data[0], norm=LogNorm(vmin=v_min_sub, vmax=v_max_sub), cmap=cmap)
                            ax.set_title(active_keys[0], fontweight='bold')
                            ax.set_xlabel("x index")
                            ax.set_ylabel("y index")
                            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                            cbar.set_label("Subaperture intensity (ADU)", rotation=270, labelpad=15)
                            with _map_col:
                                st.pyplot(fig)
                                st.caption("**How to read:** each cell is one subaperture on the pupil layout and its color is the intensity (ADU) on a log scale, so a bright cell has more guide-star light. Black is outside the pupil. A smooth illumination pattern is normal; dark cells inside the pupil are dead or vignetted subapertures, and a one-sided gradient suggests misalignment or the star drifting on the sensor. The slider selects the frame and the expander sets the color limits.")
                        else:
                            _pad_l, _map_col, _pad_r = st.columns([1, 3, 1])
                            fig, axes = plt.subplots(2, 2, figsize=(8, 7))
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
                            with _map_col:
                                st.pyplot(fig)
                                st.caption("**How to read:** each cell is one subaperture on the pupil layout and its color is the intensity (ADU) on a log scale, so a bright cell has more guide-star light. Black is outside the pupil. A smooth illumination pattern is normal; dark cells inside the pupil are dead or vignetted subapertures, and a one-sided gradient suggests misalignment or the star drifting on the sensor. The slider selects the frame and the expander sets the color limits.")
                            
                
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
            
            telemetry_graphs = ['None', 'Average Subaperture Flux Over Time', 'Centroid Distribution (Histogram)']

            # One row of controls, then one row of plots, so both plots start at the same height and share a size.
            ctl1, ctl2, ctl3 = st.columns(3)
            with ctl1:
                sel_all_graph = st.selectbox("Combined graph (all sensors overlaid):", telemetry_graphs,
                                             key="all_sensors_graph")
            with ctl2:
                sel_single_sensor = (st.selectbox("Sensor for the isolated graph:", valid_sensors,
                                                  key="single_sensor_sel") if valid_sensors else None)
            with ctl3:
                sel_single_graph = st.selectbox("Isolated graph (one sensor):", telemetry_graphs,
                                                key="single_sensor_graph")

            plot_l, plot_r = st.columns(2)

            # --- LEFT: ALL SENSORS OVERLAY ---
            with plot_l:
                if sel_all_graph != 'None' and valid_sensors:
                    fig1, ax1 = plt.subplots(figsize=(6, 4))

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
                        fig1.tight_layout()
                        st.pyplot(fig1)
                        st.caption(SHWFS_CAPTIONS[sel_all_graph])

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
                        fig1.tight_layout()
                        st.pyplot(fig1)
                        st.caption(SHWFS_CAPTIONS[sel_all_graph])
                    plt.close(fig1)

            # --- RIGHT: SINGLE SENSOR ---
            with plot_r:
                if sel_single_sensor is not None and sel_single_graph != 'None':
                    fig2, ax2 = plt.subplots(figsize=(6, 4))

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
                            fig2.tight_layout()
                            st.pyplot(fig2)
                            st.caption(SHWFS_CAPTIONS[sel_single_graph])
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
                            fig2.tight_layout()
                            st.pyplot(fig2)
                            st.caption(SHWFS_CAPTIONS[sel_single_graph])
                        else:
                            st.warning(f"No Centroid data for {sel_single_sensor}")
                    plt.close(fig2)

        except Exception as e:
            st.error(f"Error processing SHWFS files: {e}")
        finally:
            if 'raw_data' in locals() and raw_data is not None: raw_data.close()
            if 'proc_data' in locals() and proc_data is not None: proc_data.close()
                
                
                
elif selected_graph == 'Tip/Tilt Temporal PSD':
        st.write("---")
        st.write("### Tip/Tilt Temporal PSD")
        st.info("**Tip/tilt slope temporal PSD.** Uses the tip/tilt slope arrays (UTT slopes 1-4 and DTT slope, in "
                "arcseconds) from the Up/Down tip-tilt files. It is **not** the temporal PSD in *Performance Plots*, "
                "which uses the xinetics DM actuators (residual and POL, volts) from the xinetics files. Compare "
                "them only after projecting the DM residual onto tip/tilt.")
        
        # Helper to map raw array keys to the requested display names
        def format_arr_name(key):
            mapping = {
                'SHWFS1TT': 'UTTslope1',
                'SHWFS2TT': 'UTTslope2',
                'SHWFS3TT': 'UTTslope3',
                'SHWFS4TT': 'UTTslope4',
                'TTslope': 'DTTslope',
                'TTslope1': 'DTTslope'
            }
            return mapping.get(key, key)
        
# 1. Find all available .npz files in the current folder
        npz_files = sorted(list(selected_fits_dir.glob("*.npz")))
        
        valid_tt_files = []
        file_display_names = {}
        
        for f in npz_files:
            fname_lower = f.name.lower()
            if 'tiptilt' in fname_lower or 'tt' in fname_lower:
                if 'up' in fname_lower:
                    has_utt_slope = False
                    try:
                        # Quick peek to see if UTT slope arrays (SHWFSxTT) are present
                        with np.load(f) as data:
                            if any(k in data.files for k in ['SHWFS1TT', 'SHWFS2TT', 'SHWFS3TT', 'SHWFS4TT']):
                                has_utt_slope = True
                    except Exception:
                        pass
                    
                    # If it has a UTT slope array, add it. Otherwise, filter it out completely.
                    if has_utt_slope:
                        valid_tt_files.append(f)
                        file_display_names[f.name] = f.name

                elif 'down' in fname_lower:
                    valid_tt_files.append(f)
                    has_dtt_slope = False
                    try:
                        # Quick peek to see if DTT slope arrays are present
                        with np.load(f) as data:
                            if 'TTslope' in data.files or 'TTslope1' in data.files:
                                has_dtt_slope = True
                    except Exception:
                        pass
                    
                    if has_dtt_slope:
                        file_display_names[f.name] = f.name
                    else:
                        file_display_names[f.name] = f"{f.name} ⚠️ (No DTTslope)"
        
        if not valid_tt_files:
            st.warning(f"Could not find any valid Up or Down TT .npz files in {selected_fits_dir.name}")
        else:
            selected_files = st.multiselect(
                "Select Data Files to Compare (Up/Down TT only):", 
                [f.name for f in valid_tt_files],
                default=[f.name for f in valid_tt_files],
                format_func=lambda x: file_display_names.get(x, x)
            )
            
            file_arrays_to_plot = {}
            
            if selected_files:
                st.write("**Select Arrays to Overlay**")
                array_cols = st.columns(len(selected_files))
                
                for idx, fname in enumerate(selected_files):
                    with array_cols[idx]:
                        data = np.load(selected_fits_dir / fname)
                        candidates = [k for k in data.files if data[k].ndim == 2 and data[k].shape[1] >= 2]
                        
                        # Filter out all mirror commands for both Up and Down files
                        candidates = [k for k in candidates if 'mirror' not in k.lower()]
                            
                        if candidates:
                            # Prioritize slope arrays as the default selection
                            def_arr = [c for c in ['TTslope', 'TTslope1', 'SHWFS1TT'] if c in candidates]
                            default_val = [def_arr[0]] if def_arr else [candidates[0]]
                            
                            # Use format_func to display mapped names in the dropdown
                            sel_arr = st.multiselect(
                                f"Arrays in `{fname}`:", 
                                candidates, 
                                default=default_val,
                                format_func=format_arr_name
                            )
                            if sel_arr:
                                file_arrays_to_plot[fname] = sel_arr
                        else:
                            st.warning(f"No valid 2D arrays in {fname}")
                        data.close()

            # 2. Add Scaling and Overlay UI
# 2. Add Scaling and Overlay UI
            st.write("---")
            st.write("**Graph Settings & Overlays**")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                x_scale = st.selectbox("X-Axis Scale:", ["log", "linear"], index=0)
            with col_s2:
                y_scale = st.selectbox("Y-Axis Scale:", ["log", "linear"], index=0)
            with col_s3:
                show_kolm = st.checkbox("Kolmogorov (-11/3)", value=True)
                
            if file_arrays_to_plot:
                import scipy.signal as signal
                from scipy.stats import linregress
                import pandas as pd
                
                # Reserve a spot for the graph so we can render the inputs below it first
                graph_container = st.container()
                
                st.write("---")
                st.markdown("#### Slope Analysis")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    fit_min = st.number_input("Fit Min (Hz):", value=1.0, min_value=0.01, step=1.0)
                with col_f2:
                    fit_max = st.number_input("Fit Max (Hz):", value=10.0, min_value=0.1, step=1.0)
                
                fig_psd, (ax_tip, ax_tilt) = plt.subplots(2, 1, figsize=(10, 10))
                
                kolm_slope = -11/3
                freq_ref = None
                colors = plt.cm.tab10.colors
                color_idx = 0 
                results = []
                
                for fname, arrays in file_arrays_to_plot.items():
                    fpath = selected_fits_dir / fname
                    data = np.load(fpath)
                    
                    fs, fs_source = get_sample_rate(data)
                    if fs_source.startswith('assumed'):
                        st.warning(f"`{fname}`: no usable timestamp found, assuming {fs:.0f} Hz. "
                                   "Frequencies on this plot may be wrong.")
                            
                    for arr_key in arrays:
                        sensor_data = data[arr_key]
                        nperseg = min(2048, sensor_data.shape[0] // 4)
                        c = colors[color_idx % len(colors)]
                        color_idx += 1
                        
                        label_name = format_arr_name(arr_key)
                        s_res = {"Signal": label_name, "Est. Hz": round(fs, 1), "Rate source": fs_source}
                        
                        freqs_tip, psd_tip = signal.welch(sensor_data[:, 0], fs=fs, nperseg=nperseg)
                        freqs_tilt, psd_tilt = signal.welch(sensor_data[:, 1], fs=fs, nperseg=nperseg)
                        
                        if freq_ref is None:
                            freq_ref = freqs_tip[(freqs_tip >= fit_min) & (freqs_tip <= fit_max)]
                            if len(freq_ref) == 0:
                                freq_ref = freqs_tip[(freqs_tip > 1) & (freqs_tip < 10)]
                            
                        for ax_name, freqs, psd, plot_ax in [("Tip (X)", freqs_tip, psd_tip, ax_tip), ("Tilt (Y)", freqs_tilt, psd_tilt, ax_tilt)]:
                            plot_ax.plot(freqs, psd, label=label_name, color=c, alpha=0.7, linewidth=1.5)
                            
                            mask = (freqs >= fit_min) & (freqs <= fit_max) & (freqs > 0) & (psd > 0)
                            if np.sum(mask) >= 3:
                                lf, lp = np.log10(freqs[mask]), np.log10(psd[mask])
                                slope, intc, _, _, _ = linregress(lf, lp)
                                s_res[ax_name] = slope
                                
                               
                            else:
                                s_res[ax_name] = None

                                

                        
                        
                        results.append(s_res)
                    data.close()
                    
                for plot_ax, title, psd_panel in zip([ax_tip, ax_tilt], ["Tip (X) PSD", "Tilt (Y) PSD"], [psd_tip, psd_tilt]):
                    # Add shaded region for the fit bounds
                    plot_ax.axvspan(fit_min, fit_max, color='gray', alpha=0.15, label='Fit Region')
                    
                    if show_kolm and freq_ref is not None and len(freq_ref) > 0:
                        mask = (freqs_tip >= fit_min) & (freqs_tip <= fit_max)
                        if np.any(mask):
                            anchor_power = np.mean(psd_panel[mask])
                            plot_ax.plot(freq_ref, anchor_power * (freq_ref / freq_ref[0])**kolm_slope, 'k-', linewidth=2, label='Kolmogorov (-11/3)')
                    
                    plot_ax.set_xscale(x_scale)
                    plot_ax.set_yscale(y_scale)
                    plot_ax.set_xlabel(f'Frequency (Hz) [{x_scale}]')
                    plot_ax.set_ylabel(f'PSD (arcsec²/Hz) [{y_scale}]')
                    plot_ax.set_title(title, fontweight='bold')
                    plot_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                    plot_ax.grid(True, alpha=0.3, which='both')
                    
                plt.tight_layout()
                
                # Push the finished plot into the container reserved at the top
                with graph_container:
                    st.pyplot(fig_psd)
                    st.caption(
                        "**How to read:** each panel is the temporal power spectrum of one slope axis (top: tip/X, "
                        "bottom: tilt/Y) in arcsec²/Hz, one colored line per selected array. More power at a "
                        "frequency means more image motion there. Sharp peaks are vibrations (telescope, cooler or "
                        "wind shake). The gray band is the fit window used for the slopes in the table below; the "
                        "black line is a Kolmogorov -11/3 reference anchored to the mean power in that window. "
                        "Turbulence-driven tilt is shallow at low frequency and approaches -11/3 above a few Hz, so "
                        "a measured slope much flatter than -11/3 points to vibration, noise or loop suppression "
                        "rather than pure turbulence. These are the slopes as recorded, so if the tip/tilt loop was "
                        "closed they show what the loop left behind.")
                
                # Display the results dataframe directly under the inputs
                if results:
                    df = pd.DataFrame(results).set_index("Signal")
                    if "Tip (X)" in df.columns and "Tilt (Y)" in df.columns:
                        df["Avg Slope"] = df[["Tip (X)", "Tilt (Y)"]].mean(axis=1)
                    st.dataframe(df.style.format("{:.2f}", na_rep="-", subset=[c for c in df.columns if c not in ["Est. Hz", "Rate source"]]), use_container_width=True)
                    st.caption("**How to read:** the table gives the fitted log-log slope of each PSD inside the fit "
                               "window (Tip (X), Tilt (Y), and their mean), the sample rate used and where it came "
                               "from. A Kolmogorov turbulence reference is -11/3 (about -3.67); flatter slopes mean "
                               "other sources (vibration, noise, loop correction) contribute in that band.")


elif selected_graph == 'Performance Plots':
    xin_files = [f for f in filtered_files if 'xinetics' in f.name.lower()]

    if not xin_files:
        st.warning(f"No xinetics `.npz` files found in `{selected_fits_dir.name}`")
    else:
        xin_name = st.selectbox("Select xinetics file:", [f.name for f in xin_files], key='pp_file')
        xin_path = str(next(f for f in xin_files if f.name == xin_name))
        try:
            render_performance_plots(xin_path, xin_name)
        except KeyError as e:
            st.error(f"Missing expected key in `{xin_name}`: {e}")
        except Exception as e:
            st.error(f"Could not process `{xin_name}`: {e}")

            
elif selected_graph == 'Turbulence Parameters (tau0, r0, theta0)':
    xin_files = [f for f in filtered_files if 'xinetics' in f.name.lower()]

    if not xin_files:
        st.warning(f"No xinetics `.npz` files found in `{selected_fits_dir.name}`")
    else:
        xin_name = st.selectbox("Select xinetics file:", [f.name for f in xin_files], key='tb_file')
        xin_path = str(next(f for f in xin_files if f.name == xin_name))
        try:
            render_turbulence(xin_path, xin_name)
        except KeyError as e:
            st.error(f"Missing expected key in `{xin_name}`: {e}")
        except Exception as e:
            st.error(f"Could not process `{xin_name}`: {e}")
            
