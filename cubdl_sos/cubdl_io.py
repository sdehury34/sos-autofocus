"""
Loader for the CUBDL plane-wave files (.hdf5) of the five institutions (OSL, MYO, UFL, EUT, INS).

The institutions store their data with different keys. The geometry and the time-zero convention of
each institution follow the official CUBDL loader (datasets/PWDataLoaders.py of the CUBDL code):

    inst  elements                     angles                     CUBDL time zero (sample time = tau_x0 + tz)
    OSL   element_positions, centred   transmit_direction[0]      tz = -start_time
    MYO   L11-4v, pitch 0.3 mm         angles (rad)               tz = x_max |sin a| / c
    UFL   LA533, pitch 0.245 mm        angles (deg)               tz = -channel_data_t0
    EUT   element_positions, centred   transmit_direction[:, 0]   tz = x_max |sin a| / c + 10 / fc
    INS   element_positions, centred   linspace(-16, 16) deg      tz = -start_time + x_max |sin a| / c

In this tool the sample time is tau - t0e, with tau referenced to x = 0 ("origin") or to the first-firing
aperture edge ("edge"; this adds x_max |sin a| / c, the CUBDL "time zero is the center point" term), and
t0e = t0_sign * t0 + t_off. The table CUBDL below gives these values for each institution.

Files with other keys are loaded with the generic rules (sizes, units, centring), and the convention
is then found by the search in run_sos_comparison.py. Run `python cubdl_io.py FILE` to print a file.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import h5py
from scipy.signal import hilbert
try:
    import hdf5plugin  # noqa: F401  (compression filters, e.g. in the OpenH-RF / zea files)
except ImportError:
    pass

# convention and depth range (m) of the CUBDL evaluation, for each institution
CUBDL = {
    "OSL": dict(tx_ref="origin", t0_sign=1.0, t_off=0.0, zlims=(10e-3, 65e-3)),
    "MYO": dict(tx_ref="edge", t0_sign=1.0, t_off=0.0, zlims=(5e-3, 55e-3), pitch=0.3e-3),
    "UFL": dict(tx_ref="origin", t0_sign=1.0, t_off=0.0, zlims=(10e-3, 50e-3), pitch=0.245e-3),
    "EUT": dict(tx_ref="edge", t0_sign=1.0, t_off="-10/fc", zlims=(10e-3, 80e-3)),
    "INS": dict(tx_ref="edge", t0_sign=1.0, t_off=0.0, zlims=(10e-3, 60e-3), max_angle=16.0),
}


@dataclass
class PWData:
    name: str
    data: np.ndarray        # (T, S, N) complex64 baseband (IQ) data
    is_iq: bool             # True: the file stores IQ data; False: the file stores RF (demodulated here)
    fs: float               # sampling frequency (Hz)
    fdemod: float           # demodulation frequency (Hz)
    c_file: float           # sound speed stored in the file (m/s)
    angles: np.ndarray      # (T,) steering angles (rad)
    t0: np.ndarray          # (T,) start time of each transmit (s), as stored in the file
    xe: np.ndarray          # (N,) element x positions (m)
    pix: np.ndarray         # (P, 3) pixel positions (m) of bf, or None
    bf: np.ndarray          # (P,) envelope of the file's beamformed image (linear), or None
    notes: list = field(default_factory=list)
    angles_missing: bool = False
    inst: str = ""
    conv: dict = None       # CUBDL convention of the institution (tx_ref, t0_sign, t_off), or None
    zlims: tuple = None     # CUBDL depth range (m), or None


def _scalar(f, keys, default=None):
    for k in ([keys] if isinstance(keys, str) else keys):
        if k in f:
            v = np.asarray(f[k][()]).astype(np.float64).ravel()
            if v.size: return float(v[0])
    return default


def _complex(a):
    """Return a complex array from complex or compound (real/imag) storage, or None."""
    if np.iscomplexobj(a): return a
    if a.dtype.names:
        names = [n.lower() for n in a.dtype.names]
        r = a[a.dtype.names[names.index("real") if "real" in names else 0]]
        i = a[a.dtype.names[names.index("imag") if "imag" in names else 1]]
        return r + 1j * i
    return None


def _xyz(a, name):
    """(K, 3) coordinates from an array with one axis of size 3; converts mm to m."""
    a = np.asarray(a, np.float64)
    ax = [i for i, s in enumerate(a.shape) if s == 3]
    if not ax: raise ValueError(f"{name}: no axis of size 3 in shape {a.shape}")
    a = np.moveaxis(a, ax[-1] if a.shape[-1] == 3 else ax[0], -1).reshape(-1, 3)
    if np.abs(a).max() > 1.0: a = a / 1000.0           # stored in mm
    return a


def is_zea(path):
    with h5py.File(path, "r") as f:
        return "tracks/track_0/data/raw_data" in f or "data/raw_data" in f


def load_zea(path, verbose=False, frame=0):
    """OpenH-RF / zea file: one frame of plane-wave RF or IQ data.
    Convention (zea): the transmit delay is referenced to the first-firing element (aperture edge), and
    the first sample is at initial_times[tx] after the trigger (tx_ref="edge", t0_sign=+1). The check
    against the reference B-mode of the file confirms it."""
    import hdf5plugin  # noqa: F401  (compression filters of the zea files)
    path = Path(path); notes = []
    if verbose: describe(path)
    with h5py.File(path, "r") as f:
        tr = f["tracks/track_0"] if "tracks/track_0" in f else f
        sc = tr["scan"]
        raw = tr["data/raw_data"]                                      # (frames, tx, samples, elements, ch)
        a = np.asarray(raw[frame])
        g = lambda k, d=None: np.asarray(sc[k][()]) if k in sc else d
        fs = float(g("sampling_frequency")); fdem = float(g("demodulation_frequency", 0.0) or 0.0)
        fc = float(g("center_frequency", fdem)); c_file = float(g("sound_speed", 1540.0))
        ang = np.asarray(g("polar_angles"), np.float64).ravel(); t0 = np.asarray(g("initial_times"), np.float64).ravel()
        foc = np.asarray(g("focus_distances", np.full(ang.size, np.inf)), np.float64).ravel()
        el = np.asarray(f["probe/probe_geometry"][()], np.float64)
        img = tr["data/image"] if "data/image" in tr else None
        bf = pix = None
        if img is not None and "values" in img and "coordinates" in img:
            v = np.asarray(img["values"][min(frame, img["values"].shape[0] - 1)], np.float64).squeeze()
            co = np.asarray(img["coordinates"][()], np.float64).reshape(-1, 3)
            unit = img["unit"][()].decode() if "unit" in img else ""
            if unit == "dB" or (v.max() <= 0 and v.min() < -1): bf = 10 ** (np.clip(v, -300, 0) / 20)
            else: bf = 10 ** ((v / max(v.max(), 1e-9) * 60 - 60) / 20); notes.append("reference image: display values, 60 dB assumed")
            bf, pix = bf.reshape(-1), co
            if bf.size != pix.shape[0]: bf = pix = None
        refs = {k: float(f[f"custom/{k}"][()]) for k in ("through_tx_speed_of_sound", "estimated_global_speed_of_sound")
                if f"custom/{k}" in f}
    if np.any(np.isfinite(foc) & (foc != 0)):
        notes.append("WARNING: some transmits are focused or diverging; this tool assumes plane waves")
    if a.ndim == 4 and a.shape[-1] == 2: a = a[..., 0] + 1j * a[..., 1]; is_iq = True
    else: a = a[..., 0] if a.ndim == 4 else a; is_iq = np.iscomplexobj(a)
    a = a.astype(np.complex64 if is_iq else np.float32)                # (tx, samples, elements)
    if not is_iq:
        # some acquisitions have a DC offset and low-frequency drift in the int16 RF: remove the mean of
        # each channel and keep the band 0.3 fc to 1.7 fc (zero-phase Butterworth), so that no
        # non-acoustic signal adds coherently in the image
        from scipy.signal import butter, sosfiltfilt
        a = a - a.mean(axis=1, keepdims=True)
        sos = butter(4, [0.3 * fc, min(1.7 * fc, 0.45 * fs)], btype="band", fs=fs, output="sos")
        a = sosfiltfilt(sos, a, axis=1).astype(np.float32)
        notes.append("RF: channel mean removed, band-pass 0.3-1.7 fc")
        a = hilbert(a, axis=1); fdem = fdem or fc
        a = a * np.exp(-2j * np.pi * fdem * np.arange(a.shape[1]) / fs)[None, :, None]
    xe = el[:, 0].copy()
    if abs(xe.mean()) > 0.25 * abs(xe[1] - xe[0]): notes.append("element positions centred"); xe -= xe.mean()
    if pix is not None and np.abs(pix).max() > 1.0: pix = pix / 1000.0
    zl = (max(5e-3, float(pix[:, 2].min())), float(pix[:, 2].max())) if pix is not None else None
    d = PWData(path.stem, a.astype(np.complex64), is_iq, fs, fdem, c_file, ang, t0, xe, pix, bf, notes, False,
               "ZEA", dict(tx_ref="edge", t0_sign=1.0, t_off=0.0), zl)
    d.refs = refs
    if refs: notes.append("reference speed of sound in the file: " + ", ".join(f"{k}={v:.1f}" for k, v in refs.items()))
    if verbose: describe(path, d)
    return d


def load(path, verbose=False, max_angle_deg=None, frame=0):
    if is_zea(path): return load_zea(path, verbose, frame)
    path = Path(path); notes = []; inst = path.stem[:3].upper(); tab = CUBDL.get(inst)
    if verbose: describe(path)                                         # print the layout first
    with h5py.File(path, "r") as f:
        raw = f["channel_data"][()]
        fs = _scalar(f, ["sampling_frequency", "channel_data_sampling_frequency"])
        fdem = _scalar(f, "modulation_frequency", 0.0) or 0.0
        c_file = _scalar(f, "sound_speed", 1540.0)
        ang = np.asarray(f["transmit_direction"][()] if "transmit_direction" in f else
                         f["angles"][()] if "angles" in f else np.zeros(1), np.float64)
        st = np.asarray(f["start_time"][()] if "start_time" in f else
                        f["channel_data_t0"][()] if "channel_data_t0" in f else np.zeros(1), np.float64).ravel()
        el = _xyz(f["element_positions"][()], "element_positions") if "element_positions" in f else None
        pix = _xyz(f["pixel_positions"][()], "pixel_positions") if "pixel_positions" in f else None
        bf = f["beamformed_data"][()] if "beamformed_data" in f else None
        bf_fs = _scalar(f, "beamformed_data_sampling_frequency"); bf_t0 = _scalar(f, "beamformed_data_t0")
        T = _scalar(f, "transmit_count"); N = _scalar(f, "element_count")

    # ---- channel data -> (T, S, N)
    cplx = _complex(raw)
    is_iq = cplx is not None
    a = np.squeeze(cplx if is_iq else np.asarray(raw, np.float32))
    if a.ndim == 2: a = a[None]
    if N is None: N = el.shape[0] if el is not None else None
    if T is None:
        T = st.size if st.size > 1 else (max(ang.shape) if ang.size > 1 else None)
    shp = list(a.shape)
    if T is None or N is None or T == N or T not in shp or N not in shp:
        it, ie, i_s = 0, 1, 2                                          # CUBDL layout (transmits, elements, samples)
        if T is None or N is None: notes.append(f"layout (T, N, S) assumed from the shape {tuple(shp)}")
    else:
        it, ie = shp.index(int(T)), shp.index(int(N)); i_s = [i for i in range(3) if i not in (it, ie)][0]
    a = np.transpose(a, (it, i_s, ie)); T, N = a.shape[0], a.shape[2]
    if not is_iq:
        # RF -> analytic signal -> baseband, with the time of each sample relative to the first sample.
        # The beamformer interpolates the smooth baseband signal and modulates it again with the exact
        # delay, so the image and its derivative with respect to c are smooth.
        a = hilbert(a, axis=1)
        if not fdem:                                                   # centre frequency from the spectrum
            spec = np.abs(np.fft.rfft(np.real(a[:, :, ::max(1, a.shape[2] // 16)]), axis=1)).mean(axis=(0, 2))
            fr = np.fft.rfftfreq(a.shape[1], 1 / fs); fdem = float((spec * fr).sum() / spec.sum())
        a = a * np.exp(-2j * np.pi * fdem * np.arange(a.shape[1]) / fs)[None, :, None]
    data = a.astype(np.complex64)

    # ---- elements: file positions (centred), or the probe pitch of the CUBDL loader
    if el is not None:
        xe = el[:, 0].copy()
        if abs(xe.mean()) > 0.25 * abs(xe[1] - xe[0]):
            notes.append(f"element positions centred (offset {1e3 * xe.mean():.2f} mm)")
            xe = xe - xe.mean()
    elif tab and "pitch" in tab:
        xe = (np.arange(N) - (N - 1) / 2) * tab["pitch"]
        notes.append(f"element positions from the CUBDL probe pitch {tab['pitch'] * 1e3:.3f} mm")
    else:
        raise ValueError("no element_positions in the file and no known probe pitch for this institution")

    # ---- angles (rad): (T,), (T,2) azimuth/elevation or (2,T); degrees if large
    ang = np.squeeze(ang)
    if ang.ndim == 2: ang = ang[:, 0] if ang.shape[0] == T else ang[0]
    ang = np.atleast_1d(ang).ravel()[:T]
    if np.abs(ang).max() > np.pi / 2: ang = np.deg2rad(ang)
    missing = T > 1 and (ang.size < T or np.allclose(ang, 0))
    if missing:                                                        # the file does not store the angles
        span = max_angle_deg or (tab or {}).get("max_angle", 16.0)
        ang = np.deg2rad(np.linspace(-span, span, T))
        notes.append(f"no angles in the file: angles {-span:g}..{span:g} deg (CUBDL loader for INS)")
    t0 = np.full(T, st[0]) if st.size == 1 else st[:T]
    if T > 2 and not missing and not (np.all(np.diff(ang) >= 0) or np.all(np.diff(ang) <= 0)):
        o = np.argsort(ang); data, ang, t0 = data[o], ang[o], t0[o]    # angle order for the phase metric
        notes.append("transmits sorted by angle")

    # ---- beamformed image of the file (for the check only)
    if bf is not None:
        b = np.asarray(bf); bc = _complex(b)
        if bc is not None: b = np.abs(bc)
        b = b.astype(np.float64)
        if pix is None and bf_fs and b.ndim == 2 and N in b.shape:     # UFL: lines at the elements, time samples
            if b.shape[0] != N: b = b.T
            tz = (bf_t0 or 0.0) + np.arange(b.shape[1]) / bf_fs
            X, Zt = np.meshgrid(xe, tz * c_file / 2, indexing="ij")
            pix = np.stack([X.ravel(), 0 * X.ravel(), Zt.ravel()], axis=1)
            b = np.abs(hilbert(b, axis=1)); notes.append("beamformed_data: RF lines at the elements, envelope used")
        elif np.nanmax(b) <= 0 and np.nanmin(b) < -1:                  # stored in dB
            b = 10.0 ** (np.where(np.isfinite(b), b, -300) / 20.0); notes.append("beamformed_data in dB")
        b = np.abs(b).reshape(-1)
        if pix is None or b.size != pix.shape[0]:
            notes.append("beamformed_data has no matching pixel positions: not used"); b = None
        bf = b
    if bf is None: pix = None

    conv = None
    if tab:
        t_off = -10.0 / fdem if tab["t_off"] == "-10/fc" else float(tab["t_off"])
        conv = dict(tx_ref=tab["tx_ref"], t0_sign=tab["t0_sign"], t_off=t_off)
    d = PWData(path.stem, data, is_iq, fs, fdem, c_file, ang, t0, xe, pix, bf, notes, missing, inst, conv,
               tab["zlims"] if tab else None)
    if verbose: describe(path, d)
    return d


def describe(path, d=None):
    """Without d: print the keys, shapes and small values of the file. With d: print the loaded layout."""
    if d is None:
        with h5py.File(path, "r") as f:
            print(f"== {path}")
            def show(k, v):
                if not isinstance(v, h5py.Dataset) or k.startswith("metadata"): return
                s = f"{k:34s} shape={v.shape} dtype={v.dtype}"
                if v.size <= 4:
                    x = v[()]; s += f"  value={x.decode()[:50] if isinstance(x, bytes) else np.asarray(x).ravel()}"
                elif v.dtype.kind in "fiu" and v.size < 2e6:
                    x = np.asarray(v[()], np.float64); s += f"  min={x.min():.4g} max={x.max():.4g}"
                print(s)
            f.visititems(show)
    else:
        for n in d.notes: print(f"   note: {n}")
        c = d.conv
        print(f"-> data (T,S,N)={d.data.shape} iq={d.is_iq} fs={d.fs/1e6:.3f} MHz fdemod={d.fdemod/1e6:.3f} MHz "
              f"c_file={d.c_file} angles=[{np.rad2deg(d.angles.min()):.1f}, {np.rad2deg(d.angles.max()):.1f}] deg "
              f"t0=[{d.t0.min()*1e6:.3f}, {d.t0.max()*1e6:.3f}] us aperture={1e3*(d.xe.max()-d.xe.min()):.2f} mm "
              f"bf={'yes' if d.bf is not None else 'no'}")
        if c: print(f"-> convention ({d.inst}): tx_ref={c['tx_ref']} t0_sign={c['t0_sign']:+.0f} "
                    f"offset={c['t_off']*1e6:+.2f} us, depth {d.zlims[0]*1e3:.0f}-{d.zlims[1]*1e3:.0f} mm")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        load(p, verbose=True)
