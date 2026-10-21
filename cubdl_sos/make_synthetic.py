"""
Synthetic test file in the CUBDL layout (no license needed): speckle and point targets in a medium
with a known speed of sound, imaged with steered plane waves. Use it to test the installation:

    python make_synthetic.py --out synthetic/SYN001.hdf5 --c 1500
    python run_sos_comparison.py --data synthetic --out synthetic_results/results.csv

The file also contains beamformed_data from an independent NumPy delay-and-sum at 1540 m/s, so that
the convention check has a reference.
"""
import argparse
from pathlib import Path
import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("synthetic/SYN001.hdf5"))
    ap.add_argument("--c", type=float, default=1500.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--angles", type=int, default=11); ap.add_argument("--max-angle", type=float, default=10.0)
    # variants of the CUBDL files (for tests of the loader and of the convention search)
    ap.add_argument("--uncentered", action="store_true", help="element x from 0 to the aperture width")
    ap.add_argument("--bf-db", action="store_true", help="store beamformed_data in dB")
    ap.add_argument("--no-bf", action="store_true", help="do not store beamformed_data")
    ap.add_argument("--no-counts", action="store_true", help="do not store transmit_count / element_count")
    ap.add_argument("--zero-angles", action="store_true", help="store transmit_direction as zeros")
    ap.add_argument("--hidden-offset", type=float, default=0.0, help="true time offset (s) not in start_time")
    ap.add_argument("--tx-ref", choices=["origin", "edge"], default="origin", help="transmit delay reference")
    ap.add_argument("--t-start", type=float, default=1e-6, help="time of the first sample (s)")
    ap.add_argument("--pitch", type=float, default=0.3e-3)
    ap.add_argument("--format", choices=["generic", "myo", "ufl"], default="generic",
                    help="key layout: generic (OSL/EUT/INS type), myo (no positions), ufl (no positions, deg)")
    # two-layer medium (fat over muscle): --c is the speed below the interface
    ap.add_argument("--layer-h", type=float, default=0.0, help="depth (m) of a horizontal interface; 0: homogeneous")
    ap.add_argument("--c1", type=float, default=1450.0, help="speed (m/s) above the interface")
    ap.add_argument("--interface-reflector", action="store_true", help="bright scatterers along the interface")
    ap.add_argument("--slowness", choices=["angle", "delays"], default="angle",
                    help="angle: p = sin(theta)/c_top; delays: p = sin(theta)/1540 (fixed by the transmit delays)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    N, pitch, fc, fs, S, t_start = 128, a.pitch, 5e6, 20e6, 2048, a.t_start
    xe = (np.arange(N) - (N - 1) / 2) * pitch
    ang = np.deg2rad(np.linspace(-a.max_angle, a.max_angle, a.angles))
    K = 6000
    xs = rng.uniform(-16e-3, 16e-3, K); zs = rng.uniform(5e-3, 50e-3, K); amp = rng.normal(size=K)
    cyst = (xs - 4e-3) ** 2 + (zs - 30e-3) ** 2 < (4e-3) ** 2; amp[cyst] = 0
    pins = np.array([[-6e-3, 20e-3], [0.0, 25e-3], [-6e-3, 38e-3], [6e-3, 15e-3]])
    xs = np.r_[xs, pins[:, 0]]; zs = np.r_[zs, pins[:, 1]]; amp = np.r_[amp, np.full(len(pins), 25.0)]
    h = a.layer_h; c1 = a.c1 if h > 0 else a.c; c2 = a.c
    if h > 0 and a.interface_reflector:                                  # specular-like bright line (fascia)
        xi = rng.uniform(-18e-3, 18e-3, 400); zi = h + rng.normal(0, 0.05e-3, 400)
        xs = np.r_[xs, xi]; zs = np.r_[zs, zi]; amp = np.r_[amp, rng.normal(4.0, 0.5, 400)]

    def times(th, e):
        """Two-way time to each scatterer: plane wave (slowness p conserved) + refracted receive ray
        (crossing point by Newton steps on the travel time, independent NumPy code)."""
        p = np.sin(th) / (c1 if a.slowness == "angle" else 1540.0)
        q1 = np.sqrt(max(1 / c1 ** 2 - p ** 2, 1e-14)); q2 = np.sqrt(max(1 / c2 ** 2 - p ** 2, 1e-14))
        below = (zs > h) & (h > 0)
        ttx = p * (xs - xref(th)) + np.where(below, h * q1 + (zs - h) * q2, zs * (q1 if h > 0 else q2))
        D = xs - xe[e]; trx = np.sqrt(D ** 2 + zs ** 2) / c1
        if h > 0:
            Zb = np.maximum(zs - h, 1e-6); u = D * h / np.maximum(zs, h + 1e-6)
            for _ in range(10):
                A = np.sqrt(u ** 2 + h ** 2); B = np.sqrt((D - u) ** 2 + Zb ** 2)
                u = u - (u / (c1 * A) - (D - u) / (c2 * B)) / (h ** 2 / (c1 * A ** 3) + Zb ** 2 / (c2 * B ** 3))
                u = np.clip(u, np.minimum(D, 0.0), np.maximum(D, 0.0))      # crossing between element and scatterer
            trx = np.where(below, np.sqrt(u ** 2 + h ** 2) / c1 + np.sqrt((D - u) ** 2 + Zb ** 2) / c2, trx)
        return ttx + trx
    t = np.arange(S) / fs + t_start
    tp = np.arange(-24, 25) / fs
    pulse = np.cos(2 * np.pi * fc * tp) * np.exp(-(tp * fc / 1.2) ** 2)
    rf = np.zeros((len(ang), N, S), np.float32)
    xref = lambda th: (xe.min() if np.sin(th) >= 0 else xe.max()) if a.tx_ref == "edge" else 0.0
    for i, th in enumerate(ang):
        for e in range(N):
            tau = times(th, e)
            idx = (tau + a.hidden_offset - t_start) * fs; lo = np.floor(idx).astype(int); fr = idx - lo
            ok = (lo >= 0) & (lo < S - 1)
            imp = np.zeros(S)
            np.add.at(imp, lo[ok], amp[ok] * (1 - fr[ok])); np.add.at(imp, lo[ok] + 1, amp[ok] * fr[ok])
            rf[i, e] = np.convolve(imp, pulse, mode="same")
    rf += 0.02 * rf.std() * rng.normal(size=rf.shape).astype(np.float32)

    # pixel grid and an independent NumPy DAS at 1540 m/s (analytic signal, F-number 1)
    px = np.linspace(-15e-3, 15e-3, 121); pz = np.linspace(5e-3, 48e-3, 172)
    X, Z = np.meshgrid(px, pz, indexing="ij")                              # (nx, nz)
    from scipy.signal import hilbert
    an = hilbert(rf, axis=2)
    img = np.zeros(X.shape, complex); c0 = 1540.0
    for i, th in enumerate(ang):
        ttx = ((X - xref(th)) * np.sin(th) + Z * np.cos(th)) / c0
        for e in range(N):
            tau = ttx + np.sqrt((X - xe[e]) ** 2 + Z ** 2) / c0
            idx = (tau + a.hidden_offset - t_start) * fs; lo = np.clip(np.floor(idx).astype(int), 0, S - 2); fr = idx - np.floor(idx)
            v = an[i, e, lo] * (1 - fr) + an[i, e, lo + 1] * fr
            img += np.where(np.abs(X - xe[e]) <= Z / 2, v, 0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as f:
        if a.format == "myo":                                          # Mayo layout (int16, no geometry keys)
            f["channel_data"] = np.round(rf / np.abs(rf).max() * 30000).astype(np.int16)
            f["angles"] = ang.astype(np.float32); f["modulation_frequency"] = np.array([fc], np.float32)
            f["sampling_frequency"] = np.array([fs], np.float32); f["sound_speed"] = np.array([c0], np.float32)
        elif a.format == "ufl":                                        # UNIFI layout (angles in degrees)
            f["channel_data"] = rf; f["angles"] = np.rad2deg(ang).astype(np.float32)
            f["modulation_frequency"] = np.array([fc], np.float32)
            f["channel_data_sampling_frequency"] = np.array([fs], np.float32)
            f["channel_data_t0"] = np.array([t_start], np.float32); f["sound_speed"] = np.array([c0], np.float32)
        else:
            f["channel_data"] = rf                                     # (T, N, S)
            if not a.no_counts:
                f["element_count"] = np.array([N]); f["transmit_count"] = np.array([len(ang)])
            xs_file = xe - xe.min() if a.uncentered else xe
            f["element_positions"] = np.stack([xs_file, np.zeros(N), np.zeros(N)])  # (3, N)
            f["modulation_frequency"] = np.array([fc]); f["sampling_frequency"] = np.array([fs])
            f["sound_speed"] = np.array([c0]); f["start_time"] = np.full(len(ang), t_start)
            f["transmit_direction"] = np.stack([np.zeros_like(ang) if a.zero_angles else ang, np.zeros_like(ang)], axis=1)
            f["pixel_positions"] = np.stack([X, np.zeros_like(X), Z], axis=-1)  # (nx, nz, 3)
            b = np.abs(img)
            if a.bf_db: b = 20 * np.log10(b / b.max() + 1e-30)
            if not a.no_bf: f["beamformed_data"] = b.astype(np.float32)
    print(f"wrote {a.out}  (true c = {a.c} m/s" + (f", layer h = {h * 1e3:.1f} mm, c1 = {c1} m/s" if h > 0 else "")
          + f", slowness {a.slowness})")


if __name__ == "__main__":
    main()
