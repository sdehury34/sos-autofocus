"""
Speed-of-sound estimation on CUBDL plane-wave data: comparison of methods.

Methods (one row each in the output CSV):
    nominal          1540 m/s (the sound speed that all institutions submitted)
    grid_cf          grid search, maximum of the mean coherence factor
    grid_bright      grid search, maximum of the mean speckle brightness
    grid_sharp       grid search, maximum of the image sharpness (normalized 4th moment)
    grid_phase       grid search, maximum of the inter-angle phase coherence
    gd_cf            gradient descent on c, coherence-factor objective   (this work)
    gd_bright        gradient descent on c, brightness objective          (this work)
    gd_phase         gradient descent on c, inter-angle phase objective   (phase-error loss, as in
                     differentiable-beamforming autofocus; adapted to compounded plane waves)
    gds_cf, gds_bright, gds_phase
                     the same three objectives, gradient descent on the objective smoothed with a
                     Gaussian in c (width 10 -> 0.5 m/s; graduated optimization)

Usage
    python run_sos_comparison.py --inspect FILE.hdf5               # print the file layout
    python run_sos_comparison.py --check FILE.hdf5                 # beamformer check against beamformed_data
    python run_sos_comparison.py --data DIR --out results.csv      # all files in DIR (resumes)
"""
import argparse, csv, json, time, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cubdl_io
from das import Beamformer, METRICS

C_LO, C_HI = 1400.0, 1680.0
FIELDS = ["file", "institution", "method", "c_est", "time_s", "n_evals", "angles_order", "tx_ref", "t0_sign",
          "t_off_us", "phase_ref", "score_type", "score", "n_angles", "n_pixels", "ref_sos", "ref_alt",
          "c1_est", "h_mm", "h_img_mm", "h_prom_db", "slowness"]


def estimation_grid(d, nx_max, nz_max, zmin=None, zmax=None):
    """Uniform grid for the metrics: x over the aperture, z over the CUBDL depth range of the institution
    (or the pixel range of the file, or 5 mm to 90 % of the recorded depth)."""
    x0, x1 = float(d.xe.min()), float(d.xe.max())
    if d.zlims: z0, z1 = d.zlims
    elif d.pix is not None: z0, z1 = float(d.pix[:, 2].min()), float(d.pix[:, 2].max())
    else: z0, z1 = 5e-3, 0.9 * d.data.shape[1] / d.fs * d.c_file / 2
    if zmin is not None: z0 = max(z0, zmin)
    if zmax is not None: z1 = min(z1, zmax)
    return np.linspace(x0, x1, nx_max), np.linspace(z0, z1, nz_max)


def pick_angles(d, n_max):
    T = d.angles.size
    if n_max <= 0 or T <= n_max: return d
    keep = np.unique(np.round(np.linspace(0, T - 1, n_max)).astype(int))
    d.data, d.angles, d.t0 = d.data[keep], d.angles[keep], d.t0[keep]
    return d


def _ncc(a, b):
    a = 20 * np.log10(a / a.max() + 1e-6); b = 20 * np.log10(b / b.max() + 1e-6)
    a = np.clip(a, -60, 0); b = np.clip(b, -60, 0)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def find_convention(d, args):
    """Select the delay convention.
    With beamformed_data in the file: the convention (and a fixed time offset) that best reproduces
    it at the sound speed of the file (NCC of the log images).
    Without beamformed_data: the convention with the highest mean coherence factor over a coarse range
    of speeds of sound (max over c), so that a wrong convention cannot trade against a speed-of-sound
    error. No time-offset search in this case."""
    use_bf = d.bf is not None
    if args.convention == "cubdl" and d.conv is not None:
        best = dict(angles=d.angles, angles_order="cubdl", phase_ref=None, score=float("nan"), score_type="none",
                    **d.conv)
        if use_bf:                                                      # information only
            sel = np.arange(d.pix.shape[0])
            if sel.size > args.check_pixels: sel = np.random.default_rng(0).choice(sel, args.check_pixels, replace=False)
            bf = Beamformer(d, np.array([0.0]), np.array([0.02]), args.fnum, chunk=args.chunk)
            bf.set_pixels(d.pix[sel, 0], d.pix[sel, 2], args.chunk)
            bf.set_geometry(t0_sign=d.conv["t0_sign"], t_off=d.conv["t_off"], tx_ref=d.conv["tx_ref"])
            best.update(score=_ncc(bf.envelope(d.c_file), d.bf[sel]), score_type="ncc")
        print(f"  convention of {d.inst}: tx_ref={best['tx_ref']} t0_sign={best['t0_sign']:+.0f} "
              f"offset={best['t_off']*1e6:+.2f} us" + (f" | NCC with beamformed_data = {best['score']:.3f}"
                                                      if use_bf else " | no usable beamformed_data in the file"))
        return best
    if use_bf:
        sel = np.arange(d.pix.shape[0])
        if sel.size > args.check_pixels: sel = np.random.default_rng(0).choice(sel, args.check_pixels, replace=False)
        xp, zp = d.pix[sel, 0], d.pix[sel, 2]
    else:
        xs, zs = estimation_grid(d, 64, 80, args.zmin); X, Z = np.meshgrid(xs, zs); xp, zp = X.ravel(), Z.ravel()
        sel = None
    angle_sets = [("file", d.angles)]
    if d.angles_missing: angle_sets += [("reversed", d.angles[::-1].copy())]
    signs = (1.0,) if np.allclose(d.t0, 0) else (1.0, -1.0)
    phases = ("abs", "rel", "z") if d.is_iq else ("rel",)
    best = None
    for ph in phases:
        bf = Beamformer(d, np.array([0.0]), np.array([0.02]), args.fnum, chunk=args.chunk, phase_ref=ph)
        bf.set_pixels(xp, zp, args.chunk)
        cgrid = np.arange(C_LO, C_HI + 1e-6, 10.0)
        score = ((lambda: _ncc(bf.envelope(d.c_file), d.bf[sel])) if use_bf else
                 (lambda: max(bf.metrics(c)["cf"] for c in cgrid)))
        for an, ang in angle_sets:
            for ref in ("origin", "edge"):
                for sg in signs:
                    offs = np.arange(-args.max_offset, args.max_offset + 1e-12, 0.2e-6) if use_bf else [0.0]
                    res = []
                    for o in offs:
                        bf.set_geometry(angles=ang, t0_sign=sg, t_off=o, tx_ref=ref); res.append(score())
                    o = offs[int(np.nanargmax(res))]
                    if use_bf:                                          # fine offset search, 0.02 us step
                        fine = np.arange(o - 0.2e-6, o + 0.2e-6 + 1e-12, 0.02e-6); rf = []
                        for o2 in fine:
                            bf.set_geometry(angles=ang, t0_sign=sg, t_off=o2, tx_ref=ref); rf.append(score())
                        o, r = fine[int(np.nanargmax(rf))], float(np.nanmax(rf))
                    else:
                        r = float(np.nanmax(res))
                    print(f"  convention angles={an:8s} tx_ref={ref:6s} t0_sign={sg:+.0f} phase_ref={ph:3s} "
                          f"offset={o*1e6:+.2f} us: {'NCC' if use_bf else 'max CF'} = {r:.3f}")
                    if best is None or r > best["score"]:
                        best = dict(angles=ang, angles_order=an, tx_ref=ref, t0_sign=sg, t_off=float(o),
                                    phase_ref=ph, score=r, score_type="ncc" if use_bf else "cf")
    if d.angles_missing and not args.assume_max_angle:
        # the file does not store the angles: search the angle span with the selected convention,
        # then select the transmit reference and the start-time sign again at the best span
        T = d.angles.size; sgn = 1.0 if best["angles_order"] == "file" else -1.0
        bf = Beamformer(d, np.array([0.0]), np.array([0.02]), args.fnum, chunk=args.chunk, phase_ref=best["phase_ref"])
        bf.set_pixels(xp, zp, args.chunk); cgrid = np.arange(C_LO, C_HI + 1e-6, 10.0)
        spans = np.arange(8.0, 20.0 + 1e-9, 1.0); sc = []
        for sp in spans:
            bf.set_geometry(angles=sgn * np.deg2rad(np.linspace(-sp, sp, T)), t0_sign=best["t0_sign"], t_off=0.0,
                            tx_ref=best["tx_ref"])
            sc.append(max(bf.metrics(c)["cf"] for c in cgrid))
        sp = float(spans[int(np.argmax(sc))])
        print("  angle span search (max CF): " + " ".join(f"{a:.0f}:{v:.3f}" for a, v in zip(spans, sc)))
        ang = sgn * np.deg2rad(np.linspace(-sp, sp, T)); best.update(angles=ang, score=float(max(sc)))
        for ref in ("origin", "edge"):
            for sg in signs:
                bf.set_geometry(angles=ang, t0_sign=sg, t_off=0.0, tx_ref=ref)
                r = max(bf.metrics(c)["cf"] for c in cgrid)
                print(f"  span {sp:.0f} deg: tx_ref={ref:6s} t0_sign={sg:+.0f}: max CF = {r:.3f}")
                if r > best["score"]: best.update(tx_ref=ref, t0_sign=sg, score=r)
        best["angles_order"] = f"{best['angles_order']}+-{sp:.0f}deg"
    print(f"  selected: angles={best['angles_order']} tx_ref={best['tx_ref']} t0_sign={best['t0_sign']:+.0f} "
          f"phase_ref={best['phase_ref']} offset={best['t_off']*1e6:+.2f} us ({best['score_type']} {best['score']:.3f})")
    if use_bf and best["score"] < 0.6:
        print(f"  WARNING: best NCC {best['score']:.2f} < 0.6. The delay convention is not certain; look at the check image.")
    if not use_bf:
        print("  note: no usable beamformed_data; the convention was selected by the coherence factor. Check the image.")
    return best


def make_beamformer(d, conv, xs, zs, args):
    bf = Beamformer(d, xs, zs, args.fnum, chunk=args.chunk, patch=args.patch, pair_sep=args.pair_sep,
                    phase_ref=conv["phase_ref"], slowness=args.slowness)
    bf.set_geometry(angles=conv["angles"], t0_sign=conv["t0_sign"], t_off=conv["t_off"], tx_ref=conv["tx_ref"])
    return bf


def grid_search(bf, step_coarse, step_fine):
    t = time.time(); vals = {}
    for c in np.arange(C_LO, C_HI + 1e-6, step_coarse):
        vals[float(c)] = bf.metrics(c)
    best = {m: max(vals, key=lambda c: vals[c][m]) for m in METRICS}
    for m in METRICS:                                                   # fine search around each optimum
        for c in np.arange(best[m] - step_coarse, best[m] + step_coarse + 1e-6, step_fine):
            if C_LO <= c <= C_HI and float(c) not in vals: vals[float(c)] = bf.metrics(c)
    best = {m: max(vals, key=lambda c: vals[c][m]) for m in METRICS}
    return best, vals, time.time() - t


GH_X = np.array([-1.6506801, -0.5246476, 0.5246476, 1.6506801])        # 4-point Gauss-Hermite nodes
GH_W = np.array([0.0813128, 0.8049141, 0.8049141, 0.0813128]) / np.sqrt(np.pi)


def gradient_descent(bf, metric, c0=1540.0, iters=100, lr=3.0, decay=0.97, tol=0.02, patience=8,
                     smooth=False, sigma0=10.0, sigma_end=0.5, sigma_decay=0.93):
    """Adam on c (m/s) that maximizes the metric. The gradient is normalized by the first value, so
    that the step size does not depend on the scale of the data.
    smooth=True: gradient of the metric smoothed with a Gaussian in c (Gauss-Hermite average of the
    exact gradients at c + delta); the width decreases from sigma0 to sigma_end (graduated optimization),
    so small local maxima of the landscape do not stop the descent."""
    t = time.time(); c = c0; m = v = 0.0; b1, b2 = 0.9, 0.99; traj = [c]; still = 0; scale = None; n = 0
    for k in range(1, iters + 1):
        if smooth:
            sig = max(sigma_end, sigma0 * sigma_decay ** (k - 1))
            vg = [bf.value_and_grad(float(np.clip(c + np.sqrt(2) * sig * x, C_LO, C_HI)), metric) for x in GH_X]
            val = float(np.dot(GH_W, [q[0] for q in vg])); g = float(np.dot(GH_W, [q[1] for q in vg])); n += 4
        else:
            val, g = bf.value_and_grad(c, metric); n += 1
        if scale is None: scale = abs(val) + 1e-30
        g = -g / scale * 1e3                                            # minimize -metric; units: per km/s
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        step = lr * decay ** k * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + 1e-12)
        c_new = float(np.clip(c - step, C_LO, C_HI)); still = still + 1 if abs(c_new - c) < tol else 0
        c = c_new; traj.append(c)
        if still >= patience and (not smooth or sig <= sigma_end): break
    return c, n, time.time() - t, traj


C1_LIM, C2_LIM, H_LIM = (1380.0, 1580.0), (1440.0, 1700.0), (2e-3, 30e-3)


def detect_interface(d, conv, args, zrange=(3e-3, 25e-3)):
    """Depth of the strongest horizontal bright line (fat-muscle fascia) in the homogeneous image at the
    sound speed of the file: lateral mean of the log envelope, minus a 3 mm running median (baseline)."""
    from scipy.ndimage import median_filter, uniform_filter1d
    xs = np.linspace(d.xe.min() * 0.8, d.xe.max() * 0.8, 96); zs = np.arange(2e-3, zrange[1] + 3e-3, 0.1e-3)
    bf = make_beamformer(d, conv, xs, zs, args)
    env = bf.envelope(d.c_file).reshape(zs.size, xs.size)
    prof = uniform_filter1d(np.mean(20 * np.log10(env / env.max() + 1e-6), axis=1), 3)
    rel = prof - median_filter(prof, size=31, mode="nearest")
    sel = (zs >= zrange[0]) & (zs <= zrange[1]); i = int(np.argmax(np.where(sel, rel, -np.inf)))
    return float(zs[i]), float(rel[i])


def adam(vg, x0, lr, lo, hi, iters=100, decay=0.97, tol=0.02, patience=8):
    """Adam that maximizes a metric over a parameter vector; vg(x) -> (value, gradient). The gradient is
    normalized by the first value. lr, tol: steps in the units of each parameter."""
    x = np.array(x0, float); lr = np.asarray(lr, float); tol = np.asarray(tol, float)
    m = np.zeros_like(x); v = np.zeros_like(x); b1, b2 = 0.9, 0.99; scale = None; still = 0; traj = [x.tolist()]
    for k in range(1, iters + 1):
        val, g = vg(x); g = -np.asarray(g, float)
        if scale is None: scale = abs(val) + 1e-30
        g = g / scale * 1e3
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        step = lr * decay ** k * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + 1e-12)
        xn = np.clip(x - step, lo, hi); still = still + 1 if np.all(np.abs(xn - x) < tol) else 0
        x = xn; traj.append(x.tolist())
        if still >= patience: break
    return x, k, traj


def layer_methods(d, conv, bf, args, row, init):
    """Two-layer estimates: (c2, c1) by gradient descent with h from the image (l2_*), (c2, c1, h) by
    gradient descent (l3_*, h in mm), and a coarse 2-D grid search with h from the image (l2grid_*)."""
    out, trajs = [], {}
    t = time.time(); h_img, prom = detect_interface(d, conv, args); t_det = time.time() - t
    bf.h = h_img
    base = dict(row, h_img_mm=round(h_img * 1e3, 2), h_prom_db=round(prom, 2))
    print(f"  interface from the image: h = {h_img * 1e3:.1f} mm (line {prom:.1f} dB above the baseline)")

    def vg3(x, m):                                                       # x = (c2, c1, h_mm)
        val, g = bf.value_and_grad2(x[0], x[1], m, h=x[2] * 1e-3, with_h=True)
        return val, (g[0], g[1], g[2] * 1e-3)

    # The image at the file sound speed fixes the echo time of the interface, 2 h / c1 = 2 h_img / c_file,
    # so h = h_img * c1 / c_file (h changes with c1; chain rule dh/dc1 = h_img / c_file).
    k_h = h_img / d.c_file

    def vg2(x, m):                                                       # x = (c2, c1)
        val, g = bf.value_and_grad2(x[0], x[1], m, h=k_h * x[1], with_h=True)
        return val, (g[0], g[1] + g[2] * k_h)

    for m in ("cf", "bright", "phase"):
        c2_0 = float(np.clip(init.get(m, 1540.0), *C2_LIM))
        t = time.time()
        x, k, tr = adam(lambda q: vg2(q, m), [c2_0, 1460.0], [3.0, 3.0],
                        [C2_LIM[0], C1_LIM[0]], [C2_LIM[1], C1_LIM[1]], iters=args.iters, tol=[0.02, 0.02])
        out.append(dict(base, method=f"l2_{m}", c_est=round(x[0], 2), c1_est=round(x[1], 2),
                        h_mm=round(k_h * x[1] * 1e3, 2), time_s=round(time.time() - t + t_det, 2), n_evals=3 * k))
        trajs[f"l2_{m}"] = tr
        if args.layer_free_h:
            t = time.time()
            x3, k, tr = adam(lambda q: vg3(q, m), [x[0], x[1], k_h * x[1] * 1e3], [3.0, 3.0, 0.3],
                             [C2_LIM[0], C1_LIM[0], H_LIM[0] * 1e3], [C2_LIM[1], C1_LIM[1], H_LIM[1] * 1e3],
                             iters=args.iters, tol=[0.02, 0.02, 0.005])
            out.append(dict(base, method=f"l3_{m}", c_est=round(x3[0], 2), c1_est=round(x3[1], 2),
                            h_mm=round(x3[2], 2), time_s=round(time.time() - t, 2), n_evals=3 * k))
            trajs[f"l3_{m}"] = tr
    if args.layer_grid:                                                 # coarse 2-D grid, h from the image
        t = time.time(); vals = {}
        for c1 in np.arange(1400.0, 1560.0 + 1e-6, 20.0):
            for c2 in np.arange(1460.0, 1660.0 + 1e-6, 10.0):
                vals[(c2, c1)] = bf.metrics(c2, c1=c1, h=k_h * c1)
        for m in ("cf", "bright", "phase"):
            c2, c1 = max(vals, key=lambda kk: vals[kk][m])
            out.append(dict(base, method=f"l2grid_{m}", c_est=c2, c1_est=c1, h_mm=round(k_h * c1 * 1e3, 2),
                            time_s=round(time.time() - t, 2), n_evals=len(vals)))
    return out, trajs


def run_file(path, args, writer, done):
    name = Path(path).stem; inst = "ZEA" if cubdl_io.is_zea(path) else name[:3]
    if name in done: print(f"{name}: already in the results, skip"); return
    t = time.time(); d = pick_angles(cubdl_io.load(path, max_angle_deg=args.assume_max_angle or None, frame=args.frame), args.max_angles)
    print(f"{name}: T={d.angles.size} N={d.xe.size} S={d.data.shape[1]} iq={d.is_iq} load {time.time()-t:.1f}s")
    conv = find_convention(d, args)
    xs, zs = estimation_grid(d, args.nx, args.nz, args.zmin, args.zmax)
    bf = make_beamformer(d, conv, xs, zs, args)
    row = dict(file=name, institution=inst, angles_order=conv["angles_order"], tx_ref=conv["tx_ref"],
               t0_sign=conv["t0_sign"], t_off_us=round(conv["t_off"] * 1e6, 3), phase_ref=bf.phase_ref,
               score_type=conv["score_type"], score=round(conv["score"], 4),
               n_angles=d.angles.size, n_pixels=xs.size * zs.size)
    refs = getattr(d, "refs", {}) or {}                                 # reference values stored in the file (zea)
    row["ref_sos"] = refs.get("through_tx_speed_of_sound", refs.get("estimated_global_speed_of_sound", ""))
    row["ref_alt"] = refs.get("estimated_global_speed_of_sound", "") if "through_tx_speed_of_sound" in refs else ""
    out = [dict(row, method="nominal", c_est=1540.0, time_s=0.0, n_evals=0)]
    best, vals, tg = grid_search(bf, args.step, 1.0)
    for m in METRICS:
        out.append(dict(row, method=f"grid_{m}", c_est=best[m], time_s=round(tg, 2), n_evals=len(vals)))
    trajs = {}
    for pre, smooth in (("gd", False), ("gds", True)):
        for m in ("cf", "bright", "phase"):
            c, k, tm, traj = gradient_descent(bf, m, iters=args.iters, smooth=smooth)
            trajs[f"{pre}_{m}"] = traj
            out.append(dict(row, method=f"{pre}_{m}", c_est=round(c, 2), time_s=round(tm, 2), n_evals=k))
    if args.layers:
        init = {o["method"][4:]: o["c_est"] for o in out if o["method"].startswith("gds_")}
        lo, ltr = layer_methods(d, conv, bf, args, row, init); out += lo; trajs.update(ltr)
    for o in out: o.setdefault("slowness", args.slowness); writer.writerow(o)
    args.fh.flush()
    curves = {m: [[c, vals[c][m]] for c in sorted(vals)] for m in METRICS}
    (args.out.parent / "curves").mkdir(exist_ok=True)
    (args.out.parent / "curves" / f"{name}.json").write_text(json.dumps({"curves": curves, "traj": trajs}))
    print("  " + "  ".join(f"{o['method']}={o['c_est']:.0f}" for o in out) + f"   ({time.time()-t:.0f}s)")


def check(path, args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = pick_angles(cubdl_io.load(path, verbose=True, max_angle_deg=args.assume_max_angle or None, frame=args.frame), args.max_angles)
    conv = find_convention(d, args)
    xs, zs = estimation_grid(d, 300, 400)
    bf = make_beamformer(d, conv, xs, zs, args)
    env = bf.envelope(d.c_file).reshape(bf.shape)
    ext = [1e3 * xs[0], 1e3 * xs[-1], 1e3 * zs[-1], 1e3 * zs[0]]
    fig, ax = plt.subplots(1, 2 if d.bf is not None else 1, figsize=(9, 4.5), squeeze=False)
    ax[0, 0].imshow(20 * np.log10(env / env.max() + 1e-6), cmap="gray", vmin=-60, vmax=0, extent=ext)
    ax[0, 0].set_title(f"this beamformer, c = {d.c_file:.0f} m/s\nangles {conv['angles_order']}, {conv['tx_ref']}, "
                       f"t0 sign {conv['t0_sign']:+.0f}, offset {conv['t_off']*1e6:+.2f} us", fontsize=8)
    if d.bf is not None:
        b = d.bf; px = d.pix
        ax[0, 1].scatter(1e3 * px[:, 0], 1e3 * px[:, 2], c=np.clip(20 * np.log10(b / b.max() + 1e-6), -60, 0),
                         s=1, cmap="gray", marker="s"); ax[0, 1].invert_yaxis(); ax[0, 1].set_aspect("equal")
        ax[0, 1].set_title(f"beamformed_data in the file (NCC {conv['score']:.2f})", fontsize=8)
    for a in ax.ravel(): a.set_xlabel("x (mm)"); a.set_ylabel("z (mm)")
    fig.tight_layout(); out = Path(f"check_{d.name}.png"); fig.savefig(out, dpi=130)
    m = bf.metrics(d.c_file)
    print(f"metrics at {d.c_file:.0f} m/s: " + ", ".join(f"{k}={v:.4g}" for k, v in m.items()))
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="folder with the CUBDL .hdf5 files")
    ap.add_argument("--files", nargs="*", default=None, help="file names (stems) to run, e.g. OSL002 MYO001")
    ap.add_argument("--out", type=Path, default=Path("results/results.csv"))
    ap.add_argument("--inspect", type=Path); ap.add_argument("--check", type=Path)
    ap.add_argument("--max-angles", type=int, default=0, help="0 = all angles")
    ap.add_argument("--nx", type=int, default=128); ap.add_argument("--nz", type=int, default=160)
    ap.add_argument("--zmin", type=float, default=None, help="minimum depth (m); default: CUBDL depth range")
    ap.add_argument("--zmax", type=float, default=None, help="maximum depth (m) for the metrics")
    ap.add_argument("--slowness", choices=["angle", "delays"], default="angle",
                    help="plane-wave slowness: angle (p = sin/c, CUBDL) or delays (p = sin/c_file, physical)")
    ap.add_argument("--layers", action="store_true", help="also two-layer estimates (fat over muscle)")
    ap.add_argument("--layer-free-h", action="store_true", help="also estimate the interface depth h (l3_*)")
    ap.add_argument("--layer-grid", action="store_true", help="also a coarse 2-D grid search (l2grid_*)")
    ap.add_argument("--frame", type=int, default=0, help="frame index for multi-frame (zea) files")
    ap.add_argument("--convention", choices=["cubdl", "auto"], default="cubdl",
                    help="cubdl: conventions of the official CUBDL loader; auto: search (also for unknown files)")
    ap.add_argument("--fnum", type=float, default=1.0, help="receive F-number")
    ap.add_argument("--step", type=float, default=5.0, help="coarse grid step (m/s); the fine step is 1 m/s")
    ap.add_argument("--iters", type=int, default=100, help="maximum gradient-descent iterations")
    ap.add_argument("--patch", type=int, default=16, help="patch length (pixels along x) of the phase metric")
    ap.add_argument("--pair-sep", type=int, default=0, help="angle separation of the phase pairs (0: T//5)")
    ap.add_argument("--check-pixels", type=int, default=10000)
    ap.add_argument("--max-offset", type=float, default=4e-6, help="range (s) of the time-offset search")
    ap.add_argument("--assume-max-angle", type=float, default=0.0,
                    help="angle span (deg) for files with no angles; 0: search 8..20 deg")
    ap.add_argument("--chunk", type=int, default=0, help="pixels for each chunk (0: 4096 on CPU, 16384 on GPU)")
    args = ap.parse_args()
    import jax
    dev = jax.devices()[0]
    if not args.chunk: args.chunk = 4096 if dev.platform == "cpu" else 16384
    print(f"JAX device: {dev} | chunk {args.chunk} pixels")
    if args.inspect: cubdl_io.load(args.inspect, verbose=True); return
    if args.check: check(args.check, args); return
    if not args.data: ap.error("give --data DIR, --check FILE or --inspect FILE")
    files = sorted(p for p in args.data.iterdir() if p.suffix.lower() in (".hdf5", ".h5"))
    if args.files: files = [p for p in files if p.stem in set(args.files)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {r["file"] for r in csv.DictReader(open(args.out))
                if r["method"] == ("l2_phase" if args.layers else "gds_phase")}
    new = not args.out.exists()
    with open(args.out, "a", newline="") as fh:
        args.fh = fh; w = csv.DictWriter(fh, FIELDS)
        if new: w.writeheader()
        for p in files:
            try:
                run_file(p, args, w, done)
            except Exception as e:                                       # one bad file must not stop the run
                print(f"{p.stem}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
