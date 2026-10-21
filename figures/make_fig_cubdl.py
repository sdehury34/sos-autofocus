"""
CUBDL wire-target figure for the SoS paper: does the autofocus speed of sound make wire targets sharper
than the nominal 1540 m/s?

For every CUBDL file of a phantom model with wire targets, the script
    1. beamforms the image at 1540 m/s and at the autofocus estimate of the "delays" run (transmit-delay-
       consistent slowness p = sin(theta)/c_file), with the official CUBDL conventions;
    2. finds the wire targets automatically in the 1540 m/s image (isolated peaks far above the local
       speckle level);
    3. measures the lateral and axial FWHM of each wire at both speeds on a fine local grid.
The example file for the zoomed images is chosen by a fixed rule (--pick): "median" (default) takes the
wire file with the median |c_est - 1540|; "max" takes the largest. Panel (d) shows all wire files, so
the example is not the only evidence.

Run from the repository root, with your CUBDL folder mounted as /data (GPU image):
    docker run --rm --gpus all -v "${PWD}:/w" -v "D:/CUBDL:/data:ro" -w /w \
        --entrypoint python cubdl-sos-gpu figures/make_fig_cubdl.py \
        --data /data --results cubdl_sos/results/results.csv

Writes figures/fig_cubdl.pdf/png, figures/cubdl_wires.csv (every wire) and figures/cubdl_wires.txt
(summary). Only figures and numbers are written; no CUBDL data is copied.
"""
import argparse, csv, sys
from argparse import Namespace
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType, no Type 3 fonts (IEEE PDF check)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cubdl_sos"))
import cubdl_io                                                          # noqa: E402
from das import Beamformer                                              # noqa: E402
from run_sos_comparison import find_convention                          # noqa: E402

OUT = Path(__file__).resolve().parent
C0 = 1540.0
# phantom models with wire (point) targets, from data/cubdl/CUBDL_Phantom_Details.xlsx
WIRE_FILES = (["OSL%03d" % i for i in range(2, 8)] + ["MYO%03d" % i for i in range(1, 7)] +
              ["UFL%03d" % i for i in range(1, 6)] + ["EUT005", "EUT006"] +
              ["INS001", "INS002", "INS005", "INS006", "INS007", "INS008",
               "INS014", "INS015", "INS018", "INS019", "INS020", "INS021"])


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with the CUBDL .hdf5 files")
    ap.add_argument("--results", type=Path, required=True, help="results.csv of the run with --slowness delays")
    ap.add_argument("--method", default="gds_cf", help="estimate to use (default gds_cf, as for the calf)")
    ap.add_argument("--pick", choices=["median", "max"], default="median", help="rule for the example file")
    ap.add_argument("--min-snr", type=float, default=20.0, help="wire peak above the local median (dB)")
    ap.add_argument("--max-wires", type=int, default=12, help="strongest wires kept per file")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--check-dir", type=Path, default=None,
                    help="also save, for each file, the 1540 m/s image with the measured wires marked (visual check)")
    return ap.parse_args()


def estimates(path, method):
    e = {}
    for r in csv.DictReader(open(path)):
        if r["method"] == method and r["file"] in WIRE_FILES: e[r["file"]] = float(r["c_est"])
    return e


class File:
    """One acquisition: loader, delay convention and beamformer (slowness "delays")."""

    def __init__(self, path, chunk):
        self.d = cubdl_io.load(path, frame=0)
        args = Namespace(convention="cubdl", check_pixels=10000, fnum=1.0, chunk=chunk, max_offset=4e-6,
                         assume_max_angle=0.0, patch=16, pair_sep=0)
        conv = find_convention(self.d, args)
        z0, z1 = self.d.zlims or (5e-3, 0.9 * self.d.data.shape[1] / self.d.fs * self.d.c_file / 2)
        self.xs = np.arange(self.d.xe.min(), self.d.xe.max(), 0.1e-3)
        self.zs = np.arange(z0, z1, 0.05e-3)
        self.bf = Beamformer(self.d, self.xs, self.zs, 1.0, chunk=chunk, phase_ref=conv["phase_ref"], slowness="delays")
        self.bf.set_geometry(angles=conv["angles"], t0_sign=conv["t0_sign"], t_off=conv["t_off"], tx_ref=conv["tx_ref"])
        self.chunk = chunk

    def image(self, c, xs=None, zs=None):
        xs = self.xs if xs is None else xs; zs = self.zs if zs is None else zs
        X, Z = np.meshgrid(xs, zs)
        self.bf.set_pixels(X.ravel(), Z.ravel(), self.chunk)
        return self.bf.envelope(c).reshape(X.shape)


def find_wires(env, xs, zs, min_snr, max_wires):
    """Isolated peaks: local maximum in 3 x 3 mm, at least min_snr dB above the median in 8 x 8 mm, small
    -6 dB region (not a specular line), not in the first 2 mm of the depth range or near the edges."""
    dx, dz = xs[1] - xs[0], zs[1] - zs[0]
    db = 20 * np.log10(env / env.max() + 1e-9)
    loc = ndimage.maximum_filter(db, size=(int(3e-3 / dz), int(3e-3 / dx)))
    med = ndimage.median_filter(db[::4, ::4], size=(int(8e-3 / dz / 4), int(8e-3 / dx / 4)))
    med = ndimage.zoom(med, (db.shape[0] / med.shape[0], db.shape[1] / med.shape[1]), order=1)[: db.shape[0], : db.shape[1]]
    cand = np.argwhere((db == loc) & (db - med >= min_snr) & (db > -50))
    out = []
    for iz, ix in cand:
        if zs[iz] < zs[0] + 2e-3 or xs[ix] < xs[0] + 1.5e-3 or xs[ix] > xs[-1] - 1.5e-3: continue
        lab, _ = ndimage.label(db >= db[iz, ix] - 6)
        if (lab == lab[iz, ix]).sum() * dx * dz > 3e-6: continue        # -6 dB area above 3 mm^2: not a point
        out.append((db[iz, ix], xs[ix], zs[iz]))
    out.sort(reverse=True)
    return [(x, z) for _, x, z in out[:max_wires]]


def fwhm(profile, step):
    """Full width at half maximum (amplitude -6 dB) of a 1-D envelope profile, linear interpolation."""
    p = profile / profile.max(); k = int(np.argmax(p))
    lo = k
    while lo > 0 and p[lo] >= 0.5: lo -= 1
    hi = k
    while hi < p.size - 1 and p[hi] >= 0.5: hi += 1
    if p[lo] >= 0.5 or p[hi] >= 0.5: return np.nan                     # does not fall to -6 dB in the window
    left = lo + (0.5 - p[lo]) / (p[lo + 1] - p[lo]); right = hi - 1 + (p[hi - 1] - 0.5) / (p[hi - 1] - p[hi])
    return (right - left) * step


def measure(f, c, x, z):
    """Lateral and axial FWHM of the wire near (x, z) at speed c, on a 0.02 mm grid. The search window
    allows for the depth shift of the wire when c changes (depth scales with c)."""
    zc = z * c / f.d.c_file
    xs = np.arange(x - 2.5e-3, x + 2.5e-3, 0.02e-3); zs = np.arange(zc - 2e-3, zc + 2e-3, 0.02e-3)
    im = f.image(c, xs, zs)
    iz, ix = np.unravel_index(np.argmax(im), im.shape)
    return fwhm(im[iz], 0.02e-3), fwhm(im[:, ix], 0.02e-3), xs[ix], zs[iz]


def main():
    a = parse()
    est = estimates(a.results, a.method)
    files = [n for n in WIRE_FILES if n in est and (a.data / f"{n}.hdf5").exists()]
    print(f"{len(files)} wire-phantom files with a {a.method} estimate")
    rows, per_file = [], {}
    for n in files:
        f = File(a.data / f"{n}.hdf5", a.chunk); c = est[n]
        env0 = f.image(C0)
        wires = find_wires(env0, f.xs, f.zs, a.min_snr, a.max_wires)
        res = []
        for i, (x, z) in enumerate(wires):
            l0, ax0, _, _ = measure(f, C0, x, z); l1, ax1, _, _ = measure(f, c, x, z)
            if np.isfinite([l0, l1, ax0, ax1]).all():
                res.append((l0, l1, ax0, ax1, x, z))
                rows.append(dict(file=n, wire=i, x_mm=round(x * 1e3, 2), z_mm=round(z * 1e3, 2), c_est=c,
                                 lat_1540_mm=round(l0 * 1e3, 3), lat_est_mm=round(l1 * 1e3, 3),
                                 ax_1540_mm=round(ax0 * 1e3, 3), ax_est_mm=round(ax1 * 1e3, 3)))
        if res:
            r = np.array(res); per_file[n] = dict(c=c, n=len(res), lat=np.median(r[:, 1] / r[:, 0]),
                                                 ax=np.median(r[:, 3] / r[:, 2]), meas=r)
        print(f"  {n}: c_est {c:.0f} m/s, {len(wires)} wires found, {len(res)} measured"
              + (f", median lateral FWHM ratio {per_file[n]['lat']:.3f}" if n in per_file else ""))
        if a.check_dir is not None:
            a.check_dir.mkdir(parents=True, exist_ok=True)
            fg, ax = plt.subplots(figsize=(5, 6))
            ax.imshow(20 * np.log10(env0 / env0.max() + 1e-6), cmap="gray", vmin=-60, vmax=0, aspect="equal",
                      extent=[f.xs[0] * 1e3, f.xs[-1] * 1e3, f.zs[-1] * 1e3, f.zs[0] * 1e3])
            for q in res:
                ax.plot(q[4] * 1e3, q[5] * 1e3, "o", mfc="none", mec="C3", ms=10)
                ax.text(q[4] * 1e3 + 1, q[5] * 1e3, f"{q[1] / q[0]:.2f}", color="C3", fontsize=7)
            ax.set_title(f"{n}: 1540 m/s, c_est {c:.0f}; red = measured wires (lateral ratio)", fontsize=8)
            fg.savefig(a.check_dir / f"{n}.png", dpi=110, bbox_inches="tight"); plt.close(fg)
        del f                                                          # free the device memory
    if not per_file: sys.exit("no wires found; lower --min-snr")

    with open(OUT / "cubdl_wires.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    names = sorted(per_file, key=lambda k: abs(per_file[k]["c"] - C0))
    ex = names[len(names) // 2] if a.pick == "median" else names[-1]
    dc = np.array([per_file[k]["c"] - C0 for k in names]); lat = np.array([per_file[k]["lat"] for k in names])
    big = np.abs(dc) >= 30
    lines = [f"files with measured wires: {len(names)}, wires: {len(rows)}, estimate: {a.method} (slowness delays)",
             f"example ({a.pick} |c_est - 1540|): {ex}, c_est {per_file[ex]['c']:.0f} m/s",
             f"median lateral FWHM ratio (estimate / 1540): all files {np.median(lat):.3f}; "
             f"|dc| >= 30 m/s: {np.median(lat[big]) if big.any() else float('nan'):.3f} ({big.sum()} files); "
             f"|dc| < 30 m/s: {np.median(lat[~big]) if (~big).any() else float('nan'):.3f} ({(~big).sum()} files)",
             "file  c_est  n_wires  lateral_ratio  axial_ratio"]
    lines += [f"{k}  {per_file[k]['c']:.0f}  {per_file[k]['n']}  {per_file[k]['lat']:.3f}  {per_file[k]['ax']:.3f}" for k in names]
    (OUT / "cubdl_wires.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines[:3]))

    # ------------------------------------------------------------ figure (single column)
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "font.family": "serif"})
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.5), gridspec_kw=dict(hspace=0.55, wspace=0.45))
    P = per_file[ex]; f = File(a.data / f"{ex}.hdf5", a.chunk)
    # representative wire: the measured wire with the median lateral FWHM ratio; the zoom is centred on it
    r = P["meas"]; ratio = r[:, 1] / r[:, 0]; k = int(np.argsort(ratio)[len(ratio) // 2])
    x, z = r[k, 4], r[k, 5]
    xs = np.arange(x - 4e-3, x + 4e-3, 0.04e-3); zs = np.arange(z - 4e-3, z + 4e-3, 0.02e-3)
    ims = [f.image(C0, xs, zs), f.image(P["c"], xs, zs * P["c"] / f.d.c_file)]
    ref = max(im.max() for im in ims)
    for ax, im, zz, t in zip(axs[0], ims, (zs, zs * P["c"] / f.d.c_file), ("(a) 1540 m/s", f"(b) estimate {P['c']:.0f} m/s")):
        ax.imshow(20 * np.log10(im / ref + 1e-6), cmap="gray", vmin=-60, vmax=0, aspect="equal",
                  extent=[xs[0] * 1e3, xs[-1] * 1e3, zz[-1] * 1e3, zz[0] * 1e3])
        ax.set_title(t); ax.set_xlabel("x (mm)")
    axs[0, 0].set_ylabel("depth (mm)")
    # (c) lateral profile through the peak of the same wire (the FWHM values are the measured ones)
    ax = axs[1, 0]
    for c, col, lab, fw in ((C0, "0.45", "1540 m/s", r[k, 0]), (P["c"], "C3", f"{P['c']:.0f} m/s", r[k, 1])):
        _, _, xp, zp = measure(f, c, x, z)                               # peak position at this speed
        xl = np.arange(xp - 2.5e-3, xp + 2.5e-3 + 1e-9, 0.02e-3)
        pr = f.image(c, xl, np.array([zp]))[0]; pr = pr / pr.max()
        ax.plot((xl - xp) * 1e3, 20 * np.log10(pr + 1e-6), color=col, lw=1, label=f"{lab}, FWHM {fw * 1e3:.2f} mm")
    ax.axhline(-6, color="k", ls=":", lw=0.6); ax.set_ylim(-40, 19); ax.set_xlim(-2.5, 2.5); ax.set_yticks([-40, -30, -20, -10, 0])
    ax.set_xlabel("lateral offset (mm)"); ax.set_ylabel("dB"); ax.legend(fontsize=5, frameon=False, loc="upper center", ncol=1, handlelength=1.2, borderaxespad=0.2)
    ax.set_title("(c) lateral profile, wire in (a)")
    # (d) all wire files: median lateral FWHM ratio against the speed difference
    ax = axs[1, 1]
    for k2, inst_col in zip(names, [f"C{['OSL', 'MYO', 'UFL', 'EUT', 'INS'].index(k[:3])}" for k in names]):
        ax.plot(per_file[k2]["c"] - C0, per_file[k2]["lat"], "o", ms=3, color=inst_col)
    ax.plot(per_file[ex]["c"] - C0, per_file[ex]["lat"], "o", ms=6, mfc="none", color="k")
    ax.axhline(1, color="k", lw=0.6, ls=":"); ax.set_xlabel(r"$c_{\mathrm{est}}-1540$ (m/s)")
    ax.set_ylabel("lateral FWHM ratio"); ax.set_title("(d) all wire files")
    for i, inst in enumerate(["OSL", "MYO", "UFL", "EUT", "INS"]):
        if any(k[:3] == inst for k in names): ax.plot([], [], "o", ms=3, color=f"C{i}", label=inst)
    ax.legend(fontsize=5, frameon=False, ncol=2, loc="upper left", handletextpad=0.1, columnspacing=0.6)
    for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_cubdl.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "fig_cubdl.pdf", "cubdl_wires.csv", "cubdl_wires.txt")


if __name__ == "__main__":
    main()
