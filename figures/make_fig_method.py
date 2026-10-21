"""
Method and image figure for the SoS paper (Waterloo calf, OpenH-RF).

    (a) schematic of the two plane-wave delay models (angles exaggerated)
    (b) CF against the beamforming speed for one calf, both slowness models (grid curves of cubdl_sos)
    (c) B-mode images of the same calf at 1540 m/s and at the two CF estimates; 30-60 mm window marked

Run in the cubdl-sos-gpu image, from the repository root:
    docker run --rm --gpus all -v "$PWD":/w -w /w --entrypoint python cubdl-sos-gpu figures/make_fig_method.py

The example calf is the file whose error (consistent slowness, CF, 30-60 mm) is nearest to the median
absolute error, so that the example is typical. Writes figures/fig_method.pdf/png.
"""
import csv, json, sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType, no Type 3 fonts (IEEE PDF check)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cubdl_sos"))
import cubdl_io                                                          # noqa: E402
from das import Beamformer                                              # noqa: E402
from run_sos_comparison import find_convention                          # noqa: E402

RES = ROOT / "data" / "openh_rf"
DATA = RES / "waterloo_calf"
SETS = {"angle": RES / "results_waterloo_30_60mm", "delays": RES / "results_layers_30_60mm"}
OUT = Path(__file__).resolve().parent
C0, DR = 1540.0, 60


def estimates(folder, method="gds_cf"):
    e = {}
    for r in csv.DictReader(open(folder / "results.csv")):
        if r["method"] == method: e[r["file"]] = (float(r["c_est"]), float(r["ref_sos"]))
    return e


EST = {k: estimates(v) for k, v in SETS.items()}
err = {f: c - ref for f, (c, ref) in EST["delays"].items()}
med = np.median(np.abs(list(err.values())))
FILE = min(err, key=lambda f: abs(abs(err[f]) - med))
REF = EST["delays"][FILE][1]
C_ANG, C_DEL = EST["angle"][FILE][0], EST["delays"][FILE][0]
print(f"example {FILE}: ref {REF:.0f}, angle {C_ANG:.0f}, delays {C_DEL:.0f}, median |error| {med:.1f}")

# ---------------------------------------------------------------- images
d = cubdl_io.load(DATA / f"{FILE}.hdf5", frame=0)
args = Namespace(convention="cubdl", check_pixels=10000, fnum=1.0, chunk=16384, max_offset=4e-6,
                 assume_max_angle=0.0, patch=16, pair_sep=0)
conv = find_convention(d, args)
xs = np.linspace(float(d.xe.min()), float(d.xe.max()), 256)
zs = np.linspace(5e-3, 65e-3, 600)
imgs = []
for c, sl in ((C0, "delays"), (C_ANG, "angle"), (C_DEL, "delays")):
    bf = Beamformer(d, xs, zs, 1.0, chunk=args.chunk, phase_ref=conv["phase_ref"], slowness=sl)
    bf.set_geometry(angles=conv["angles"], t0_sign=conv["t0_sign"], t_off=conv["t_off"], tx_ref=conv["tx_ref"])
    imgs.append(bf.envelope(c).reshape(bf.shape))
ref_max = max(im.max() for im in imgs)

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "font.family": "serif"})
fig = plt.figure(figsize=(7.16, 2.4))
gs = fig.add_gridspec(1, 5, width_ratios=[1.2, 1.2, 1, 1, 1], wspace=0.6, left=0.015, right=0.99, bottom=0.2, top=0.86)

# (a) schematic: array at z = 0, firing delays with slope sin(theta)/c0, wave fronts in a medium with c > c0
ax = fig.add_subplot(gs[0]); th = np.deg2rad(20); ratio = 1.4                 # exaggerated for visibility
th_true = np.arcsin(ratio * np.sin(th))
ZB = 1.9                                                                  # bottom of the drawn medium
ax.fill_between([-1.2, 1.2], 0, ZB, color="0.94", lw=0)
ax.plot([-1, 1], [0, 0], "k", lw=3, solid_capstyle="butt")
for ls, col, a in (("--", "0.45", th), ("-", "C3", th_true)):
    for r in (0.45, 0.95, 1.45):                                         # wave fronts at three times
        x = np.linspace(-1.2, 1.2, 200); z = (r - x * np.sin(a)) / np.cos(a)
        ok = (z >= 0) & (z <= ZB); ax.plot(x[ok], z[ok], ls, color=col, lw=0.9)
    ax.annotate("", xy=(-0.6 + 1.1 * np.sin(a), 0.15 + 1.1 * np.cos(a)), xytext=(-0.6, 0.15),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
ax.text(0, -0.12, r"firing delays: slope $\sin\theta/c_0$", fontsize=6, ha="center", va="bottom")
ax.plot([], [], "--", color="0.45", lw=0.9, label=r"model: $p=\sin\theta/c$")
ax.plot([], [], "-", color="C3", lw=0.9, label=r"true: $p=\sin\theta/c_0$")
ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, handlelength=1.8)
ax.set_xlim(-1.25, 1.25); ax.set_ylim(ZB, -0.35); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(r"(a) schematic, $c>c_0$")

# (b) CF against c (grid curves of the run)
ax = fig.add_subplot(gs[1])
for sl, col, lab, ce in (("angle", "0.45", r"$p=\sin\theta/c$", C_ANG), ("delays", "C3", r"$p=\sin\theta/c_0$", C_DEL)):
    cur = np.array(json.loads((SETS[sl] / "curves" / f"{FILE}.json").read_text())["curves"]["cf"])
    v = cur[:, 1] / cur[:, 1].max()
    ax.plot(cur[:, 0], v, color=col, lw=1, label=lab)
    ax.plot([ce], [np.interp(ce, cur[:, 0], v)], "o", color=col, ms=3)
ax.axvline(REF, color="k", ls=":", lw=0.9, label="through-transmission")
ax.set_ylim(None, 1.02); ax.set_xlim(1480, 1640); ax.set_xlabel("beamforming speed $c$ (m/s)"); ax.set_ylabel("CF (normalized)", labelpad=1)
ax.legend(fontsize=5.5, loc="lower center", frameon=False)
ax.set_title("(b) focus metric, one calf")

# (c) B-mode images
labels = [f"(c) 1540 m/s", f"$\\sin\\theta/c$: {C_ANG:.0f} m/s", f"$\\sin\\theta/c_0$: {C_DEL:.0f} m/s"]
win = (zs >= 30e-3) & (zs <= 60e-3)
ref_lvl = imgs[0].max()                                                   # 0 dB: maximum of the 1540 m/s image
for i, (im, lab) in enumerate(zip(imgs, labels)):
    ax = fig.add_subplot(gs[2 + i])
    db = 20 * np.log10(np.minimum(im / ref_lvl, 1) + 1e-6)
    ax.imshow(db, cmap="gray", vmin=-DR, vmax=0, aspect="equal",
              extent=[xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3])
    for z in (30, 60): ax.axhline(z, color="C1", ls="--", lw=0.7)
    ax.set_title(lab); ax.set_xlabel("x (mm)")
    if i == 0: ax.set_ylabel("depth (mm)")
    else: ax.set_yticklabels([])
for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_method.{ext}", dpi=300)
print("wrote", OUT / "fig_method.pdf")
