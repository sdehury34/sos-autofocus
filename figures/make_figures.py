"""
Figures and tables for the ISBI paper (Waterloo calf, OpenH-RF), from the result files of cubdl_sos.

    python figures/make_figures.py        (in the cubdl-sos-gpu image: needs numpy, scipy, matplotlib)

Writes figures/fig_scatter.pdf, figures/table_main.tex, figures/numbers.txt.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, pearsonr, linregress
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType, no Type 3 fonts (IEEE PDF check)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "openh_rf"
OUT = Path(__file__).resolve().parent
SETS = {  # (slowness, depth window) -> result file
    ("angle", "5-80"): "results_waterloo_5_80mm/results.csv",
    ("angle", "30-60"): "results_waterloo_30_60mm/results.csv",
    ("delays", "5-80"): "results_layers_5_80mm/results.csv",
    ("delays", "30-60"): "results_layers_30_60mm/results.csv",
}


def load(p):
    res = defaultdict(dict); ref = {}; alt = {}
    for r in csv.DictReader(open(RES / p)):
        res[r["method"]][r["file"]] = float(r["c_est"]); ref[r["file"]] = float(r["ref_sos"]); alt[r["file"]] = float(r["ref_alt"])
    return res, ref, alt


DATA = {k: load(v) for k, v in SETS.items()}
_, REF, ALT = DATA[("delays", "30-60")]
FILES = sorted(REF); R = np.array([REF[f] for f in FILES]); A = np.array([ALT[f] for f in FILES])
PART = [f.split("_")[1] for f in FILES]; LEG = np.array(["left" in f for f in FILES])


def est(cfg, m):
    return np.array([DATA[cfg][0][m][f] for f in FILES])


def stats(E):
    lr = np.mean([abs(E[i] - E[j]) for i in range(len(FILES)) for j in range(i + 1, len(FILES)) if PART[i] == PART[j]])
    return dict(mae=np.mean(np.abs(E - R)), bias=np.mean(E - R), r=pearsonr(E, R)[0], lr=lr, sd=np.std(E - R),
                slope=linregress(R, E).slope)


def num(x, spec):
    t = format(x, spec)
    return t.replace("-", "$-$", 1) if t.startswith("-") else t.replace("+", "$+$", 1)


# ---------------------------------------------------------------- table
rows = [(r"Xiao et al.~\cite{xiao2024}", None, None, A)]
for (sl, zw), lab in [(("angle", "5-80"), "angle"), (("angle", "30-60"), "angle"), (("delays", "5-80"), "delays"),
                      (("delays", "30-60"), "delays")]:
    for m, mname in (("gds_cf", "CF"), ("gds_phase", "phase"), ("gds_bright", "brightness")):
        rows.append((f"global, {mname}", lab, zw, est((sl, zw), m)))
for zw in ("5-80", "30-60"):
    rows.append(("two-layer, CF", "delays", zw, est(("delays", zw), "l2_cf")))
tex = [r"\begin{tabular}{llcrrrrr}", r"\toprule",
       r"Model, objective & $p$ & Depth (mm) & MAE & Bias & $r$ & Slope & L--R \\", r"\midrule"]
for lab, sl, zw, E in rows:
    s = stats(E)
    sl_t = {"angle": r"$\sin\theta/c$", "delays": r"$\sin\theta/c_0$", None: "--"}[sl]
    tex.append(f"{lab} & {sl_t} & {zw.replace('-', '--') if zw else '--'} & {s['mae']:.1f} & {num(s['bias'], '+.1f')} & "
               f"{num(s['r'], '.2f')} & {num(s['slope'], '.2f')} & {s['lr']:.1f} \\\\")
    if lab.startswith("Xiao") or (lab == "global, brightness"): tex.append(r"\addlinespace[1pt]")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_main.tex").write_text("\n".join(tex))

# ---------------------------------------------------------------- numbers for the text
old = est(("angle", "30-60"), "gds_cf"); new = est(("delays", "30-60"), "gds_cf")
w = wilcoxon(np.abs(old - R), np.abs(new - R))
lines = [f"files {len(FILES)}, participants {len(set(PART))}",
         f"through-tx mean {R.mean():.1f} sd {R.std():.1f} range {R.min():.1f}-{R.max():.1f}",
         f"left-right difference of through-tx: {stats(R)['lr']:.1f}",
         f"CF 30-60: angle MAE {stats(old)['mae']:.1f} -> delays MAE {stats(new)['mae']:.1f}; Wilcoxon on |error| p = {w.pvalue:.2e}",
         f"files improved: {int((np.abs(new - R) < np.abs(old - R)).sum())} of {len(FILES)}"]
P = np.array(PART); PS = sorted(set(PART))
po = np.array([np.abs(old - R)[P == q].mean() for q in PS]); pn = np.array([np.abs(new - R)[P == q].mean() for q in PS])
lines.append(f"per participant (mean |error| of both legs): Wilcoxon p = {wilcoxon(po, pn).pvalue:.2e}, improved {int((pn < po).sum())} of {len(PS)}")
lines.append(f"slope of estimate on reference (CF 30-60): angle {stats(old)['slope']:.2f}, delays {stats(new)['slope']:.2f}, Xiao {stats(A)['slope']:.2f}")
for k in ("gds_cf", "gds_phase", "grid_phase"):
    s = stats(est(("delays", "30-60"), k)); lines.append(f"delays 30-60 {k}: " + ", ".join(f"{a} {b:.2f}" for a, b in s.items()))
s = stats(A); lines.append("Xiao: " + ", ".join(f"{a} {b:.2f}" for a, b in s.items()))
(OUT / "numbers.txt").write_text("\n".join(lines)); print("\n".join(lines))

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6})
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), sharex=True, sharey=True)
panels = [(old, r"(a) global CF, $p=\sin\theta/c$, 30–60 mm"), (new, r"(b) global CF, $p=\sin\theta/c_0$, 30–60 mm"),
          (A, "(c) Xiao et al. (file values)")]
lo, hi = 1465, 1610
for ax, (E, title) in zip(axs, panels):
    s = stats(E)
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.7, ls="--", zorder=1)
    ax.scatter(R[LEG], E[LEG], s=10, color="#1f5aa6", label="left calf", zorder=2, lw=0)
    ax.scatter(R[~LEG], E[~LEG], s=10, color="#d9822b", marker="s", label="right calf", zorder=2, lw=0)
    ax.set_title(title, fontsize=7.5)
    ax.text(0.04, 0.96, f"MAE {s['mae']:.1f} m/s\nbias {s['bias']:+.1f} m/s\n$r$ = {s['r']:.2f}", transform=ax.transAxes,
            va="top", fontsize=7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlabel("Through-transmission SoS (m/s)")
axs[0].set_ylabel("Pulse-echo estimate (m/s)"); axs[0].legend(loc="lower right", fontsize=6.5, frameon=False)
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig_scatter.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / "fig_scatter.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote fig_scatter.pdf, table_main.tex, numbers.txt")
