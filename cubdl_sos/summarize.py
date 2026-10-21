"""
Summary of the CUBDL speed-of-sound comparison.

    python summarize.py results/results.csv [reference.csv]

Reads the results and reference_sos.csv (CUBDL "optimized speed of sound", used for the CUBDL
evaluation) and writes:
    results/summary.txt         errors of each method (all files and each institution)
    results/summary_table.tex   LaTeX table
    results/scatter.png         estimate against reference, one panel for each method
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ORDER = ["nominal", "grid_cf", "grid_bright", "grid_sharp", "grid_phase", "gd_cf", "gd_bright", "gd_phase",
         "gds_cf", "gds_bright", "gds_phase"]
LABEL = {"nominal": "Nominal 1540 m/s", "grid_cf": "Grid: coherence factor", "grid_bright": "Grid: brightness",
         "grid_sharp": "Grid: sharpness", "grid_phase": "Grid: phase coherence",
         "gd_cf": "GD: coherence factor (ours)", "gd_bright": "GD: brightness (ours)",
         "gd_phase": "GD: phase error (DBUA-style)",
         "gds_cf": "Smoothed GD: coherence factor", "gds_bright": "Smoothed GD: brightness",
         "gds_phase": "Smoothed GD: phase error"}
INST = ["OSL", "MYO", "UFL", "EUT", "INS"]


def main(path, ref_path=None):
    path = Path(path); out = path.parent
    ref_path = Path(ref_path) if ref_path else next((q for q in (out / "reference_sos.csv", HERE / "reference_sos.csv")
                                                     if q.exists()), out / "reference_sos.csv")
    ref = {r["file"]: float(r["reference_sos"]) for r in csv.DictReader(open(ref_path))}
    res = defaultdict(dict); tim = defaultdict(list); nev = defaultdict(list); ncc = {}
    for r in csv.DictReader(open(path)):
        if r["file"] not in ref and r.get("ref_sos") not in (None, ""):
            ref[r["file"]] = float(r["ref_sos"])                       # reference stored in the file (zea)
        if r["file"] not in ref: continue
        res[r["method"]][r["file"]] = float(r["c_est"])
        tim[r["method"]].append(float(r["time_s"])); nev[r["method"]].append(float(r["n_evals"]))
        ncc[r["file"]] = (float(r["score"]) if r.get("score_type") == "ncc" and r["score"] not in ("", "nan")
                          else np.nan)
    files = sorted(set.intersection(*[set(res[m]) for m in ORDER if m in res])) if res else []
    methods = [m for m in ORDER if m in res]
    if not files:
        print(f"No file of {ref_path.name} is complete in {path}. (Synthetic files: give a reference CSV "
              "as the second argument, with the columns file,reference_sos.)"); return
    nv = [ncc[f] for f in files if np.isfinite(ncc[f])]
    lines = [f"Files: {len(files)} of {len(ref)} in {ref_path.name}",
             (f"Beamformer check (NCC with beamformed_data, {len(nv)} files): median {np.median(nv):.3f}, "
              f"min {np.min(nv):.3f}" if nv else "Beamformer check: no file with beamformed_data"),
             f"Files with the convention selected by the coherence factor (no beamformed_data): {len(files) - len(nv)}",
             ""]
    hdr = f"{'Method':32s} {'MAE':>6s} {'Median':>7s} {'Bias':>6s} {'Max':>6s} {'<=10':>5s} {'<=20':>5s} {'Time/file':>10s} {'Evals':>6s}"
    lines += [hdr, "-" * len(hdr)]
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Method & MAE & Median AE & Bias & $\le$10~m/s & Time/file (s) \\", r"\midrule"]
    INST[:] = sorted({f[:3] for f in files}, key=lambda i: (INST + [i]).index(i))
    for m in methods:
        e = np.array([res[m][f] - ref[f] for f in files]); a = np.abs(e)
        lines.append(f"{LABEL[m]:32s} {a.mean():6.1f} {np.median(a):7.1f} {e.mean():+6.1f} {a.max():6.0f} "
                     f"{(a <= 10).sum():5d} {(a <= 20).sum():5d} {np.mean(tim[m]):10.1f} {np.mean(nev[m]):6.0f}")
        tex.append(f"{LABEL[m]} & {a.mean():.1f} & {np.median(a):.1f} & {e.mean():+.1f} & {(a <= 10).sum()}/{len(files)} "
                   f"& {np.mean(tim[m]):.0f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    lines += ["", "MAE (m/s) for each institution:", f"{'Method':32s} " + " ".join(f"{i:>6s}" for i in INST)]
    for m in methods:
        cells = []
        for i in INST:
            fi = [f for f in files if f.startswith(i)]
            cells.append(f"{np.mean([abs(res[m][f] - ref[f]) for f in fi]):6.1f}" if fi else f"{'-':>6s}")
        lines.append(f"{LABEL[m]:32s} " + " ".join(cells))
    lines += ["", "Units: m/s. Time and evaluations are for the method only (gradient descent: per method; "
              "grid: one shared sweep gives all four grid methods)."]
    (out / "summary.txt").write_text("\n".join(lines)); (out / "summary_table.tex").write_text("\n".join(tex))
    print("\n".join(lines))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = len(methods); nc = 4; nr = int(np.ceil(k / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.2 * nr), squeeze=False)
    cols = dict(zip(INST, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
    for ax, m in zip(axs.ravel(), methods):
        for i in INST:
            fi = [f for f in files if f.startswith(i)]
            ax.scatter([ref[f] for f in fi], [res[m][f] for f in fi], s=14, color=cols[i], label=i)
        ax.plot([1420, 1640], [1420, 1640], "k--", lw=0.6)
        a = np.mean([abs(res[m][f] - ref[f]) for f in files])
        ax.set_title(f"{LABEL[m]}\nMAE {a:.1f} m/s", fontsize=8); ax.set_xlim(1420, 1640); ax.set_ylim(1380, 1690)
        ax.tick_params(labelsize=7)
    for ax in axs.ravel()[k:]: ax.axis("off")
    axs[0, 0].legend(fontsize=6); fig.supxlabel("CUBDL reference speed of sound (m/s)", fontsize=9)
    fig.supylabel("Estimated speed of sound (m/s)", fontsize=9); fig.tight_layout()
    fig.savefig(out / "scatter.png", dpi=150); print(f"\nsaved {out/'summary.txt'}, {out/'summary_table.tex'}, {out/'scatter.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/results.csv", sys.argv[2] if len(sys.argv) > 2 else None)
