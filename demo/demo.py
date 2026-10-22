"""
Demo without any download: speed-of-sound autofocus on simulated plane-wave data.

For each true speed of sound, the script simulates a medium (speckle and point targets, 31 plane waves
of +-15 deg, firing delays computed with c0 = 1540 m/s, as on a real scanner) and estimates the speed
of sound by the maximum of the coherence factor (CF) with two delay models:

    conventional slowness       p = sin(theta) / c      (keeps the angle when c changes)
    transmit-delay-consistent   p = sin(theta) / c0     (keeps the slowness fixed by the firing delays)

The conventional model pulls the estimate toward c0; the consistent model finds the true speed.

    python demo/demo.py                         # about 5 min on a laptop CPU
    python demo/demo.py --speeds 1520 1600      # other true speeds (m/s)
    python demo/demo.py --gd                    # also gradient descent through the beamformer (slower)

Writes demo/out/demo.png and prints the estimates.
"""
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cubdl_sos"))
import cubdl_io                                                          # noqa: E402
from das import Beamformer                                              # noqa: E402
from run_sos_comparison import gradient_descent                         # noqa: E402

C0 = 1540.0
MODELS = {"angle": r"conventional, $p=\sin\theta/c$", "delays": r"consistent, $p=\sin\theta/c_0$"}


def simulate(c_true, out_dir, angles=31, max_angle=15.0):
    """Simulated channel data (CUBDL layout) with true speed c_true and firing delays computed with c0."""
    path = Path(out_dir) / f"SYN{c_true:.0f}_{angles}x{max_angle:.0f}.hdf5"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(ROOT / "cubdl_sos" / "make_synthetic.py"), "--out", str(path),
                        "--c", str(c_true), "--angles", str(angles), "--max-angle", str(max_angle), "--slowness", "delays"],
                       check=True, stdout=subprocess.DEVNULL)
    return path


def beamformer(path, slowness, nx=96, nz=120, zlim=(10e-3, 45e-3)):
    d = cubdl_io.load(path)
    xs = np.linspace(d.xe.min(), d.xe.max(), nx); zs = np.linspace(*zlim, nz)
    bf = Beamformer(d, xs, zs, 1.0, chunk=4096, slowness=slowness)
    bf.set_geometry(angles=d.angles, t0_sign=1.0, t_off=0.0, tx_ref="origin")    # convention of the simulator
    return bf


def autofocus(bf, lo=1440.0, hi=1700.0, step=10.0):
    """Grid search of the CF maximum: coarse step, then 1 m/s around the maximum."""
    curve = {float(c): bf.metrics(c)["cf"] for c in np.arange(lo, hi + 1e-6, step)}
    best = max(curve, key=curve.get)
    for c in np.arange(best - step + 1, best + step, 1.0):
        if lo <= c <= hi and float(c) not in curve: curve[float(c)] = bf.metrics(c)["cf"]
    cs = np.array(sorted(curve)); cf = np.array([curve[c] for c in cs])
    return float(cs[np.argmax(cf)]), cs, cf


def plot(res, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    colors = {"angle": "0.45", "delays": "C3"}
    speeds = sorted(res)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ex = max(speeds, key=lambda c: abs(c - C0))
    for sl in MODELS:
        c_est, cs, cf = res[ex][sl]
        ax[0].plot(cs, cf / cf.max(), color=colors[sl], lw=1.2, label=MODELS[sl])
        ax[0].plot([c_est], [1.0], "o", color=colors[sl], ms=4)
    ax[0].axvline(ex, color="k", ls=":", lw=1, label=f"true speed ({ex:.0f} m/s)")
    ax[0].axvline(C0, color="C0", ls="--", lw=0.8, label=r"$c_0$ (1540 m/s)")
    ax[0].set_xlabel("beamforming speed $c$ (m/s)"); ax[0].set_ylabel("CF (normalized)")
    ax[0].set_title("(a) focus metric for one medium"); ax[0].legend(fontsize=6, frameon=False, loc="upper left")
    lo, hi = min(speeds) - 30, max(speeds) + 30
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=0.7, label="identity")
    for sl in MODELS:
        ax[1].plot(speeds, [res[c][sl][0] for c in speeds], "o-", color=colors[sl], ms=4, lw=1, label=MODELS[sl])
    ax[1].axhline(C0, color="C0", ls="--", lw=0.8)
    ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi); ax[1].set_aspect("equal")
    ax[1].set_xlabel("true speed (m/s)"); ax[1].set_ylabel("estimate (m/s)")
    ax[1].set_title("(b) estimate against true speed"); ax[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speeds", type=float, nargs="+", default=[1500.0, 1580.0, 1620.0], help="true speeds (m/s)")
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "out")
    ap.add_argument("--gd", action="store_true", help="also gradient descent (consistent model, last speed)")
    a = ap.parse_args()
    import jax
    print(f"JAX device: {jax.devices()[0]}")
    res = {}
    for c_true in a.speeds:
        path = simulate(c_true, a.out); res[c_true] = {}
        for sl in MODELS:
            t = time.time(); res[c_true][sl] = autofocus(beamformer(path, sl))
            print(f"true {c_true:.0f} m/s | {sl:6s} slowness: estimate {res[c_true][sl][0]:.0f} m/s "
                  f"(error {res[c_true][sl][0] - c_true:+.0f}, {time.time() - t:.0f} s)", flush=True)
    if a.gd:
        c_true = a.speeds[-1]; t = time.time()
        c, n, _, _ = gradient_descent(beamformer(simulate(c_true, a.out), "delays"), "cf")
        print(f"gradient descent (consistent, start 1540): {c:.1f} m/s for true {c_true:.0f} m/s "
              f"({n} gradient evaluations, {time.time() - t:.0f} s)")
    plot(res, a.out / "demo.png"); print(f"wrote {a.out / 'demo.png'}")


if __name__ == "__main__":
    main()
