"""
Interactive demo of speed-of-sound autofocus (runs in the web browser, no download).

    pip install -r cubdl_sos/requirements.txt gradio
    python demo/app.py             # opens http://127.0.0.1:7860
    python demo/app.py --share     # also a public link (for example from Colab)
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo                                                             # noqa: E402

C0, LO, HI = 1540.0, 1420.0, 1720.0
PRESETS = [("Fat", 1460), ("Muscle", 1580), ("Fast tissue", 1640)]
OUT = Path(__file__).resolve().parent / "out"
MODELS = {"angle": ("Conventional", r"$p=\sin\theta/c$", "p = sin θ / c", "0.45"),
          "delays": ("Transmit-delay-consistent", r"$p=\sin\theta/c_0$", "p = sin θ / c₀", "#d62728")}
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})

INTRO = """
# Speed-of-sound autofocus: does the delay model matter?

A scanner fires steered plane waves with delays computed for **1540 m/s**. If the tissue has another
speed of sound, the beamformer must model how these waves travel. This demo simulates a medium with a
speed of sound **you choose**, and then estimates it back from the echoes with two delay models:

* **Conventional** (used by most beamformers): keeps the steering angle fixed when the speed changes.
* **Transmit-delay-consistent** (our paper): keeps the slowness that the firing delays impose.

**How to use:** 1. pick the true speed with the slider (or press a tissue button); 2. press **Run**;
3. compare the two estimates with the true speed.
"""

EXPLAIN = """
### What am I seeing?

* **Focus curve:** for each trial speed, the echoes are beamformed and the *coherence factor* (how well
  the echoes of all elements line up) is measured. The estimate is the speed with the highest value.
* **Images:** the same medium beamformed at 1540 m/s and at the two estimates. In this noise-free
  simulation the point targets become sharp at the right speed. In real tissue the difference is much
  harder to see (Fig. 1c of the paper), but the estimated speed, the biomarker, still changes.
* **Try it:** move the true speed far from 1540 m/s (for example 1460 or 1640). The conventional
  estimate stays closer to 1540 m/s; the consistent estimate follows the true speed. A larger steering
  angle (Advanced) makes the difference larger.

The data are simulated (128-element linear array, 5 MHz, speckle with a cyst and four point targets).
The paper shows the same effect in 62 in vivo calf scans: the error decreases from 27.6 to 12.5 m/s.
"""


def search(bf, progress, frac0, frac1, label):
    """Coarse-to-fine grid search of the coherence-factor maximum (20, 5, 1 m/s)."""
    curve, n_total, n_done = {}, 16 + 9 + 11, 0
    lo, hi, step = LO, HI, 20.0
    for nxt in (5.0, 1.0, None):
        for c in np.arange(lo, hi + 1e-6, step):
            c = float(c)
            if c not in curve and LO <= c <= HI:
                curve[c] = bf.metrics(c)["cf"]
            n_done += 1
            progress(frac0 + (frac1 - frac0) * min(n_done / n_total, 1), desc=label)
        best = max(curve, key=curve.get)
        if nxt is None: break
        lo, hi, step = best - step, best + step, nxt
    cs = np.array(sorted(curve)); cf = np.array([curve[c] for c in cs])
    return float(cs[np.argmax(cf)]), cs, cf


def curve_figure(res, c_true):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for sl, (name, eq, _, col) in MODELS.items():
        c_est, cs, cf = res[sl]
        ax.plot(cs, cf / cf.max(), "-", color=col, lw=1.8, label=f"{name} ({eq}): {c_est:.0f} m/s")
        ax.plot([c_est], [1.0], "o", color=col, ms=8, zorder=3)
    ax.axvline(c_true, color="k", ls=":", lw=1.5, label=f"true speed: {c_true:.0f} m/s")
    ax.axvline(C0, color="C0", ls="--", lw=1, label="scanner speed $c_0$: 1540 m/s")
    ax.set_xlim(LO, HI); ax.set_xlabel("trial speed of sound (m/s)"); ax.set_ylabel("coherence factor (normalized)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Focus curve: the estimate is the peak", fontsize=10)
    fig.tight_layout()
    return fig


def image_figure(imgs, xs, zs):
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0), sharey=True)
    ref = imgs[0][1].max()
    for ax, (title, im) in zip(axs, imgs):
        db = 20 * np.log10(np.minimum(im / ref, 1) + 1e-6)
        ax.imshow(db, cmap="gray", vmin=-50, vmax=0, aspect="equal",
                  extent=[xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3])
        ax.set_title(title, fontsize=8.5); ax.set_xlabel("x (mm)")
    axs[0].set_ylabel("depth (mm)")
    fig.tight_layout()
    return fig


def summary(res, c_true, seconds):
    rows = ["| Delay model | Estimate (m/s) | Error (m/s) |", "|---|---|---|"]
    for sl, (name, _, eq, _) in MODELS.items():
        c = res[sl][0]
        rows.append(f"| {name.replace('Transmit-delay-consistent', 'Consistent')}<br><sub>{eq}</sub> | **{c:.0f}** | {c - c_true:+.0f} |")
    e_conv, e_cons = abs(res["angle"][0] - c_true), abs(res["delays"][0] - c_true)
    if abs(c_true - C0) < 15:
        note = "The true speed is close to 1540 m/s, where both models are the same. Move the slider further away."
    elif e_cons < e_conv:
        note = (f"The conventional model is pulled toward 1540 m/s (error {e_conv:.0f} m/s); "
                f"the consistent model is within {e_cons:.0f} m/s of the truth.")
    else:
        note = "Both models give a similar error for this setting."
    return f"### Result (true speed {c_true:.0f} m/s)\n\n" + "\n".join(rows) + f"\n\n{note}\n\n*Run time: {seconds:.0f} s*"


def run(c_true, max_angle, n_angles, progress=gr.Progress()):
    t = time.time(); c_true = float(c_true); n_angles = int(n_angles) | 1
    progress(0.02, desc="Simulating the echoes")
    path = demo.simulate(c_true, OUT, angles=n_angles, max_angle=float(max_angle))
    res = {}
    for i, sl in enumerate(MODELS):
        bf = demo.beamformer(path, sl, nx=64, nz=80)
        res[sl] = search(bf, progress, 0.08 + 0.38 * i, 0.46 + 0.38 * i, f"Autofocus, {MODELS[sl][0].lower()} model")
    progress(0.88, desc="Making the images")
    xs, zs = np.linspace(-19e-3, 19e-3, 128), np.linspace(5e-3, 50e-3, 200)
    imgs = []
    for title, sl, c in (("1540 m/s (scanner)", "delays", C0),
                         (f"Conventional: {res['angle'][0]:.0f} m/s", "angle", res["angle"][0]),
                         (f"Consistent: {res['delays'][0]:.0f} m/s", "delays", res["delays"][0])):
        bf = demo.beamformer(path, sl, nx=xs.size, nz=zs.size, zlim=(zs[0], zs[-1]))
        imgs.append((title, bf.envelope(c).reshape(bf.shape)))
    xs = np.linspace(bf.d.xe.min(), bf.d.xe.max(), xs.size)
    return curve_figure(res, c_true), image_figure(imgs, xs, zs), summary(res, c_true, time.time() - t)


def build():
    with gr.Blocks(title="Speed-of-sound autofocus demo", theme=gr.themes.Soft()) as app:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                c_true = gr.Slider(1440, 1660, value=1620, step=10, label="True speed of sound in the medium (m/s)",
                                   info="Muscle is about 1580 m/s, fat about 1450 m/s.")
                with gr.Accordion("Advanced", open=False):
                    max_angle = gr.Slider(5, 20, value=15, step=1, label="Largest steering angle (degrees)")
                    n_angles = gr.Slider(11, 41, value=31, step=2, label="Number of plane waves")
                btn = gr.Button("Run", variant="primary", size="lg")
                gr.Markdown("**Or try a tissue in one click:**")
                with gr.Row():
                    presets = [gr.Button(f"{name}\n{c} m/s", size="sm") for name, c in PRESETS]
                result = gr.Markdown("Press **Run** to start. One run takes about 15-40 s on a CPU.")
            with gr.Column(scale=2):
                curve = gr.Plot(show_label=False)
                images = gr.Plot(show_label=False)
        gr.Markdown(EXPLAIN)
        outputs = [curve, images, result]
        btn.click(run, [c_true, max_angle, n_angles], outputs)
        for b, (_, c) in zip(presets, PRESETS):
            b.click(lambda c=c: c, None, c_true).then(run, [c_true, max_angle, n_angles], outputs)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--share", action="store_true"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().queue().launch(share=a.share, server_port=a.port, inbrowser=not a.share)
