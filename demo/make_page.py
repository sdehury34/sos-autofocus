"""
Precomputes the interactive demo page: for each true speed of sound, simulates the medium, finds the
coherence-factor maximum with both delay models and makes the B-mode images. Writes docs/index.html,
a single file that runs in any browser without Python.

    python demo/make_page.py            # about 15 min on a laptop CPU; results are cached in demo/out
"""
import base64, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import demo                                                             # noqa: E402

C0, LO, HI = 1540.0, 1420.0, 1720.0
SPEEDS = list(range(1440, 1661, 10))
CACHE = ROOT / "demo" / "out" / "page"
XS_MM, ZS = None, np.linspace(5e-3, 50e-3, 200)


def search(bf):
    curve, (lo, hi, step) = {}, (LO, HI, 20.0)
    for nxt in (5.0, 1.0, None):
        for c in np.arange(lo, hi + 1e-6, step):
            if LO <= c <= HI and float(c) not in curve: curve[float(c)] = bf.metrics(float(c))["cf"]
        best = max(curve, key=curve.get)
        if nxt is None: break
        lo, hi, step = best - step, best + step, nxt
    cs = sorted(curve); cf = np.array([curve[c] for c in cs])
    return {"est": float(cs[int(np.argmax(cf))]), "c": cs, "cf": (cf / cf.max()).round(4).tolist()}


def png(env, ref):
    db = 20 * np.log10(np.minimum(env / ref, 1) + 1e-6)
    g = np.clip((db + 50) / 50 * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(g).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def one(c_true):
    f = CACHE / f"{c_true}.json"
    if f.exists(): return json.loads(f.read_text())
    path = demo.simulate(float(c_true), CACHE)
    res = {sl: search(demo.beamformer(path, sl, nx=64, nz=80)) for sl in ("angle", "delays")}
    envs = []
    for sl, c in (("delays", C0), ("angle", res["angle"]["est"]), ("delays", res["delays"]["est"])):
        bf = demo.beamformer(path, sl, nx=128, nz=ZS.size, zlim=(ZS[0], ZS[-1]))
        envs.append(bf.envelope(c).reshape(bf.shape))
    res["img"] = [png(e, envs[0].max()) for e in envs]
    res["x_mm"] = [float(bf.d.xe.min() * 1e3), float(bf.d.xe.max() * 1e3)]
    f.write_text(json.dumps(res)); print(f"{c_true} m/s: conventional {res['angle']['est']:.0f}, "
                                        f"consistent {res['delays']['est']:.0f}", flush=True)
    return res


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    data = {str(c): one(c) for c in SPEEDS}
    page = (ROOT / "demo" / "page_template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "docs" / "index.html"; out.parent.mkdir(exist_ok=True); out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
