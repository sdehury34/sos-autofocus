"""
Tests of the two-layer delay model in das.py (run in the Docker image, from the tool folder):

    python test_layers.py SYNTHETIC_FILE.hdf5 [das_old.py]

1. Regression: homogeneous metrics equal those of the previous das.py (if given).
2. Two layers with c1 = c2 give the homogeneous metrics.
3. The Newton crossing point gives the minimum travel time of a brute-force search (Fermat).
4. Forward-mode derivatives with respect to c2 and c1 agree with central finite differences.
"""
import sys, importlib.util
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, "/tool")
import cubdl_io
from das import Beamformer

path = sys.argv[1]
d = cubdl_io.load(path)
xs, zs = np.linspace(d.xe.min(), d.xe.max(), 96), np.linspace(5e-3, 45e-3, 120)
bf = Beamformer(d, xs, zs); bf.set_geometry(tx_ref="origin")
ok = True

if len(sys.argv) > 2:                                                   # 1. regression
    spec = importlib.util.spec_from_file_location("das_old", sys.argv[2]); old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    bo = old.Beamformer(d, xs, zs); bo.set_geometry(tx_ref="origin")
    for c in (1480.0, 1540.0, 1600.0):
        a, b = bf.metrics(c), bo.metrics(c)
        err = max(abs(a[k] - b[k]) / (abs(b[k]) + 1e-12) for k in a)
        print(f"1. regression c={c:.0f}: max relative difference {err:.2e}"); ok &= err < 5e-4                                   # float32 rounding

for c in (1500.0, 1560.0):                                              # 2. c1 = c2 with a layer
    a, b = bf.metrics(c), bf.metrics(c, c1=c, h=15e-3)
    err = max(abs(a[k] - b[k]) / (abs(a[k]) + 1e-12) for k in a)
    print(f"2. layer with c1 = c2 = {c:.0f}: max relative difference {err:.2e}"); ok &= err < 5e-4                                   # float32 rounding

rng = np.random.default_rng(0)                                         # 3. Newton vs brute force
xp = rng.uniform(-15e-3, 15e-3, 200).astype(np.float32); zp = rng.uniform(16e-3, 45e-3, 200).astype(np.float32)
h, c1, c2 = 12e-3, 1450.0, 1590.0
tau, _ = bf._delays(jnp.float32(0.0), jnp.float32(0.0), jnp.asarray([c2, c1], jnp.float32), jnp.float32(h),
                    jnp.asarray(xp), jnp.asarray(zp))
p = 0.0; q1, q2 = 1 / c1, 1 / c2
trx_newton = np.asarray(tau) - (zp[:, None] - h) * q2 - h * q1          # remove the transmit part (theta = 0)
xe = np.asarray(d.xe)
u = np.linspace(-40e-3, 40e-3, 40001)[None, None, :]
D = (xp[:, None] - xe[None, :])[..., None]; Z = zp[:, None, None]
brute = (np.sqrt(u ** 2 + h ** 2) / c1 + np.sqrt((D - u) ** 2 + (Z - h) ** 2) / c2).min(axis=-1)
err = np.abs(trx_newton - brute).max()
print(f"3. Newton vs brute force: max time difference {err * 1e9:.3f} ns"); ok &= err < 2e-9

for metric in ("cf", "bright"):                                         # 4. gradients
    c2, c1 = 1560.0, 1460.0
    v, (g2, g1) = bf.value_and_grad2(c2, c1, metric, h=12e-3)
    fd2 = (bf.metrics(c2 + 0.5, c1=c1, h=12e-3)[metric] - bf.metrics(c2 - 0.5, c1=c1, h=12e-3)[metric]) / 1.0
    fd1 = (bf.metrics(c2, c1=c1 + 0.5, h=12e-3)[metric] - bf.metrics(c2, c1=c1 - 0.5, h=12e-3)[metric]) / 1.0
    r2 = abs(g2 - fd2) / (abs(fd2) + 1e-12); r1 = abs(g1 - fd1) / (abs(fd1) + 1e-12)
    print(f"4. {metric}: d/dc2 {g2:+.3e} (fd {fd2:+.3e}), d/dc1 {g1:+.3e} (fd {fd1:+.3e})")
    ok &= r2 < 0.05 and r1 < 0.05
print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
