"""
Differentiable plane-wave delay-and-sum (JAX) and the speed-of-sound metrics.

All metrics are ratios of pixel sums, so the image is processed in pixel chunks and the derivative
d(sums)/dc comes from forward-mode differentiation (jax.jvp). The memory use is thus independent of
the number of pixels and of the number of transmit angles.

The data, the pixels and the delay convention (angles, start times, time offset, transmit reference)
(and the channel data) are runtime inputs of the compiled functions, so a change of the convention does not compile again.

Pixel sums (one value for each chunk, then added):
    0  sum CF_p                  coherence factor of each pixel, CF = |sum_e s_e|^2 / (N_a sum_e |s_e|^2)
    1  sum env                   env = |sum_e sum_theta s_e,theta|
    2  sum env^2
    3  sum env^4
    4  sum Re(C)                 C = S_i conj S_i+k, S_i: image of transmit i (transmits in angle order)
    5  sum over patches |sum_patch C|   (patches of `patch` adjacent pixels along x)
    6  number of pixels
Metrics (all made so that a LARGER value is better):
    cf        = sums0 / sums6                      mean coherence factor
    bright    = sums1 / sums6                      mean speckle brightness
    sharp     = sums3 * sums6 / sums2^2            normalized 4th moment (image sharpness)
    phase     = sums4 / sums5                      weighted mean of cos(phase difference) between the
                                                   images of transmit pairs (i, i+k), averaged in patches
"""
import math
import numpy as np
import jax
import jax.numpy as jnp

METRICS = ("cf", "bright", "sharp", "phase")
EPS = 1e-20


def _abs2(z):
    return jnp.real(z) ** 2 + jnp.imag(z) ** 2


def _sabs(z):
    """|z| with a finite derivative at z = 0 (pixels with no signal, e.g. below the recorded depth)."""
    return jnp.sqrt(_abs2(z) + EPS)


def metrics_from_sums(s):
    n = s[6]
    return {"cf": s[0] / n, "bright": s[1] / n, "sharp": s[3] * n / (s[2] ** 2 + 1e-30),
            "phase": s[4] / (s[5] + 1e-30)}


class Beamformer:
    """Holds the data of one acquisition on the device, the pixel grid and the delay convention."""

    def __init__(self, d, xs, zs, fnum=1.0, chunk=4096, patch=16, pair_sep=0, phase_ref=None, slowness="angle",
                 newton_steps=4):
        """slowness: horizontal slowness p of each plane wave.
            "angle":  p = sin(theta) / c_top (the angle is kept when c changes; CUBDL and most DAS codes)
            "delays": p = sin(theta) / c_file (fixed by the transmit delays that the scanner computed with the
                      sound speed of the file; physically exact)
        The medium is homogeneous (layer depth h = 0) or has two layers: speed c1 from the skin to depth h,
        speed c2 below (horizontal interface, refraction by Snell's law)."""
        self.d = d
        self.slowness, self.newton_steps = slowness, int(newton_steps)
        self.h = 0.0
        self.rf = jnp.asarray(d.data)                                  # (T, S, N), transmits in angle order
        self.xe = jnp.asarray(d.xe, jnp.float32)
        self.fs, self.fdem, self.fnum = float(d.fs), float(d.fdemod), float(fnum)
        # time reference of the IQ phase: "rel" = time from the first sample (RF demodulated here),
        # "abs" = full delay tau, "z" = tau - 2z/c (IQ demodulated relative to the pixel depth)
        self.phase_ref = phase_ref or ("abs" if d.is_iq else "rel")
        T = d.data.shape[0]
        self.patch = int(patch)
        self.k = min(int(pair_sep) if pair_sep > 0 else max(1, T // 5), max(T - 1, 1))
        self.angles, self.t0_sign, self.t_off, self.tx_ref = d.angles, 1.0, 0.0, "origin"
        self.set_geometry()
        self._sums = jax.jit(self._sums_impl)
        # forward-mode derivatives of the pixel sums with respect to theta = (c2, c1), for a given tangent
        self._sums_jvp = jax.jit(lambda th, t, g, p, h, th_: jax.jvp(lambda q, hh: self._sums_impl(q, g, p, hh),
                                                                      (th, h), (t, th_)))
        self._img = jax.jit(self._image_impl)
        X, Z = np.meshgrid(xs, zs)                                     # (nz, nx)
        self.shape = X.shape
        self.set_pixels(X.ravel(), Z.ravel(), chunk)

    # ---------------------------------------------------------------- convention and pixels
    def set_geometry(self, angles=None, t0_sign=None, t_off=None, tx_ref=None):
        """Delay convention: sample time = tau - (t0_sign * start_time + t_off); the transmit delay is
        referenced to x = 0 ("origin") or to the first-firing aperture edge ("edge")."""
        if angles is not None: self.angles = np.asarray(angles, np.float64)
        if t0_sign is not None: self.t0_sign = float(t0_sign)
        if t_off is not None: self.t_off = float(t_off)
        if tx_ref is not None: self.tx_ref = tx_ref
        a = self.angles; xe = np.asarray(self.d.xe)
        xr = np.where(np.sin(a) >= 0, xe.min(), xe.max()) if self.tx_ref == "edge" else np.zeros_like(a)
        self.g = (self.rf, jnp.asarray(a, jnp.float32), jnp.asarray(self.t0_sign * self.d.t0 + self.t_off, jnp.float32),
                  jnp.asarray(xr, jnp.float32))

    def set_pixels(self, xp, zp, chunk=4096):
        chunk = max(self.patch, chunk // self.patch * self.patch)      # whole patches in a chunk
        P = xp.size; nch = max(1, math.ceil(P / chunk)); pad = nch * chunk - P
        w = np.r_[np.ones(P), np.zeros(pad)].astype(np.float32)
        xp = np.r_[xp, np.full(pad, xp[0])].astype(np.float32); zp = np.r_[zp, np.full(pad, zp[0])].astype(np.float32)
        self.P = P
        self.p = (jnp.asarray(xp.reshape(nch, chunk)), jnp.asarray(zp.reshape(nch, chunk)),
                  jnp.asarray(w.reshape(nch, chunk)))

    # ---------------------------------------------------------------- per-transmit channel data
    def _delays(self, th, xr, theta, h, xp, zp):
        """Two-way delay (Pc, N) in a medium with speed c1 above depth h and c2 below (h = 0: homogeneous c2).
        Transmit: plane wave with horizontal slowness p (conserved across the interface), exact.
        Receive: refracted ray, crossing point from Newton steps on the travel time (Fermat). The time is
        stationary at the crossing point, so its derivative needs no derivative of the crossing point."""
        c2, c1 = theta[0], theta[1]
        ctop = jnp.where(h > 0, c1, c2)
        p = jnp.sin(th) / (ctop if self.slowness == "angle" else self.d.c_file)
        q1 = jnp.sqrt(jnp.maximum(1.0 / ctop ** 2 - p ** 2, 1e-14))
        q2 = jnp.sqrt(jnp.maximum(1.0 / c2 ** 2 - p ** 2, 1e-14))
        below = (zp > h) & (h > 0)
        ttx = p * (xp - xr) + jnp.where(below, h * q1 + (zp - h) * q2, zp * jnp.where(h > 0, q1, q2))
        D = xp[:, None] - self.xe[None, :]; Z = zp[:, None]
        straight = jnp.sqrt(D ** 2 + Z ** 2) / ctop
        hs = jnp.maximum(h, 1e-4); Zb = jnp.maximum(Z - hs, 1e-6)
        u = D * hs / jnp.maximum(Z, hs + 1e-6)                          # straight-line start
        for _ in range(self.newton_steps):
            a = jnp.sqrt(u ** 2 + hs ** 2); b = jnp.sqrt((D - u) ** 2 + Zb ** 2)
            f1 = u / (c1 * a) - (D - u) / (c2 * b)
            f2 = hs ** 2 / (c1 * a ** 3) + Zb ** 2 / (c2 * b ** 3)
            u = jnp.clip(u - f1 / f2, jnp.minimum(D, 0.0), jnp.maximum(D, 0.0))  # crossing between element and pixel
        u = jax.lax.stop_gradient(u)
        layered = jnp.sqrt(u ** 2 + hs ** 2) / c1 + jnp.sqrt((D - u) ** 2 + Zb ** 2) / c2
        trx = jnp.where(below[:, None], layered, straight)
        return ttx[:, None] + trx, D

    def _chan(self, rf, th, t0e, xr, theta, h, xp, zp):
        tau, dx = self._delays(th, xr, theta, h, xp, zp)
        samp = (tau - t0e) * self.fs
        S = rf.shape[0]
        fl = jnp.floor(samp); fr = (samp - fl).astype(jnp.float32); lo = jnp.clip(fl.astype(jnp.int32), 0, S - 2)
        cols = jnp.arange(rf.shape[1])[None, :]
        v = rf[lo, cols] * (1 - fr) + rf[lo + 1, cols] * fr
        ok = (fl >= 0) & (fl < S - 1) & (jnp.abs(dx) <= zp[:, None] / (2 * self.fnum) + 1e-4)
        tph = {"rel": samp / self.fs, "abs": tau, "z": tau - 2 * zp[:, None] / theta[0]}[self.phase_ref]
        v = v * jnp.exp(1j * 2 * math.pi * self.fdem * tph).astype(jnp.complex64)
        return jnp.where(ok, v, 0), ok

    def _chunk_sums(self, theta, h, g, xp, zp, w):
        k, p = self.k, self.patch
        def body(carry, inp):
            acc, ring, re, ab, i = carry
            v, ok = self._chan(*inp, theta, h, xp, zp)
            Si = jnp.sum(v, axis=1)
            C = w * Si * jnp.conj(ring[0])                               # pair (i - k, i)
            Cp = jnp.sum(C.reshape(-1, p), axis=1)                       # patch sums
            use = i >= k
            re = re + jnp.where(use, jnp.sum(jnp.real(C)), 0.0)
            ab = ab + jnp.where(use, jnp.sum(_sabs(Cp)), 0.0)
            ring = jnp.concatenate([ring[1:], Si[None]], axis=0)
            return (acc + v, ring, re, ab, i + 1), ok
        Pc, N = xp.shape[0], self.xe.shape[0]
        init = (jnp.zeros((Pc, N), jnp.complex64), jnp.zeros((k, Pc), jnp.complex64),
                jnp.float32(0), jnp.float32(0), jnp.int32(0))
        (acc, _, re, ab, _), oks = jax.lax.scan(body, init, g)
        na = jnp.maximum(jnp.sum(oks[0], axis=1), 1).astype(jnp.float32)
        Sp = jnp.sum(acc, axis=1); env = _sabs(Sp)
        den = na * jnp.sum(_abs2(acc), axis=1)
        ok = den > 1e-20                                                # pixels with no signal: CF = 0
        cf = jnp.where(ok, _abs2(Sp) / jnp.where(ok, den, 1.0), 0.0)    # safe division (finite derivative)
        return jnp.stack([jnp.sum(w * cf), jnp.sum(w * env), jnp.sum(w * env ** 2), jnp.sum(w * env ** 4),
                          re, ab, jnp.sum(w)])

    def _sums_impl(self, theta, g, p, h):
        return jnp.sum(jax.lax.map(lambda a: self._chunk_sums(theta, h, g, *a), p), axis=0)

    def _image_impl(self, theta, g, p, h):
        def f(a):
            xp, zp, _ = a
            def body(acc, inp):
                v, _ = self._chan(*inp, theta, h, xp, zp); return acc + jnp.sum(v, axis=1), None
            acc, _ = jax.lax.scan(body, jnp.zeros(xp.shape[0], jnp.complex64), g)
            return jnp.abs(acc)
        return jax.lax.map(f, p).reshape(-1)

    # ---------------------------------------------------------------- public API
    # homogeneous medium: c is the one speed of sound (theta = (c, c), h = 0)
    def _theta(self, c2, c1=None):
        return jnp.asarray([c2, c2 if c1 is None else c1], jnp.float32)

    def metrics(self, c, c1=None, h=None):
        """Metrics for a homogeneous medium (c1 None) or for two layers (c1 above depth h, c below)."""
        hh = jnp.float32(0.0 if c1 is None else (self.h if h is None else h))
        s = self._sums(self._theta(c, c1), self.g, self.p, hh)
        return {k: float(v) for k, v in metrics_from_sums(s).items()}

    def value_and_grad(self, c, metric):
        """Homogeneous medium: metric value and d(metric)/dc (forward mode, tangent (1, 1))."""
        s, ds = self._sums_jvp(self._theta(c), jnp.ones(2, jnp.float32), self.g, self.p, jnp.float32(0.0),
                               jnp.float32(0.0))
        val, g = jax.jvp(lambda q: metrics_from_sums(q)[metric], (s,), (ds,))
        return float(val), float(g)

    def value_and_grad2(self, c2, c1, metric, h=None, with_h=False):
        """Two layers: metric value and (d/dc2, d/dc1[, d/dh]), one forward-mode pass for each parameter."""
        th = self._theta(c2, c1); hh = jnp.float32(self.h if h is None else h); out = []
        tangents = [(jnp.asarray([1.0, 0.0], jnp.float32), 0.0), (jnp.asarray([0.0, 1.0], jnp.float32), 0.0)]
        if with_h: tangents.append((jnp.zeros(2, jnp.float32), 1.0))
        for t, th_ in tangents:
            s, ds = self._sums_jvp(th, t, self.g, self.p, hh, jnp.float32(th_))
            val, g = jax.jvp(lambda q: metrics_from_sums(q)[metric], (s,), (ds,)); out.append(float(g))
        return float(val), tuple(out)

    def envelope(self, c, c1=None, h=None):
        hh = jnp.float32(0.0 if c1 is None else (self.h if h is None else h))
        return np.asarray(self._img(self._theta(c, c1), self.g, self.p, hh))[: self.P]
