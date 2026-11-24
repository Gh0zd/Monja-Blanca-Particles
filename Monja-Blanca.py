import time
import numpy as np
import cv2

# ============================================================
# Configuración
# ============================================================
W, H = 720, 612                    # resolución interna (divisible por 4)
DISPLAY_W, DISPLAY_H = 1080, 918   # tamaño al mostrar
S = W / 1080.0                     # escala respecto al diseño base
FX, FY = int(540 * S), int(390 * S)

DENS = 2.6        # densidad de partículas en la flor (por píxel)
LDENS = 1.1       # densidad en hojas y tallo

rng = np.random.default_rng(7)
P = dict(stem=1, bloom=1, stamen=1, zoom=1)

LIGHT = np.array([-0.35, 0.55, 0.75], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


# ============================================================
# Utilidades
# ============================================================
def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def lerp(a, b, t):
    return a + (b - a) * t


def count(length, halfw, dens):
    return int(length * S * halfw * S * 1.2 * dens) + 50


def sample(n, p, q):
    """u con densidad proporcional al ancho -> área uniformemente cubierta."""
    ug = np.linspace(0, 1, 512)
    w = ug ** p * (1 - ug) ** q
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    u = np.interp(rng.random(n), cdf, ug).astype(np.float32)
    v = (rng.random(n) * 2 - 1).astype(np.float32)
    return u, v


class Part:
    """Superficie paramétrica tipo pétalo: u a lo largo, v a lo ancho."""

    def __init__(self, length, halfw, p, q, dens=DENS):
        n = count(length, halfw, dens)
        self.u, self.v = sample(n, p, q)
        self.p, self.q = p, q
        um = p / (p + q)
        self.norm = um ** p * (1 - um) ** q

    def surf(self, o, a, length, halfw, cup=0.0, bend=0.0, ref=(0, 0, 1)):
        a = unit(a)
        s = np.cross(a, ref)
        s = (s / np.linalg.norm(s)).astype(np.float32)
        nr = np.cross(s, a).astype(np.float32)
        u, v = self.u, self.v
        w = halfw * (u ** self.p) * ((1 - u) ** self.q) / self.norm
        lat = v * w
        off = cup * lat * lat / halfw + bend * u * u * length
        pos = (np.asarray(o, np.float32) + a * (u * length)[:, None]
               + s * lat[:, None] + nr * off[:, None])
        sl = 2 * cup * lat / halfw
        sa = 2 * bend * u
        n = nr - s * sl[:, None] - a * sa[:, None]
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return pos.astype(np.float32), n.astype(np.float32)


# ============================================================
# Partes (se crean una sola vez)
# ============================================================
DORSAL = Part(225, 66, 0.50, 0.95)
LAT = [Part(240, 78, 0.45, 0.80), Part(240, 78, 0.45, 0.80)]
PET = [Part(120, 40, 0.50, 0.95), Part(120, 40, 0.50, 0.95)]
LIP = Part(66, 30, 0.60, 0.50, dens=DENS * 1.4)
TONGUE = Part(45, 9, 0.50, 0.50, dens=DENS * 1.5)
COLUMN = Part(45, 9, 0.40, 0.40, dens=DENS * 1.5)


class Leaf:
    def __init__(self, phi, length, halfw, th0, k):
        self.phi, self.length, self.halfw = phi, length, halfw
        self.th0, self.k = th0, k
        self.part = Part(length, halfw, 0.75, 0.55, dens=LDENS)


LEAVES = [
    Leaf(-2.9, 400, 44, 0.18, 0.9),
    Leaf(2.6, 390, 42, 0.22, 1.0),
    Leaf(-2.2, 380, 46, 0.30, 1.0),
    Leaf(1.9, 370, 44, 0.28, 1.1),
    Leaf(-1.5, 340, 40, 0.42, 1.1),
    Leaf(1.3, 330, 40, 0.45, 1.2),
    Leaf(-0.9, 270, 36, 0.55, 1.2),
    Leaf(0.8, 260, 36, 0.55, 1.2),
]

# tallo
NST = int(420 * S * 14 * S * LDENS * 6) + 200
ST_U = rng.random(NST).astype(np.float32)
ST_A = (rng.random(NST) * 2 * np.pi).astype(np.float32)

# fondo (gradiente suave)
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r2 = ((_xx - FX) / (W * 0.55)) ** 2 + ((_yy - FY * 1.05) / (H * 0.60)) ** 2
BG = (0.010 + 0.040 * np.exp(-_r2 * 1.6))[..., None] * np.array([0.85, 1.0, 0.80], np.float32)


# ============================================================
# Cámara, splat con z-buffer y bloom
# ============================================================
def cam(t, p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    ang = 0.42 * np.sin(0.45 * t)
    c, s = np.cos(ang), np.sin(ang)
    x, z = x * c + z * s, -x * s + z * c
    el = np.radians(16)
    y2 = y * np.cos(el) - z * np.sin(el)
    z2 = y * np.sin(el) + z * np.cos(el)
    f = (1 + 0.0007 * z2) * P["zoom"] * S
    return x * f, -y2 * f, z2


KERN = np.ones((3, 3), np.uint8)
LAYERS = [([], []), ([], [])]   # 0: hojas y tallo, 1: flor
CUR_LAYER = 0


def splat(p, col):
    """Solo encola; la proyección y el z-buffer se hacen una vez por capa."""
    LAYERS[CUR_LAYER][0].append(p)
    LAYERS[CUR_LAYER][1].append(col)


def flush_layer(layer):
    qp, qc = LAYERS[layer]
    buf = np.zeros((H * W, 3), np.float32)
    cov = np.zeros(H * W, np.float32)
    if qp:
        p = np.concatenate(qp)
        c = np.concatenate(qc)
        qp.clear(); qc.clear()
        X, Y, z = cam(CUR_T, p)
        i = (X + FX).astype(np.int32)
        j = (Y + FY).astype(np.int32)
        ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
        k = (j * W + i)[ok]
        c = c[ok]
        o = np.argsort(z[ok]).astype(np.int32)   # lejos -> cerca; el último gana
        win = np.full(H * W, -1, np.int32)
        win[k[o]] = o
        m = win >= 0
        buf[m] = c[win[m]]
        cov[m] = 1.0
    buf = buf.reshape(H, W, 3)
    cov = cov.reshape(H, W)
    # rellena huecos pequeños entre partículas (cierre morfológico)
    closed = cv2.morphologyEx(cov, cv2.MORPH_CLOSE, KERN)
    fill = (closed > 0) & (cov == 0)
    if fill.any():
        np.copyto(buf, cv2.dilate(buf, KERN), where=fill[..., None])
        cov[fill] = 1.0
    return buf, cov


def flush():
    plant, _ = flush_layer(0)
    fl, cf = flush_layer(1)
    m = cf > 0
    plant[m] = fl[m]
    return plant


def bloom(x):
    def level(k, sigma):
        s = cv2.resize(x, (W // k, H // k), interpolation=cv2.INTER_AREA)
        s = cv2.GaussianBlur(s, (0, 0), sigma * S)
        return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)

    return x + 0.30 * level(2, 2.2) + 0.55 * level(4, 6.0)


# ============================================================
# Color
# ============================================================
def shade(n):
    return 0.52 + 0.48 * np.abs(n @ LIGHT)


def petal_color(n, u, v, tint=(1.0, 1.0, 1.0), ao=0.0):
    inten = shade(n)
    inten = inten * (0.80 + 0.20 * smooth(0.0, 0.35, u))
    if ao:
        inten = inten * (1 - ao + ao * smooth(0.0, 0.5, u))
    inten = inten * (1 - 0.06 * np.exp(-(v * 6) ** 2))        # nervio central
    base = np.array([0.94, 0.97, 0.86], np.float32)           # base verdosa
    t = np.array(tint, np.float32)
    col = lerp(base, t, smooth(0.0, 0.35, u)[:, None])
    return col * inten[:, None]


# ============================================================
# Flor
# ============================================================
def place(pos, nrm, c, tilt):
    ct, st = np.cos(tilt), np.sin(tilt)

    def rot(p):
        y = p[:, 1] * ct + p[:, 2] * st
        z = -p[:, 1] * st + p[:, 2] * ct
        return np.stack([p[:, 0], y, z], 1)

    return rot(pos) + c, rot(nrm)


def flower(t):
    b = float(ease(P["bloom"]))
    yc = -420 + 420 * P["stem"]
    tilt = 0.36 + 0.04 * np.sin(0.8 * t)
    c = np.array([0, yc, 0], np.float32)

    gs = 0.45 + 0.55 * b      # escala de largo
    gw = 0.55 + 0.45 * b      # escala de ancho
    cupc = 1 + 1.6 * (1 - b)  # más cerrado cuando es capullo

    def A(closed, opened):
        return lerp(np.array(closed, np.float32), np.array(opened, np.float32), b)

    # --- sépalo dorsal
    pos, n = DORSAL.surf((0, 10, -6), A((0, 1, 0), (0, 0.97, -0.20)),
                         225 * gs, 66 * gw, cup=0.25 * cupc, bend=0.05)
    pos, n = place(pos, n, c, tilt)
    splat(pos, petal_color(n, DORSAL.u, DORSAL.v, ao=0.30))

    # --- sépalos laterales
    for sg, part in zip((-1, 1), LAT):
        pos, n = part.surf((sg * 8, -4, -4),
                           A((sg * 0.10, 0.99, 0.05), (sg * 0.93, -0.37, 0.0)),
                           240 * gs, 78 * gw, cup=0.20 * cupc, bend=0.03)
        pos, n = place(pos, n, c, tilt)
        splat(pos, petal_color(n, part.u, part.v, ao=0.30))

    if b < 0.08:
        return

    k = float(smooth(0.08, 0.5, b))

    # --- pétalos (capuchón)
    for sg, part in zip((-1, 1), PET):
        pos, n = part.surf((sg * 6, 8, 12),
                           A((sg * 0.05, 1, 0.1), (sg * 0.26, 0.90, 0.34)),
                           120 * gs * k, 40 * gw * k, cup=0.5, bend=0.12)
        pos, n = place(pos, n, c, tilt)
        splat(pos, petal_color(n, part.u, part.v, tint=(0.99, 0.98, 0.92), ao=0.15))

    # --- columna
    st = float(P["stamen"])
    pos, n = COLUMN.surf((0, 2, 8), (0, -0.25, 0.97),
                         45 * k * (0.5 + 0.5 * st), 9 * k, cup=1.5, bend=0.0)
    pos, n = place(pos, n, c, tilt)
    col = np.array([1.0, 0.92, 0.62], np.float32) * shade(n)[:, None]
    splat(pos, col)

    # --- labio
    pos, n = LIP.surf((0, -4, 18), (0, -0.80, 0.60),
                      72 * gs * k, 32 * gw * k, cup=0.45, bend=-0.08)
    pos, n = place(pos, n, c, tilt)
    inten = shade(n)
    white = petal_color(n, LIP.u, LIP.v)
    ym = smooth(0.95, 0.25, np.abs(LIP.v)) * (1 - smooth(0.55, 0.95, LIP.u))
    yellow = np.array([1.0, 0.76, 0.20], np.float32) * inten[:, None]
    splat(pos, lerp(white, yellow, ym[:, None]))

    # --- callo / lengüeta amarilla
    pos, n = TONGUE.surf((0, -4, 24), (0, -0.70, 0.70),
                         45 * k, 9 * k, cup=0.5, bend=0.0)
    pos, n = place(pos, n, c, tilt)
    col = np.array([1.0, 0.86, 0.38], np.float32) * (shade(n) + 0.1)[:, None]
    splat(pos, col)


# ============================================================
# Hojas y tallo
# ============================================================
def fog(z):
    return np.clip(0.8 + 0.25 * z / 300, 0.5, 1.1)


_LEAF_CACHE = {}


def _build_leaves(g):
    P_, C_ = [], []
    for lf in LEAVES:
        part = lf.part
        u, v = part.u, part.v
        L_ = lf.length * g
        ns = 80
        s = np.linspace(0, 1, ns)
        th = lf.th0 + lf.k * s ** 1.3
        ds = L_ / ns
        r = np.cumsum(np.sin(th)) * ds
        h = np.cumsum(np.cos(th)) * ds
        rr = np.interp(u, s, r)
        hh = np.interp(u, s, h)
        thu = np.interp(u, s, th)

        w = lf.halfw * (u ** part.p) * ((1 - u) ** part.q) / part.norm
        lat = v * w
        sp, cp = np.sin(lf.phi), np.cos(lf.phi)
        Nx, Ny, Nz = -np.cos(thu) * sp, np.sin(thu), -np.cos(thu) * cp
        off = 0.30 * np.abs(lat) + 2.5 * np.cos(v * np.pi * 6)

        x = sp * rr + cp * lat + Nx * off
        y = -415 + hh + Ny * off
        z = -10 + cp * rr - sp * lat + Nz * off
        pos = np.stack([x, y, z], 1).astype(np.float32)

        n = np.stack([Nx - 0.35 * np.sign(v) * cp, Ny, Nz + 0.35 * np.sign(v) * sp], 1)
        n = (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)

        lum = (0.30 + 0.70 * np.abs(n @ LIGHT)) * (0.82 + 0.18 * np.cos(v * np.pi * 6))
        lum = lum * (1 + 0.5 * np.exp(-(v * 10) ** 2))
        lum = lum * (0.5 + 0.5 * smooth(0.0, 0.3, u)) * fog(z)
        P_.append(pos)
        C_.append(np.array([0.07, 0.40, 0.14], np.float32) * lum[:, None].astype(np.float32))
    return np.concatenate(P_), np.concatenate(C_)


def leaves(t):
    g = round(0.30 + 0.70 * float(P["stem"]), 2)
    if g not in _LEAF_CACHE:
        _LEAF_CACHE.clear()
        _LEAF_CACHE[g] = _build_leaves(g)
    pos, col = _LEAF_CACHE[g]
    splat(pos, col)


def stem(t):
    yc = -420 + 420 * P["stem"]
    u, a = ST_U, ST_A
    y = -420 + (yc - 10 + 420) * u
    rad = 5 + 4 * u ** 3
    x = 10 * (np.sin(y / 130) - np.sin(yc / 130)) + rad * np.cos(a)
    z = -10 + rad * np.sin(a)
    pos = np.stack([x, y, z], 1).astype(np.float32)
    n = np.stack([np.cos(a), 0 * a, np.sin(a)], 1).astype(np.float32)
    lum = 0.35 + 0.65 * np.abs(n @ LIGHT)
    base = lerp(np.array([0.12, 0.38, 0.10], np.float32),
                np.array([0.55, 0.72, 0.35], np.float32), (u ** 2)[:, None])
    splat(pos, base * lum[:, None] * 0.9)


# ============================================================
def finish(buf, t):
    np.multiply(buf, -1.5, out=buf)
    np.exp(buf, out=buf)
    out = (BG + 1.0) - buf            # 1 - exp(-1.5 x) + fondo
    return cv2.convertScaleAbs(out, alpha=255)


CUR_T = 0.0


def frame(t):
    global CUR_T, CUR_LAYER
    CUR_T = t
    CUR_LAYER = 0
    leaves(t)
    stem(t)
    CUR_LAYER = 1
    flower(t)
    buf = flush()
    buf = bloom(buf)
    return finish(buf, t)


# ============================================================
if __name__ == "__main__":
    cv2.namedWindow("Monja Blanca", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Monja Blanca", 900, 760)

    fps = 30
    t = 0.0
    dt = 1.0 / fps
    last = time.perf_counter()
    frames = 0

    while True:
        P["stem"] = ease(t * 0.9)
        P["bloom"] = ease((t - 1.2) * 0.6)
        P["stamen"] = ease((t - 2.5) * 0.6)
        P["zoom"] = 1 + 0.10 * ease(t * 0.2)

        img = frame(t)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (DISPLAY_W, DISPLAY_H),
                             interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Monja Blanca", img_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        t += dt
        if t > 8.5:
            t = 0.0

        frames += 1
        now = time.perf_counter()
        if now - last >= 1.0:
            print(f"{frames / (now - last):.1f} FPS")
            frames = 0
            last = now

    cv2.destroyAllWindows()