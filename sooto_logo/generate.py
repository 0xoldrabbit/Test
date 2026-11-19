#!/usr/bin/env python3
"""Sooto logo generator — parametric, numpy only.

Pipeline
  1. Letter bodies   : skeleton strokes with variable radius -> signed distance field
  2. Fusion          : polynomial smooth-min between letters (one "fleshiness" knob K)
  3. Outline         : marching squares on the zero level set
  4. Slant           : global shear x += SLANT * y
  5. Fur             : displace the outline along (normal + wind), gated by a
                       direction mask so every tuft obeys the same wind
  6. Details         : counters = rotated ellipses, night mark = astroid
  7. Optical sizes   : L / M / S differ only in fur parameters

Run:  python3 generate.py      (writes ./out/*.svg and ./out/sheet.html)
"""
import math
import os
import numpy as np

# ----------------------------------------------------------------- parameters
R = 100.0            # outer radius of the "o" (x-height = 2R)
K = 7.0              # smooth-union radius between letters
SLANT = 0.10         # global shear
WIND = (-0.72, -0.69)  # fur direction (down-left), y is up

COUNTER = dict(a=41.0, b=24.0, rot=52.0)   # one ellipse for all three o's
OVERLAP = 5.0        # how far neighbouring letters sink into each other

S = dict(rho=60.0, r=39.0, sx=1.22, a_top=22.0, a_bot=205.0, bulb=1.16)
T = dict(stem_r=34.0, top=168.0, bar_y=112.0, bar_r=27.0, bar_l=46.0, bar_rt=40.0,
         foot=26.0)
EAR = dict(angle=31.0, base_d=72.0, tip_d=141.0, base_r=33.0, tip_r=6.0)
MARK = dict(y=64.0, r=16.0)                # night mark on the cat's forehead

# optical sizes: lam = tuft wavelength, amp = tuft height, lo/hi = mask window
LOD = {
    "L": dict(lam=25.0, amp=19.0, lo=0.10, hi=0.60, mark=True),
    "M": dict(lam=40.0, amp=19.0, lo=0.45, hi=0.85, mark=True),
    "S": dict(lam=0.0, amp=0.0, lo=0.0, hi=0.0, mark=False),
}

INK, PAPER, PURPLE, MOSS = "#1f1b16", "#fbece8", "#ae8bd0", "#c9f2d3"


# ------------------------------------------------------------------ SDF tools
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def cone(X, Y, p0, p1, r0, r1):
    """Stroke segment whose radius goes linearly from r0 to r1."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t = np.clip(((X - p0[0]) * dx + (Y - p0[1]) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(X - p0[0] - t * dx, Y - p0[1] - t * dy) - (r0 + (r1 - r0) * t)


def stroke(X, Y, pts, radii):
    d = np.full(X.shape, 1e9)
    for i in range(len(pts) - 1):
        d = np.minimum(d, cone(X, Y, pts[i], pts[i + 1], radii[i], radii[i + 1]))
    return d


# -------------------------------------------------------------------- letters
def letter_S(X, Y, cx, cy):
    """Two tangent arcs of radius rho; radius swells toward both terminals."""
    rho, r, sx = S["rho"], S["r"], S["sx"]
    up = np.radians(np.linspace(S["a_top"], 270.0, 36))          # ccw
    lo = np.radians(np.linspace(90.0, 90.0 - (450.0 - S["a_bot"]), 36))  # cw
    pts = [(cx + sx * rho * math.cos(a), cy + rho + rho * math.sin(a)) for a in up]
    pts += [(cx + sx * rho * math.cos(a), cy - rho + rho * math.sin(a)) for a in lo[1:]]
    u = np.linspace(-1, 1, len(pts))
    radii = r * (1 + (S["bulb"] - 1) * np.abs(u) ** 3)
    return stroke(X, Y, pts, radii)


def letter_o(X, Y, cx):
    return np.hypot(X - cx, Y) - R


def letter_t(X, Y, cx):
    sr = T["stem_r"]
    stem = stroke(X, Y, [(cx, T["top"]), (cx, -R + sr), (cx + T["foot"], -R + sr - 3)],
                  [sr * 0.86, sr, sr * 0.9])
    bar = cone(X, Y, (cx - T["bar_l"], T["bar_y"]), (cx + T["bar_rt"], T["bar_y"]),
               T["bar_r"], T["bar_r"])
    return smin(stem, bar, 10.0)


def cat_o(X, Y, cx):
    d = letter_o(X, Y, cx)
    for sgn in (-1, 1):
        a = math.radians(90 + sgn * EAR["angle"])
        c, s = math.cos(a), math.sin(a)
        ear = cone(X, Y, (cx + EAR["base_d"] * c, EAR["base_d"] * s),
                   (cx + EAR["tip_d"] * c, EAR["tip_d"] * s), EAR["base_r"], EAR["tip_r"])
        d = smin(d, ear, 8.0)
    return d


def layout():
    """x centres, derived from widths and the shared OVERLAP."""
    s_half = S["sx"] * S["rho"] + S["r"]
    x_o1 = s_half + R - OVERLAP
    x_o2 = x_o1 + 2 * R - OVERLAP
    x_t = x_o2 + R + T["stem_r"] - OVERLAP + 4
    x_o3 = x_t + T["foot"] + T["stem_r"] + R - OVERLAP - 9
    return dict(S=0.0, o1=x_o1, o2=x_o2, t=x_t, o3=x_o3)


def wordmark_field(X, Y):
    L = layout()
    s_cy = -R - 6 + 2 * S["rho"] + S["r"] * S["bulb"]      # sits on baseline, overshoot 6
    d = letter_S(X, Y, L["S"], s_cy)
    for part in (letter_o(X, Y, L["o1"]), letter_o(X, Y, L["o2"]),
                 letter_t(X, Y, L["t"]), cat_o(X, Y, L["o3"])):
        d = smin(d, part, K)
    return d


def icon_field(X, Y):
    return cat_o(X, Y, 0.0)


# ------------------------------------------------------------ marching squares
def contours(field_fn, x0, x1, y0, y1, step=1.0):
    xs = np.arange(x0, x1 + step, step)
    ys = np.arange(y0, y1 + step, step)
    X, Y = np.meshgrid(xs, ys)
    F = field_fn(X, Y)
    ins = F < 0
    pts, links = {}, {}

    def edge_pt(key):
        if key not in pts:
            i, j, kind = key
            i2, j2 = (i, j + 1) if kind == "h" else (i + 1, j)
            a, b = F[i, j], F[i2, j2]
            t = a / (a - b)
            pts[key] = (xs[j] + t * (xs[j2] - xs[j]), ys[i] + t * (ys[i2] - ys[i]))
        return key

    def link(a, b):
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)

    mixed = (ins[:-1, :-1] != ins[1:, 1:]) | (ins[:-1, 1:] != ins[1:, :-1]) | \
            (ins[:-1, :-1] != ins[:-1, 1:])
    for i, j in zip(*np.nonzero(mixed)):
        a, b, c, d = ins[i, j], ins[i, j + 1], ins[i + 1, j + 1], ins[i + 1, j]
        e = []
        if a != b: e.append(edge_pt((i, j, "h")))
        if b != c: e.append(edge_pt((i, j + 1, "v")))
        if c != d: e.append(edge_pt((i + 1, j, "h")))
        if d != a: e.append(edge_pt((i, j, "v")))
        if len(e) == 2:
            link(e[0], e[1])
        elif len(e) == 4:                      # saddle: resolve with cell centre
            centre = F[i:i + 2, j:j + 2].mean() < 0
            if centre == a:
                link(e[0], e[1]); link(e[2], e[3])
            else:
                link(e[0], e[3]); link(e[1], e[2])

    loops, seen = [], set()
    for start in links:
        if start in seen:
            continue
        loop, prev, cur = [], None, start
        while cur not in seen:
            seen.add(cur)
            loop.append(pts[cur])
            nxt = [n for n in links[cur] if n != prev]
            prev, cur = cur, (nxt[0] if nxt else start)
        P = np.array(loop)
        if area(P) < 0:
            P = P[::-1]
        loops.append(P)
    return loops


def area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def is_hole(P, field_fn):
    """A ccw loop is a hole if stepping outward lands inside the body."""
    t = P[1] - P[0]
    q = P[0] + 1.5 * np.array([t[1], -t[0]]) / np.hypot(*t)
    return float(field_fn(np.array([q[0]]), np.array([q[1]]))[0]) < 0


def resample(P, ds):
    Q = np.vstack([P, P[:1]])
    seg = np.hypot(*np.diff(Q, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(8, int(round(s[-1] / ds)))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])]), s[-1]


def shear(P):
    return np.column_stack([P[:, 0] + SLANT * P[:, 1], P[:, 1]])


# ------------------------------------------------------------------------ fur
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def box(v, w):
    k = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(np.concatenate([v[-w:], v, v[:w]]), k, mode="valid")


def tuft(u, c):
    """One clump: hollow flanks meeting in a sharp tip at u=c (c may vary per point)."""
    rise = np.clip(u / c, 0, 1) ** 1.4
    fall = np.clip((1 - u) / (1 - c), 0, 1) ** 1.4
    return np.where(u < c, rise, fall)


def add_fur(P, lod, field_fn, seed=3.0, ds=0.5):
    if lod["amp"] <= 0:
        return P
    P, L = resample(P, ds)
    n = len(P)
    tan = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    tan /= np.hypot(tan[:, 0], tan[:, 1])[:, None]
    nor = np.column_stack([tan[:, 1], -tan[:, 0]])           # outward for ccw loops
    ang = np.unwrap(np.arctan2(tan[:, 1], tan[:, 0]))
    kappa = box(np.gradient(np.concatenate([ang, [ang[0] + 2 * np.pi]]))[:-1] / ds,
                int(10 / ds))                                 # >0 where convex

    w = np.array(WIND) / np.hypot(*WIND)
    mask = smoothstep(lod["lo"], lod["hi"], nor @ w)          # faces the wind
    mask *= smoothstep(-0.012, 0.002, kappa)                  # none in creases
    mask = box(mask, int(6 / ds))

    count = max(1, round(L / lod["lam"]))
    s = np.arange(n) / n
    phase = s * count + min(0.45, count / 40) * np.sin(2 * np.pi * 5 * s + seed)
    idx = np.floor(phase) % count
    u = phase - np.floor(phase)
    rnd = np.modf(np.abs(np.sin(idx * 12.9898 + seed) * 43758.5453))[0]
    amp = lod["amp"] * (0.50 + 0.50 * rnd)

    d = nor + 0.9 * w
    d /= np.hypot(d[:, 0], d[:, 1])[:, None]
    probe = P + 1.3 * lod["amp"] * d                          # room to grow?
    room = field_fn(probe[:, 0] - SLANT * probe[:, 1], probe[:, 1])
    mask *= box(smoothstep(0.10, 0.45, room / (1.3 * lod["amp"])), int(4 / ds))
    lean = 0.5 + 0.32 * (tan @ w)                             # tip slides downwind
    return P + (amp * mask * tuft(u, lean))[:, None] * d


def simplify(P, tol=0.12):
    """Ramer-Douglas-Peucker on a closed loop."""
    keep = np.zeros(len(P), bool)
    keep[0] = keep[len(P) // 2] = True
    stack = [(0, len(P) // 2), (len(P) // 2, len(P) - 1)]
    keep[-1] = True
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = P[i], P[j]
        ab = b - a
        seg = P[i + 1:j] - a
        dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / (np.hypot(*ab) + 1e-12)
        m = int(np.argmax(dist))
        if dist[m] > tol:
            keep[i + 1 + m] = True
            stack += [(i, i + 1 + m), (i + 1 + m, j)]
    return P[keep]


# -------------------------------------------------------------------- details
def ellipse(cx, cy, a, b, rot, n=72):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    x, y = a * np.cos(t), b * np.sin(t)
    return np.column_stack([cx + x * c - y * s, cy + x * s + y * c])


def astroid(cx, cy, r, n=64):
    """|x|^(2/3) + |y|^(2/3) = r^(2/3)"""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t) ** 3, cy + r * np.sin(t) ** 3])


# ---------------------------------------------------------------------- build
def build(kind, lod_name):
    lod = LOD[lod_name]
    if kind == "wordmark":
        L = layout()
        field = wordmark_field
        loops = contours(field, -170, L["o3"] + 150, -150, 260)
        o_centres, cat_x = [L["o1"], L["o2"], L["o3"]], L["o3"]
    else:
        field = icon_field
        loops = contours(field, -150, 150, -150, 180)
        o_centres, cat_x = [0.0], 0.0

    body = []
    for i, P in enumerate(loops):
        if area(P) < 4:
            continue
        if is_hole(P, field):
            body.append(simplify(shear(resample(P, 1.0)[0])))
        else:
            body.append(simplify(add_fur(shear(P), lod, field, seed=3.0 + i)))

    grow = 1.12 if lod_name == "S" else 1.0          # open counters at tiny sizes
    holes = [shear(ellipse(cx, 0, COUNTER["a"] * grow, COUNTER["b"] * grow, COUNTER["rot"]))
             for cx in o_centres]
    mark = shear(astroid(cat_x, MARK["y"], MARK["r"])) if lod["mark"] else None
    return body, holes, mark


def path(P):
    return "M" + "L".join(f"{x:.1f} {-y:.1f}" for x, y in P) + "Z"


def bounds(loops, pad):
    A = np.vstack(loops)
    x0, x1, y0, y1 = A[:, 0].min(), A[:, 0].max(), -A[:, 1].max(), -A[:, 1].min()
    return x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad


def svg(kind, lod_name, ink=INK, bg=None, mark_fill=None, pad=None, radius=0):
    body, holes, mark = build(kind, lod_name)
    pad = (60 if kind == "icon" else 24) if pad is None else pad
    x, y, w, h = bounds(body, pad)
    if kind == "icon":                                # square canvas
        side = max(w, h)
        x, y, w, h = x - (side - w) / 2, y - (side - h) / 2, side, side
    d = " ".join(path(P) for P in body + holes)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:.1f} {y:.1f} {w:.1f} {h:.1f}">']
    if bg:
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                   f'rx="{radius * w:.1f}" fill="{bg}"/>')
    if mark is not None and mark_fill is None:
        d += " " + path(mark)                         # knocked out
    out.append(f'<path fill="{ink}" fill-rule="evenodd" d="{d}"/>')
    if mark is not None and mark_fill:
        out.append(f'<path fill="{mark_fill}" d="{path(mark)}"/>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "out")
    os.makedirs(out, exist_ok=True)
    files = {}
    for lod in LOD:
        files[f"sooto-wordmark-{lod}.svg"] = svg("wordmark", lod)
        files[f"sooto-icon-{lod}.svg"] = svg("icon", lod, bg=PURPLE, radius=0.22)
    files["sooto-wordmark-L-dark.svg"] = svg("wordmark", "L", ink=PAPER, bg=INK,
                                             mark_fill=MOSS, pad=60)
    files["sooto-icon-L-dark.svg"] = svg("icon", "L", ink=PAPER, bg=INK, mark_fill=MOSS,
                                         radius=0.22)
    for name, text in files.items():
        with open(os.path.join(out, name), "w") as f:
            f.write(text + "\n")

    def img(name, width):
        return f'<img src="{name}" width="{width}" alt="{name}">'

    html = f"""<!doctype html><meta charset="utf-8"><title>Sooto logo sheet</title>
<style>body{{margin:0;padding:48px;background:{PAPER};font:13px/1.4 system-ui;color:{INK}}}
.row{{display:flex;align-items:center;gap:40px;margin:28px 0;flex-wrap:wrap}}
small{{display:block;opacity:.6;margin-top:6px}}</style>
<div class="row">{img("sooto-wordmark-L.svg", 900)}</div>
<div class="row">
<div>{img("sooto-wordmark-L.svg", 420)}<small>L · 420px</small></div>
<div>{img("sooto-wordmark-M.svg", 220)}<small>M · 220px</small></div>
<div>{img("sooto-wordmark-S.svg", 96)}<small>S · 96px</small></div>
<div>{img("sooto-wordmark-S.svg", 56)}<small>S · 56px</small></div></div>
<div class="row">{img("sooto-wordmark-L-dark.svg", 460)}
{img("sooto-icon-L.svg", 180)}{img("sooto-icon-L-dark.svg", 180)}
<div>{img("sooto-icon-M.svg", 64)}<small>M · 64</small></div>
<div>{img("sooto-icon-S.svg", 32)}<small>S · 32</small></div>
<div>{img("sooto-icon-S.svg", 16)}<small>S · 16</small></div></div>
"""
    with open(os.path.join(out, "sheet.html"), "w") as f:
        f.write(html)
    print("wrote", len(files) + 1, "files to", out)


if __name__ == "__main__":
    main()
