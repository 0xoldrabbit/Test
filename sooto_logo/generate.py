#!/usr/bin/env python3
"""Lay out and export the Sooto logo from shape.json (see trace.py).

Layout rules (all lengths in units of U = width of the bare "o" body)
  size     the three o's share one body width U and one counter ellipse
  rails    letters alternate between two baselines:  S o o t o  ->  low high low high low
           the high rail sits STEP * U above the low one
  spacing  neighbours overlap by the same horizontal amount  GAP * U  in the body zone,
           but stay CLEAR * U apart above the x-height so ears and the t bar read
  export   outlines are refitted with cubic Beziers, corners (fur tips) kept sharp

Run:  python3 trace.py && python3 generate.py     -> ./out
"""
import json
import math
import os
import numpy as np
from geom import area, resample
from trace import find_corners, DS

HERE = os.path.dirname(os.path.abspath(__file__))
INK, PAPER, PURPLE, MOSS = "#1f1b16", "#fbece8", "#ae8bd0", "#c9f2d3"

LETTERS = ("S", "o1", "o2", "t", "o3")
RAIL = dict(S=0, o1=1, o2=0, t=1, o3=0)       # 0 = low rail, 1 = high rail
STEP = 0.15                                   # rail distance / U
GAP = -0.035                                  # horizontal gap between neighbours / U
XHEIGHT = 0.88                                # body rows = this far up from a letter's foot / U
CLEAR = 0.03                                  # minimum gap above the x-height / U (ears, t bar)
FIT_TOL = 0.30                                # Bezier fitting tolerance, px
MARK_LEVELS = ("L", "M")


# -------------------------------------------------------------------- layout
def row_extent(P, side):
    """Left-most (side=-1) or right-most (side=+1) x of a loop for every integer row."""
    Q = resample(P, ds=0.4)
    rows = np.round(Q[:, 1]).astype(int)
    out = {}
    for y, x in zip(rows, Q[:, 0]):
        if y not in out or side * x > side * out[y]:
            out[y] = x
    return out


def gap(A, B, U):
    """Smallest horizontal distance from A (left) to B (right), negative = overlap.
    Returned separately for B's body rows and for the rows above its x-height."""
    ra, lb = row_extent(A, +1), row_extent(B, -1)
    split = B[:, 1].max() - XHEIGHT * U
    rows = ra.keys() & lb.keys()
    body = min(lb[y] - ra[y] for y in rows if y >= split)
    above = min((lb[y] - ra[y] for y in rows if y < split), default=np.inf)
    return body, above


def layout(d):
    """Affine transform (scale, centre, shift) per letter, derived from the bare body."""
    body = {k: np.array(v) for k, v in d["letters"]["S"].items()}
    o_keys = ("o1", "o2", "o3")
    centre = {k: np.array([c["cx"], c["cy"]]) for k, c in zip(o_keys, d["counters"])}
    width = {k: np.ptp(body[k][:, 0]) for k in o_keys}
    U = float(np.mean(list(width.values())))
    T = {k: dict(s=1.0, c=np.zeros(2), shift=np.zeros(2)) for k in LETTERS}
    for k in o_keys:
        T[k].update(s=U / width[k], c=centre[k])

    def place(k, P):
        t = T[k]
        return (P - t["c"]) * t["s"] + t["c"] + t["shift"]

    low = max(place(k, body[k])[:, 1].max() for k in LETTERS if RAIL[k] == 0)
    report = []
    for k in LETTERS:                                     # rails
        bottom = place(k, body[k])[:, 1].max()
        T[k]["shift"][1] = low - RAIL[k] * STEP * U - bottom
    for a, b in zip(LETTERS[:-1], LETTERS[1:]):           # spacing, left to right
        g_body, g_above = gap(place(a, body[a]), place(b, body[b]), U)
        T[b]["shift"][0] = max(GAP * U - g_body, CLEAR * U - g_above)
        report.append((a + "|" + b, g_body, g_above))
    return U, T, place, report


def counters(d, U, place, grow=1.0):
    cs = d["counters"]
    a = np.mean([c["a"] for c in cs]) * grow
    b = np.mean([c["b"] for c in cs]) * grow
    rot = math.radians(np.mean([c["rot"] for c in cs]))
    t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    x, y = a * np.cos(t), b * np.sin(t)
    E = np.column_stack([x * math.cos(rot) - y * math.sin(rot),
                         x * math.sin(rot) + y * math.cos(rot)])
    return {k: place(k, np.array([[c["cx"], c["cy"]]]))[0] + E
            for k, c in zip(("o1", "o2", "o3"), cs)}


def astroid(m, n=96):
    """|x|^(2/3) + |y|^(2/3) = r^(2/3)"""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([m["cx"] + m["r"] * np.cos(t) ** 3, m["cy"] + m["r"] * np.sin(t) ** 3])


# ------------------------------------------------------------- Bezier fitting
def unit(v):
    return v / (np.hypot(*v) + 1e-12)


def bezier(c, u):
    u = u[:, None]
    return ((1 - u) ** 3 * c[0] + 3 * u * (1 - u) ** 2 * c[1] +
            3 * u * u * (1 - u) * c[2] + u ** 3 * c[3])


def fit_cubic(pts, t0, t1, out, depth=0):
    """Schneider-style least squares: end points and end tangents fixed, split on error."""
    p0, p3 = pts[0], pts[-1]
    chord = np.hypot(*(p3 - p0))
    seg = np.hypot(*np.diff(pts, axis=0).T)
    u = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-12)
    b1, b2 = 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u)
    a1, a2 = b1[:, None] * t0, b2[:, None] * t1
    rest = pts - ((1 - u) ** 3 + b1)[:, None] * p0 - (b2 + u ** 3)[:, None] * p3
    M = np.array([[(a1 * a1).sum(), (a1 * a2).sum()], [(a1 * a2).sum(), (a2 * a2).sum()]])
    rhs = np.array([(a1 * rest).sum(), (a2 * rest).sum()])
    try:
        l1, l2 = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        l1 = l2 = -1
    if l1 < 0.02 * chord or l2 < 0.02 * chord or max(l1, l2) > 3 * seg.sum():
        l1 = l2 = seg.sum() / 3
    ctrl = np.array([p0, p0 + l1 * t0, p3 + l2 * t1, p3])
    err = np.hypot(*(bezier(ctrl, u) - pts).T)
    k = int(np.argmax(err))
    if err[k] <= FIT_TOL or len(pts) < 8 or depth > 12:
        out.append(ctrl)
        return
    k = min(max(k, 3), len(pts) - 4)
    tm = unit(pts[k + 2] - pts[k - 2])
    fit_cubic(pts[:k + 1], t0, -tm, out, depth + 1)
    fit_cubic(pts[k:], tm, t1, out, depth + 1)


def fit_loop(P):
    """Closed loop -> list of cubic segments, breaking only at sharp corners."""
    P = resample(P, ds=DS)
    corners = list(find_corners(P, 50.0))
    if len(corners) < 2:                                   # smooth loop: break at extremes
        corners = sorted({int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0])),
                          int(np.argmin(P[:, 1])), int(np.argmax(P[:, 1]))})
    out = []
    for i, j in zip(corners, corners[1:] + [corners[0] + len(P)]):
        pts = P[np.arange(i, j + 1) % len(P)]
        if len(pts) < 4:
            out.append(np.array([pts[0], pts[0], pts[-1], pts[-1]]))
            continue
        m = min(4, len(pts) - 1)
        fit_cubic(pts, unit(pts[m] - pts[0]), unit(pts[-1 - m] - pts[-1]), out)
    return out


def path(P, sign):
    """SVG path of loop P, wound so that its signed area has the given sign."""
    if area(P) * sign < 0:
        P = P[::-1]
    segs = fit_loop(P)
    d = f"M{segs[0][0][0]:.1f} {segs[0][0][1]:.1f}"
    for c in segs:
        d += "C" + " ".join(f"{v:.1f}" for v in c[1:].ravel())
    return d + "Z", len(segs)


# -------------------------------------------------------------------- export
def compose(d, level, which=LETTERS):
    U, T, place, _ = layout(d)
    loops = [place(k, np.array(d["letters"][level][k])) for k in which]
    grow = 1.10 if level == "S" else 1.0
    holes = [E for k, E in counters(d, U, place, grow).items() if k in which]
    mark = None
    if level in MARK_LEVELS and "S" in which:
        m = d["mark"]
        mark = astroid(dict(m, cx=m["cx"] + T["S"]["shift"][0], cy=m["cy"] + T["S"]["shift"][1]))
    return loops, holes, mark


def svg(loops, holes, mark, ink=INK, bg=None, mark_fill=None, pad=20, square=False, radius=0):
    A = np.vstack(loops)
    x, y = A.min(axis=0) - pad
    w, h = np.ptp(A, axis=0) + 2 * pad
    if square:
        side = max(w, h)
        x, y, w, h = x - (side - w) / 2, y - (side - h) / 2, side, side
    parts = [path(P, +1) for P in loops] + [path(P, -1) for P in holes]
    if mark is not None and not mark_fill:
        parts.append(path(mark, -1))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:.1f} {y:.1f} {w:.1f} {h:.1f}">']
    if bg:
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                   f'rx="{radius * w:.1f}" fill="{bg}"/>')
    out.append(f'<path fill="{ink}" fill-rule="nonzero" d="{" ".join(p for p, _ in parts)}"/>')
    if mark is not None and mark_fill:
        out.append(f'<path fill="{mark_fill}" d="{path(mark, +1)[0]}"/>')
    out.append("</svg>")
    return "\n".join(out), sum(n for _, n in parts)


def main():
    with open(os.path.join(HERE, "shape.json")) as f:
        d = json.load(f)
    U, T, _, report = layout(d)
    print(f"U = {U:.1f}px   rail step = {STEP * U:.1f}px   target gap = {GAP * U:.1f}px")
    for pair, g_body, g_above in report:
        print(f"  gap {pair:6s} before: body {g_body:6.1f}  above x-height {g_above:6.1f}")
    for k in LETTERS:
        print(f"  {k:3s} scale {T[k]['s']:.3f}  shift {T[k]['shift'].round(1)}")

    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    files = {}
    for lv in ("L", "M", "S"):
        files[f"sooto-wordmark-{lv}.svg"] = svg(*compose(d, lv))
        files[f"sooto-icon-{lv}.svg"] = svg(*compose(d, lv, ("o3",)), bg=PURPLE, pad=55,
                                            square=True, radius=0.22)
    files["sooto-wordmark-L-dark.svg"] = svg(*compose(d, "L"), ink=PAPER, bg=INK,
                                             mark_fill=MOSS, pad=50)
    files["sooto-icon-L-dark.svg"] = svg(*compose(d, "L", ("o3",)), ink=PAPER, bg=INK,
                                         pad=55, square=True, radius=0.22)
    for name, (text, n) in files.items():
        with open(os.path.join(out, name), "w") as f:
            f.write(text + "\n")
        print(f"  {name:28s} {n:4d} Bezier segments")

    def img(name, width, label=None):
        cap = f"<small>{label}</small>" if label else ""
        return f'<div><img src="{name}" width="{width}">{cap}</div>'

    x0, y0, x1, y1 = d["crop"]
    k = 500 / (x1 - x0)
    html = f"""<!doctype html><meta charset="utf-8"><title>Sooto logo sheet</title>
<style>body{{margin:0;padding:40px;background:{PAPER};font:13px/1.4 system-ui;color:{INK}}}
.row{{display:flex;align-items:flex-end;gap:36px;margin:26px 0;flex-wrap:wrap}}
small{{display:block;opacity:.6;margin-top:6px}}
.ref{{width:500px;height:{k * (y1 - y0):.0f}px;background:url(../reference.jpg)
 -{x0 * k:.0f}px -{y0 * k:.0f}px/{1152 * k:.0f}px}}</style>
<div class="row"><div><div class="ref"></div><small>reference (raster)</small></div>
{img("sooto-wordmark-L.svg", 500, "regularised · L")}</div>
<div class="row">{img("sooto-wordmark-L.svg", 1000)}</div>
<div class="row">{img("sooto-wordmark-L.svg", 400, "L · 400px")}
{img("sooto-wordmark-M.svg", 220, "M · 220px")}{img("sooto-wordmark-S.svg", 110, "S · 110px")}
{img("sooto-wordmark-S.svg", 60, "S · 60px")}</div>
<div class="row">{img("sooto-wordmark-L-dark.svg", 460)}{img("sooto-icon-L.svg", 170)}
{img("sooto-icon-L-dark.svg", 170)}{img("sooto-icon-M.svg", 64, "M · 64")}
{img("sooto-icon-S.svg", 32, "S · 32")}{img("sooto-icon-S.svg", 16, "S · 16")}</div>
"""
    with open(os.path.join(out, "sheet.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
