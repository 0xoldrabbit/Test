#!/usr/bin/env python3
"""Compose the Sooto logo files from shape.json (see trace.py).

  logo = outline at one fur level  -  counters (ellipses)  -  night mark (astroid)

Run:  python3 trace.py && python3 generate.py     -> ./out
"""
import json
import math
import os
import numpy as np
from geom import simplify

HERE = os.path.dirname(os.path.abspath(__file__))
INK, PAPER, PURPLE, MOSS = "#1f1b16", "#fbece8", "#ae8bd0", "#c9f2d3"

LOD = {"L": dict(mark=True), "M": dict(mark=True), "S": dict(mark=False)}
UNIFY_COUNTERS = True      # one aspect ratio and one tilt for all three o's


def ellipse(c, n=72):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    co, si = math.cos(math.radians(c["rot"])), math.sin(math.radians(c["rot"]))
    x, y = c["a"] * np.cos(t), c["b"] * np.sin(t)
    return np.column_stack([c["cx"] + x * co - y * si, c["cy"] + x * si + y * co])


def astroid(m, n=64):
    """|x|^(2/3) + |y|^(2/3) = r^(2/3)"""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([m["cx"] + m["r"] * np.cos(t) ** 3, m["cy"] + m["r"] * np.sin(t) ** 3])


def path(P):
    return "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in P) + "Z"


def load():
    with open(os.path.join(HERE, "shape.json")) as f:
        d = json.load(f)
    d["levels"] = {k: [np.array(p) for p in v] for k, v in d["levels"].items()}
    if UNIFY_COUNTERS:
        ratio = np.mean([c["b"] / c["a"] for c in d["counters"]])
        rot = np.mean([c["rot"] for c in d["counters"]])
        for c in d["counters"]:
            size = math.sqrt(c["a"] * c["b"])          # keep each counter's area
            c.update(a=size / math.sqrt(ratio), b=size * math.sqrt(ratio), rot=rot)
    return d


def compose(d, lod_name, cat_only=False):
    """Return (ink loops, mark loop or None). cat_only = just the eared o."""
    pieces = d["levels"][lod_name]
    cat_x = d["counters"][-1]["cx"]
    if cat_only:        # the piece around the last counter, cut free of its neighbours
        pieces = [p for p in pieces if p[:, 0].min() < cat_x < p[:, 0].max()]
    loops = [simplify(p, 0.06) for p in pieces]
    lo = np.min([L.min(axis=0) for L in loops], axis=0)
    hi = np.max([L.max(axis=0) for L in loops], axis=0)
    grow = 1.10 if lod_name == "S" else 1.0
    for c in d["counters"]:
        if lo[0] < c["cx"] < hi[0] and not (cat_only and c["cx"] != cat_x):
            loops.append(ellipse(dict(c, a=c["a"] * grow, b=c["b"] * grow)))
    m = d["mark"]
    mark = astroid(m) if LOD[lod_name]["mark"] and not cat_only else None
    return loops, mark


def svg(loops, mark, ink=INK, bg=None, mark_fill=None, pad=20, square=False, radius=0):
    A = np.vstack(loops)
    x, y = A.min(axis=0) - pad
    w, h = A.max(axis=0) - A.min(axis=0) + 2 * pad
    if square:
        side = max(w, h)
        x, y, w, h = x - (side - w) / 2, y - (side - h) / 2, side, side
    d = " ".join(path(P) for P in loops)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x:.1f} {y:.1f} {w:.1f} {h:.1f}">']
    if bg:
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                   f'rx="{radius * w:.1f}" fill="{bg}"/>')
    if mark is not None and not mark_fill:
        d += " " + path(mark)
    out.append(f'<path fill="{ink}" fill-rule="evenodd" d="{d}"/>')
    if mark is not None and mark_fill:
        out.append(f'<path fill="{mark_fill}" d="{path(mark)}"/>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    d = load()
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    files = {}
    for lod in LOD:
        files[f"sooto-wordmark-{lod}.svg"] = svg(*compose(d, lod))
        files[f"sooto-icon-{lod}.svg"] = svg(*compose(d, lod, cat_only=True), bg=PURPLE,
                                             pad=55, square=True, radius=0.22)
    files["sooto-wordmark-L-dark.svg"] = svg(*compose(d, "L"), ink=PAPER, bg=INK,
                                             mark_fill=MOSS, pad=50)
    files["sooto-icon-L-dark.svg"] = svg(*compose(d, "L", cat_only=True), ink=PAPER, bg=INK,
                                         pad=55, square=True, radius=0.22)
    for name, text in files.items():
        with open(os.path.join(out, name), "w") as f:
            f.write(text + "\n")

    def img(name, width, label=None):
        cap = f"<small>{label}</small>" if label else ""
        return f'<div><img src="{name}" width="{width}">{cap}</div>'

    x0, y0, x1, y1 = d["crop"]
    html = f"""<!doctype html><meta charset="utf-8"><title>Sooto logo sheet</title>
<style>body{{margin:0;padding:40px;background:{PAPER};font:13px/1.4 system-ui;color:{INK}}}
.row{{display:flex;align-items:flex-end;gap:36px;margin:26px 0;flex-wrap:wrap}}
small{{display:block;opacity:.6;margin-top:6px}}
.ref{{width:500px;height:{500 * (y1 - y0) / (x1 - x0):.0f}px;background:url(../reference.jpg)
 -{x0 * 500 / (x1 - x0):.0f}px -{y0 * 500 / (x1 - x0):.0f}px/{1152 * 500 / (x1 - x0):.0f}px}}</style>
<div class="row"><div><div class="ref"></div><small>reference (raster)</small></div>
{img("sooto-wordmark-L.svg", 500, "rebuilt · L")}</div>
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
    print("wrote", len(files) + 1, "files")


if __name__ == "__main__":
    main()
