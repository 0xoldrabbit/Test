#!/usr/bin/env python3
"""Vectorise reference.jpg into clean math primitives -> shape.json

  outline  : iso-contour of the luminance (marching squares, 3x supersampled)
  corners  : points where the outline turns sharply (fur tips, notches, letter joints)
  clean    : outline smoothed by pinned diffusion, corners held fixed, so curves get
             cleaner while every fur tip stays sharp
  levels   : morphological opening of the ink with a disk of radius r erases fur
             thinner than 2r; one knob gives the optical sizes
                 L  r = 0     every fur tip
                 M  r = 3.5   only the big tufts
                 S  r = 16    bare body
  counters : ellipse fitted to each hole by second moments
  mark     : astroid fitted to the sparkle
All coordinates are reference-image pixels (y down).
"""
import json
import os
import numpy as np
from PIL import Image
from geom import marching_squares, area, resample, gauss_closed, lowpass_closed

HERE = os.path.dirname(os.path.abspath(__file__))
CROP = (60, 70, 1100, 520)      # the large wordmark
UP = 3                          # supersampling
DS = 0.5                        # outline sample spacing, reference px
CORNER = dict(reach=3.0, angle=50.0)        # px either side, minimum turn in degrees
SMOOTH = dict(L=2.2, M=3.0, S=8.0)          # diffusion sigma, reference px
OPEN_R = dict(L=0.0, M=3.5, S=16.0)
# the bare body keeps only undulations longer than this (px); t bar and ears need a finer one
# a strand = outline lifting more than `lift` px off the bare body; drop the lowest fraction.
# then a flick = tip and notch within `pair` px with a back shorter than `reach`; drop `teeth` of them
FUR = dict(lift=1.5, drop=0.35, teeth=0.5, pair=22.0, reach=140.0)
BODY_CUTOFF = dict(S=85.0, o1=85.0, o2=85.0, t=42.0, o3=62.0)


def disk_filter(M, r):
    """Fraction of a radius-r disk covered by M, via FFT convolution."""
    h, w = M.shape
    yy, xx = np.ogrid[:h, :w]
    yy = np.minimum(yy, h - yy); xx = np.minimum(xx, w - xx)
    D = (xx * xx + yy * yy <= r * r).astype(float)
    return np.fft.irfft2(np.fft.rfft2(M) * np.fft.rfft2(D), s=M.shape) / D.sum()


def opening(ink, r):
    eroded = (disk_filter(ink, r) > 1 - 1e-6).astype(float)
    return (disk_filter(eroded, r) > 1e-6).astype(float)


def find_corners(P, min_angle):
    """Indices where the outline turns by more than min_angle within CORNER[reach]."""
    k = int(round(CORNER["reach"] / DS))
    a = P - np.roll(P, k, axis=0)
    b = np.roll(P, -k, axis=0) - P
    turn = np.abs(np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
                                        (a * b).sum(axis=1))))
    n, w = len(P), 2 * k
    ext = np.concatenate([turn[-w:], turn, turn[:w]])
    win = np.lib.stride_tricks.sliding_window_view(ext, 2 * w + 1)
    return np.flatnonzero((turn >= win.max(axis=1)) & (turn > min_angle))


def smooth_pinned(P, corners, sigma):
    """Heat diffusion along the curve with the corner points clamped."""
    if len(corners) == 0:
        return gauss_closed(P, sigma / DS)
    P = P.copy()
    pinned = P[corners].copy()
    for _ in range(int(2 * (sigma / DS) ** 2)):
        P = 0.25 * np.roll(P, 1, axis=0) + 0.5 * P + 0.25 * np.roll(P, -1, axis=0)
        P[corners] = pinned
    return P


def fit_ellipse(P):
    P = resample(P, n=400)
    c = P.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((P - c).T))
    rot = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])) % 180
    return dict(cx=c[0], cy=c[1], a=np.sqrt(2 * evals[1]), b=np.sqrt(2 * evals[0]), rot=rot)


def rnd(d):
    return {k: round(float(v), 2) for k, v in d.items()}


def main():
    im = Image.open(os.path.join(HERE, "reference.jpg")).convert("L").crop(CROP)
    im = im.resize((im.width * UP, im.height * UP), Image.BICUBIC)
    G = disk_filter(np.asarray(im, float), 1.5)             # tame JPEG noise
    thr = 0.5 * (np.percentile(G, 2) + np.percentile(G, 98))
    ink = (G < thr).astype(float)
    to_ref = lambda P: P / UP + np.array(CROP[:2])

    def loops_of(field, level):
        out = [to_ref(L) for L in marching_squares(field, level)]
        return sorted([L for L in out if abs(area(L)) > 40], key=lambda L: -abs(area(L)))

    def is_hole(L):
        """Step 2px from the boundary toward the loop's interior: paper means hole."""
        P = resample(L, n=64)
        t = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
        inward = np.column_stack([-t[:, 1], t[:, 0]]) * np.sign(area(P))
        q = (P + 2 * inward / np.hypot(*inward.T)[:, None] - np.array(CROP[:2])) * UP
        return ink[q[:, 1].astype(int), q[:, 0].astype(int)].mean() < 0.5

    raw = loops_of(G, thr)
    holes = [L for L in raw if is_hole(L)]
    counters = sorted([rnd(fit_ellipse(L)) for L in holes if abs(area(L)) > 800],
                      key=lambda c: c["cx"])
    star = resample(min(holes, key=lambda L: abs(area(L))), n=200)
    mark = rnd(dict(cx=star[:, 0].mean(), cy=star[:, 1].mean(),
                    r=np.abs(star - star.mean(axis=0)).max()))

    levels = {}
    for name, r in OPEN_R.items():
        if r > 0:
            pieces = [L for L in loops_of(disk_filter(opening(ink, r * UP), 1.5), 0.5)
                      if not is_hole(L)]
        else:
            pieces = [L for L in raw if not is_hole(L)]
        out = []
        for L in pieces:
            P = resample(L if area(L) > 0 else L[::-1], ds=DS)
            out.append(smooth_pinned(P, find_corners(P, MIN_TURN[name]), SMOOTH[name]))
        levels[name] = sorted(out, key=lambda P: P[:, 0].min())      # Soo, t, cat-o

    counts = [thin_fur(levels["L"], levels["S"], i) for i in range(len(levels["L"]))]
    print("fur strands removed / total:", sum(c[0] for c in counts), "/", sum(c[1] for c in counts))
    counts = [thin_teeth(levels["L"], i) for i in range(len(levels["L"]))]
    print("flicks removed / total:", sum(c[0] for c in counts), "/", sum(c[1] for c in counts))

    joints = crevices(levels["S"][0])
    letters = {}
    for name, (soo, t, cat) in levels.items():
        S, o1, o2 = split_soo(soo, joints)
        if name == "S":       # letters are separate now, so the body can be ironed flat
            S, o1, o2, t, cat = (lowpass_closed(resample(P, ds=DS), BODY_CUTOFF[k], DS)
                                 for k, P in zip(LETTERS, (S, o1, o2, t, cat)))
        letters[name] = {k: np.round(v, 2).tolist()
                         for k, v in zip(LETTERS, (S, o1, o2, t, cat))}

    with open(os.path.join(HERE, "shape.json"), "w") as f:
        json.dump(dict(crop=CROP, letters=letters, counters=counters, mark=mark), f)
    print("joints", joints.round(1).tolist())
    for k in LETTERS:
        P = np.array(letters["S"][k])
        print(k, "x", P[:, 0].min(), P[:, 0].max(), "y", P[:, 1].min(), P[:, 1].max())
    for c in counters:
        print("counter", c)
    print("mark", mark)


def thin_fur(fur, bare, i):
    """Drop the shortest strands of piece i: where the furry outline leaves the bare body
    for a short, low excursion that ends in a sharp tip, follow the body instead."""
    P, B = fur[i], bare[i]
    step = 4
    near = np.empty(len(P), int)
    dist = np.empty(len(P))
    for a in range(0, len(P), 500):                       # nearest body point, chunked
        D = np.hypot(P[a:a + 500, None, 0] - B[None, ::step, 0],
                     P[a:a + 500, None, 1] - B[None, ::step, 1])
        near[a:a + 500] = D.argmin(axis=1) * step
        dist[a:a + 500] = D.min(axis=1)
    off = dist > FUR["lift"]
    shift = int(np.argmin(off))                           # start outside any strand
    idx = (np.arange(len(P)) + shift) % len(P)
    edges = np.flatnonzero(np.diff(off[idx].astype(int)))
    tips = set(find_corners(P, 50.0).tolist())
    runs = []
    for a, b in zip(edges[::2], edges[1::2] + 1):         # a, b: last/first body-hugging points
        span = idx[a:b + 1]
        if tips & set(span.tolist()):                     # ears and t lobes have no sharp tip
            runs.append((dist[span].max(), a, b))
    if not runs:
        return 0, 0
    limit = np.quantile([r[0] for r in runs], FUR["drop"])
    keep = np.ones(len(P), bool)
    patches = {}
    for h, a, b in runs:
        j0, j1 = near[idx[a]], near[idx[b]]
        n = (j1 - j0) % len(B)
        if h <= limit and n < len(B) // 4:
            keep[idx[a + 1:b]] = False
            patches[idx[a]] = B[(j0 + np.arange(1, n)) % len(B)]
    out = []
    for k in range(len(P)):
        if keep[k]:
            out.append(P[k:k + 1])
        if k in patches:
            out.append(patches[k])
    fur[i] = smooth_pinned(resample(np.vstack(out), ds=DS),
                           find_corners(resample(np.vstack(out), ds=DS), 50.0), 1.0)
    return len(patches), len(runs)


def thin_teeth(fur, i):
    """Remove the smallest remaining flicks, never two neighbours.
    A flick is a tip and a notch close together plus one long smooth back, so the
    outline from the notch across the back to the next corner is replaced by a
    Hermite curve that continues the directions on either side."""
    P = fur[i]
    n = len(P)
    turn = signed_turn(P, int(round(CORNER["reach"] / DS)))
    corners = [int(c) for c in find_corners(P, 50.0)]
    m = len(corners)
    near, far = int(2 / DS), int(9 / DS)
    flicks = []
    for c in range(m):
        w, a, b, x = corners[c - 1], corners[c], corners[(c + 1) % m], corners[(c + 2) % m]
        if turn[a] * turn[b] >= 0 or (b - a) % n > FUR["pair"] / DS:
            continue
        # notch first: the back runs on to x.  tip first: the back came from w.
        p0, p1 = (a, x) if turn[a] < 0 else (w, b)
        if (p1 - p0) % n > FUR["reach"] / DS:
            continue
        tip = b if turn[a] < 0 else a
        ab, v = P[p1] - P[p0], P[tip] - P[p0]
        flicks.append((abs(ab[0] * v[1] - ab[1] * v[0]) / np.hypot(*ab), p0, p1, turn[a] < 0))
    used, patches = set(), {}
    keep = np.ones(n, bool)
    for h, p0, p1, notch_first in sorted(flicks):
        if len(patches) >= FUR["teeth"] * len(flicks):
            break
        if p0 in used or p1 in used:
            continue
        used |= {p0, p1}
        t0 = P[(p0 - near) % n] - P[(p0 - far) % n]
        t1 = (P[(p1 - near) % n] - P[(p1 - far) % n]) if notch_first else \
             (P[(p1 + far) % n] - P[(p1 + near) % n])
        hlen = np.hypot(*(P[p1] - P[p0])) / 3
        c0 = P[p0] + hlen * t0 / np.hypot(*t0)
        c1 = P[p1] - hlen * t1 / np.hypot(*t1)
        u = np.linspace(0, 1, 60)[1:-1, None]
        patches[p0] = ((1 - u) ** 3 * P[p0] + 3 * u * (1 - u) ** 2 * c0 +
                       3 * u * u * (1 - u) * c1 + u ** 3 * P[p1])
        keep[(p0 + 1 + np.arange((p1 - p0 - 1) % n)) % n] = False
    out = []
    for k in range(n):
        if keep[k]:
            out.append(P[k:k + 1])
        if k in patches:
            out.append(patches[k])
    Q = resample(np.vstack(out), ds=DS)
    fur[i] = smooth_pinned(Q, find_corners(Q, 50.0), 1.0)
    return len(patches), len(flicks)


LETTERS = ("S", "o1", "o2", "t", "o3")
MIN_TURN = dict(L=50.0, M=50.0, S=95.0)


def signed_turn(P, k):
    a = P - np.roll(P, k, axis=0)
    b = np.roll(P, -k, axis=0) - P
    return np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1)))


def crevices(P):
    """The four sharp concave corners where S|o and o|o meet (P is positively wound)."""
    k = int(round(CORNER["reach"] / DS))
    turn = signed_turn(P, k)
    idx = [i for i in find_corners(P, 95.0) if turn[i] < 0]
    assert len(idx) == 4, idx
    return P[idx]


def bridge(run_a, run_b):
    """Hidden part of a letter under its neighbour: cubic Bezier continuing both tangents."""
    near, far = int(4 / DS), int(18 / DS)
    p0, p1 = run_a[-1], run_b[0]
    t0 = run_a[-near] - run_a[-far]
    t1 = run_b[far] - run_b[near]
    h = 0.5 * np.hypot(*(p1 - p0))
    c0 = p0 + h * t0 / np.hypot(*t0)
    c1 = p1 - h * t1 / np.hypot(*t1)
    u = np.linspace(0, 1, 40)[1:-1, None]
    return (1 - u) ** 3 * p0 + 3 * u * (1 - u) ** 2 * c0 + 3 * u * u * (1 - u) * c1 + u ** 3 * p1


def split_soo(P, joints):
    """Cut the fused S-o-o piece at the joints and complete each letter."""
    i0, i1, i2, i3 = sorted(int(np.argmin(np.hypot(*(P - j).T))) for j in joints)
    outer = np.vstack([P[i3:], P[:i0 + 1]])          # joints nest like brackets
    mid_a, mid_b = P[i0:i1 + 1], P[i2:i3 + 1]
    inner = P[i1:i2 + 1]
    parts = [np.vstack([outer, bridge(outer, outer)]),
             np.vstack([mid_a, bridge(mid_a, mid_b), mid_b, bridge(mid_b, mid_a)]),
             np.vstack([inner, bridge(inner, inner)])]
    return sorted(parts, key=lambda L: L[:, 0].mean())


if __name__ == "__main__":
    main()
