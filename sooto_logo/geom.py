"""Shared geometry helpers (numpy only)."""
import numpy as np


def marching_squares(F, level=0.0):
    """Closed iso-contours of grid F (row = y, col = x), linear interpolation."""
    F = F - level
    ins = F < 0
    pts, links = {}, {}

    def edge_pt(i, j, kind):
        key = (i, j, kind)
        if key not in pts:
            i2, j2 = (i, j + 1) if kind == "h" else (i + 1, j)
            a, b = F[i, j], F[i2, j2]
            t = a / (a - b)
            pts[key] = (j + t * (j2 - j), i + t * (i2 - i))
        return key

    def link(a, b):
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)

    mixed = (ins[:-1, :-1] != ins[1:, 1:]) | (ins[:-1, 1:] != ins[1:, :-1]) | \
            (ins[:-1, :-1] != ins[:-1, 1:])
    for i, j in zip(*np.nonzero(mixed)):
        a, b, c, d = ins[i, j], ins[i, j + 1], ins[i + 1, j + 1], ins[i + 1, j]
        e = []
        if a != b: e.append(edge_pt(i, j, "h"))
        if b != c: e.append(edge_pt(i, j + 1, "v"))
        if c != d: e.append(edge_pt(i + 1, j, "h"))
        if d != a: e.append(edge_pt(i, j, "v"))
        if len(e) == 2:
            link(e[0], e[1])
        elif len(e) == 4:
            if (F[i:i + 2, j:j + 2].mean() < 0) == a:
                link(e[0], e[1]); link(e[2], e[3])
            else:
                link(e[0], e[3]); link(e[1], e[2])

    loops, seen = [], set()
    for start in links:
        if start in seen or len(links[start]) != 2:
            continue
        loop, prev, cur = [], None, start
        while cur not in seen:
            seen.add(cur)
            loop.append(pts[cur])
            nxt = [n for n in links[cur] if n != prev]
            prev, cur = cur, (nxt[0] if nxt else start)
        loops.append(np.array(loop))
    return loops


def area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def length(P):
    return np.hypot(*(np.roll(P, -1, axis=0) - P).T).sum()


def resample(P, ds=None, n=None):
    """Uniform arclength resampling of a closed loop."""
    Q = np.vstack([P, P[:1]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    n = n or max(8, int(round(s[-1] / ds)))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])


def gauss_closed(v, sigma):
    """Gaussian low-pass of a periodic signal == Gaussian taper on its Fourier series."""
    k = np.fft.fftfreq(len(v)) * 2 * np.pi
    w = np.exp(-0.5 * (k * sigma) ** 2)
    return np.real(np.fft.ifft(np.fft.fft(v, axis=0) * (w[:, None] if v.ndim > 1 else w), axis=0))


def lowpass_closed(P, cutoff, ds, order=3):
    """Butterworth low-pass along arclength: removes ripples shorter than `cutoff`
    without shrinking the larger shape (unlike a Gaussian)."""
    f = np.abs(np.fft.fftfreq(len(P), d=ds))
    H = 1.0 / (1.0 + (f * cutoff) ** (2 * order))
    return np.real(np.fft.ifft(np.fft.fft(P, axis=0) * H[:, None], axis=0))


def centroid(P):
    """Area centroid of a closed polygon."""
    x, y = P[:, 0], P[:, 1]
    c = x * np.roll(y, -1) - np.roll(x, -1) * y
    return np.array([((x + np.roll(x, -1)) * c).sum(), ((y + np.roll(y, -1)) * c).sum()]) / (3 * c.sum())


def frame(P):
    """Unit tangent and left-hand normal of a closed loop."""
    t = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    t /= np.hypot(t[:, 0], t[:, 1])[:, None]
    return t, np.column_stack([t[:, 1], -t[:, 0]])


def simplify(P, tol):
    """Ramer-Douglas-Peucker on a closed loop."""
    keep = np.zeros(len(P), bool)
    h = len(P) // 2
    keep[[0, h, -1]] = True
    stack = [(0, h), (h, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, ab = P[i], P[j] - P[i]
        seg = P[i + 1:j] - a
        dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / (np.hypot(*ab) + 1e-12)
        m = int(np.argmax(dist))
        if dist[m] > tol:
            keep[i + 1 + m] = True
            stack += [(i, i + 1 + m), (i + 1 + m, j)]
    return P[keep]
