"""Scene helpers exposed in the REPL. numpy only; frames are 64x64 int arrays, index [y, x]."""


def color_counts(frame):
    """{colour: pixel count} for a frame."""
    vals, counts = np.unique(frame, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def diff(a, b):
    """Where frames a and b differ: count, bbox (x0,y0,x1,y1) and old->new colour transition counts."""
    changed = np.argwhere(a != b)
    if len(changed) == 0:
        return {"count": 0, "x0": None, "y0": None, "x1": None, "y1": None, "transitions": {}}
    ys, xs = changed[:, 0], changed[:, 1]
    trans = {}
    for y, x in changed:
        k = f"{int(a[y, x])}->{int(b[y, x])}"
        trans[k] = trans.get(k, 0) + 1
    return {"count": int(len(changed)), "x0": int(xs.min()), "y0": int(ys.min()),
            "x1": int(xs.max()), "y1": int(ys.max()), "transitions": trans}


def components(frame, color=None, connectivity=8):
    """Connected components as dicts: color, size, bbox x0,y0,x1,y1, centre cx,cy.
    Skips the background (most common colour) unless `color` is given."""
    h, w = frame.shape
    if color is None:
        counts = color_counts(frame)
        colors = [c for c in counts if c != max(counts, key=counts.get)]
    else:
        colors = [color]
    out = []
    for c in colors:
        mask = frame == c
        seen = np.zeros_like(mask, dtype=bool)
        for y0, x0 in np.argwhere(mask):
            if seen[y0, x0]:
                continue
            stack, pts = [(int(y0), int(x0))], []
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if connectivity == 4 and dy and dx:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            out.append({"color": int(c), "size": len(pts), "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys),
                        "cx": (min(xs) + max(xs)) // 2, "cy": (min(ys) + max(ys)) // 2})
    return out


def show(frame=None, x0=0, y0=0, x1=64, y1=64):
    """Print a region of a frame (default: the current frame) as one hex digit per cell with row/column labels."""
    f = curr if frame is None else frame
    digits = "0123456789ABCDEF"
    x1, y1 = min(x1, f.shape[1]), min(y1, f.shape[0])
    print("    " + "".join(str(x % 10) for x in range(x0, x1)))
    for y in range(y0, y1):
        print(f"{y:3d} " + "".join(digits[int(v) % 16] for v in f[y, x0:x1]))
