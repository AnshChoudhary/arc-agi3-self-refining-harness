"""Default per-step analysis. `analyze()` runs in the REPL namespace before every
decision and its output is shown to the student. Tools from harness/tools/ are
already loaded, as are frames, curr, prev, steps, etc."""


def analyze():
    lines = []
    counts = color_counts(curr)
    bg = max(counts, key=counts.get)
    lines.append(f"colours (count): {counts}; background={bg}")
    if len(frames) > 1:
        d = diff(prev, curr)
        if d["count"] == 0:
            lines.append("last action: no pixel changed")
        else:
            lines.append(f"last action: {d['count']} px changed in x[{d['x0']}..{d['x1']}] y[{d['y0']}..{d['y1']}]; "
                         f"transitions old->new: {d['transitions']}")
    comps = components(curr)
    comps.sort(key=lambda c: c["size"])
    lines.append(f"{len(comps)} non-background components (smallest first, up to 12):")
    for c in comps[:12]:
        lines.append(f"  colour {c['color']} size {c['size']} bbox x[{c['x0']}..{c['x1']}] y[{c['y0']}..{c['y1']}] centre ({c['cx']},{c['cy']})")
    return "\n".join(lines)
