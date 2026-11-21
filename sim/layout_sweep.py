"""
Layout sweep: find lift-magnet / floater / coil-ring layouts that hover LOW (good coil
authority) while keeping enough VERTICAL MARGIN (peak lift / weight) to survive knocks.

Reuses the model in levitation_sizing.py. For every lift layout x floater it finds the hover
point; layouts passing the margin filter are then evaluated with each coil candidate at
every feasible coil-ring radius.

Run:  uv run sim/layout_sweep.py            (~1-3 min, all cores)
      uv run sim/layout_sweep.py --quick    (smaller grid)
Writes sim/out/layout_sweep.csv and prints the best layouts per coil.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import levitation_sizing as s

MM = s.MM
MIN_MARGIN = 1.5  # peak lift / weight
MIN_GAP = 3.0 * MM  # floater bottom above tallest part
PLACE_CLEAR = 0.5 * MM  # body-to-body clearance on the PCA
SENSOR_KEEPOUT = 2.5 * MM  # free radius around the centre sensor

COIL_NAMES = ("RLB1314", "VLU1012", "VLU1415")


def layouts(quick: bool):
    discs = [(5, 2), (6, 2), (6, 3), (8, 2), (8, 3), (10, 3)]
    stacks = [1, 2, 3, 4]
    radii = [14, 17, 20, 23, 26, 30]
    floaters = [(15, 3), (20, 3), (20, 4), (25, 4), (30, 4)]
    bases = [0, 5, 10]  # lift stacks raised on spacers (mm) toward the coil tops
    if quick:
        discs, stacks, radii, floaters = discs[::2], [1, 3], radii[::2], floaters[1:4]
    for (dd, dh), n, r, (fd, fh), b in itertools.product(discs, stacks, radii, floaters, bases):
        yield dict(disc_d=dd, disc_h=dh, per_stack=n, radius=r, fl_d=fd, fl_h=fh, base=b)


def coil_ring_options(cand, layout):
    """Feasible coil-ring radii (mm): no coil/coil or coil/magnet overlap, sensor keep-out."""
    rd = cand.body_d / 2
    r_min = max(
        (cand.body_d + PLACE_CLEAR) / math.sqrt(2),  # neighbouring coils
        rd + SENSOR_KEEPOUT,  # centre sensor
    )
    ml = layout.radius / math.sqrt(2)
    opts = []
    for rc in np.arange(r_min, r_min + 8.01 * MM, 2 * MM):
        if math.hypot(rc - ml, ml) >= rd + layout.disc_d / 2 + PLACE_CLEAR:
            opts.append(rc)
    return opts


def evaluate(p):
    layout = s.LiftLayout(
        radius=p["radius"] * MM, disc_d=p["disc_d"] * MM, disc_h=p["disc_h"] * MM,
        per_stack=p["per_stack"], base_z=p["base"] * MM,
    )
    opt = s.FloaterOption(f"{p['fl_d']}x{p['fl_h']}", p["fl_d"] * MM, p["fl_h"] * MM)
    lift = s.build_lift(layout)
    floater = s.make_floater(opt, 120)
    weight = opt.mass * s.G

    z_floor = layout.top_z + 1 * MM + opt.height / 2
    zs = np.arange(z_floor, z_floor + 60 * MM, 1 * MM)
    fz = np.array([s.force_on(floater, lift, (0, 0, z))[2] for z in zs])
    if fz.max() / weight < MIN_MARGIN:
        return []
    # stable crossing on the falling side of the peak
    ipk = int(np.argmax(fz))
    z_hover = None
    for i in range(ipk, len(zs) - 1):
        if fz[i] >= weight > fz[i + 1]:
            lo, hi = zs[i], zs[i + 1]
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if s.force_on(floater, lift, (0, 0, mid))[2] >= weight else (lo, mid)
            z_hover = 0.5 * (lo + hi)
            break
    if z_hover is None:
        return []

    x1, x2 = 1 * MM, s.X_RECOVER
    f1 = s.force_on(floater, lift, (x1, 0, z_hover))[0]
    f2 = s.force_on(floater, lift, (x2, 0, z_hover))[0]
    k = f1 / x1
    if k <= 0:
        return []
    growth_ms = 1e3 / math.sqrt(k / opt.mass)

    rows = []
    for cand in s.COILS:
        if not any(n in cand.name for n in COIL_NAMES):
            continue
        part_top = max(layout.top_z, cand.body_h)
        gap = z_hover - opt.height / 2 - part_top
        if gap < MIN_GAP:
            continue
        # Margin only counts lift the magnet can actually reach: it cannot sink below the
        # tops of the coils/magnets, so peak lift is taken over z >= part_top + h/2.
        reach = zs >= part_top + opt.height / 2
        margin_c = fz[reach].max() / weight if reach.any() else 0.0
        if margin_c < MIN_MARGIN:
            continue
        best = None
        for rc in coil_ring_options(cand, layout):
            s.COIL_RING_RADIUS = rc
            pair = s.build_x_pair(cand, 1.0, 3, 4)
            fc = abs(s.force_on(floater, pair, (x2, 0, z_hover))[0])
            at_req = s.MARGIN * f2 / fc
            if best is None or at_req < best[1]:
                best = (rc, at_req, fc)
        if best is None:
            continue
        rc, at_req, fc = best
        n = cand.turns_estimate()
        rows.append(dict(
            coil=cand.name.split(" (")[0], lift=f"{p['per_stack']}x D{p['disc_d']}x{p['disc_h']}",
            spacer_mm=p["base"],
            lift_r_mm=p["radius"], floater=opt.name, mass_g=round(opt.mass * 1e3, 1),
            margin=round(margin_c, 2), hover_z_mm=round(z_hover / MM, 1), gap_mm=round(gap / MM, 1),
            k_N_per_m=round(k, 2), growth_ms=round(growth_ms), coil_ring_mm=round(rc / MM, 1),
            mN_per_AT=round(fc * 1e3, 4), AT_req=round(at_req, 1),
            I_req_A=round(at_req / n, 3),
            I_rated_A=cand.i_rated if cand.i_rated else "",
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    params = list(layouts(args.quick))
    print(f"Evaluating {len(params)} lift x floater layouts on {os.cpu_count()} cores ...")
    rows = []
    with ProcessPoolExecutor() as ex:
        for r in ex.map(evaluate, params, chunksize=4):
            rows.extend(r)

    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    if not rows:
        print("No layout met the margin/gap filters.")
        return
    with open(out / "layout_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} feasible (layout, coil) combos with margin >= {MIN_MARGIN} "
          f"and gap >= {MIN_GAP / MM:.0f} mm  (air-core coils, {s.MARGIN}x force margin "
          f"at {s.X_RECOVER / MM:.0f} mm)")
    cols = ["lift", "spacer_mm", "lift_r_mm", "floater", "mass_g", "margin", "gap_mm", "growth_ms",
            "coil_ring_mm", "mN_per_AT", "AT_req", "I_req_A", "I_rated_A"]
    for coil in sorted({r["coil"] for r in rows}):
        best = sorted((r for r in rows if r["coil"] == coil), key=lambda r: r["I_req_A"])[:8]
        print(f"\n== {coil}: lowest current needed ==")
        print("  " + "  ".join(f"{c:>12s}" for c in cols))
        for r in best:
            print("  " + "  ".join(f"{str(r[c]):>12s}" for c in cols))
    print(f"\nFull table: {out / 'layout_sweep.csv'}")


if __name__ == "__main__":
    main()
