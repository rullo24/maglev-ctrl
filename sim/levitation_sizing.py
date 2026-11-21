"""
Levitation sizing model for maglev-ctrl.

Repulsive ("float above the board") layout:

    * 4 permanent lift-magnet stacks on the diagonals, north up  -> passive lift (Z)
    * 4 radial drum-core inductors on the +/-X and +/-Y axes      -> active X/Y control
    * 1 Hall sensor (e.g. TMAG5170) at the centre, on the board
    * 1 floating disc magnet above, north DOWN (repelled)

For each floating-magnet option the script finds:
    1. hover height (lift force == weight, vertically stable),
    2. lateral instability: destabilising force F(x) and stiffness k,
    3. how fast the instability grows (needs the control loop to be faster),
    4. field seen by the centre sensor (range check).

For each candidate inductor it then finds:
    5. restoring force per ampere-turn of one X coil pair (driven in opposition),
    6. ampere-turns and current needed to recover from X_RECOVER with MARGIN,
    7. pass/fail against the part's current rating, plus I^2R heat and L/R.

Coils are modelled as AIR-CORE multi-layer windings (ferrite core ignored) unless
CORE_GAIN > 1, so coil force is a conservative lower bound. Calibrate CORE_GAIN with
one bench measurement (scale or pendulum test) for real numbers.

All values in this file are kit-like DEFAULTS - replace with real dimensions.

Run:  uv run sim/levitation_sizing.py            (full run, writes plots to sim/out/)
      uv run sim/levitation_sizing.py --quick    (coarser, faster)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import magpylib as magpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from magpylib_force import getFT  # noqa: E402

MU0 = 4e-7 * math.pi
G = 9.81
RHO_NDFEB = 7500.0  # kg/m^3
MM = 1e-3

# --------------------------------------------------------------------------------------
# Design inputs (edit these)
# --------------------------------------------------------------------------------------

X_RECOVER = 2.0 * MM  # largest lateral offset the coils must be able to push back from
MARGIN = 2.5  # force margin over the destabilising force at X_RECOVER
PAYLOAD_G = 0.0  # extra mass carried by the floating magnet (grams)
CORE_GAIN = 1.0  # ferrite enhancement of coil field (1.0 = air core, conservative)
MU_EFF = 15.0  # assumed effective permeability of an open drum core (turns estimate)
SENSOR_POS = (0.0, 0.0, 0.5 * MM)  # centre of the Hall sensor die, above board top (z=0)
SENSOR_RANGE_T = 0.100  # e.g. TMAG5170A1 widest range = +/-100 mT (A2 goes to +/-300 mT)


@dataclass
class FloaterOption:
    name: str
    diameter: float  # m
    height: float  # m
    br: float = 1.32  # T, ~N42

    @property
    def mass(self) -> float:
        return RHO_NDFEB * math.pi * (self.diameter / 2) ** 2 * self.height + PAYLOAD_G * 1e-3


@dataclass
class LiftLayout:
    """4 stacks of small disc magnets on the diagonals, north up."""

    # Default = best balanced layout from sim/layout_sweep.py: hovers low (~11 mm above
    # the coils) with >= 2x vertical margin. Stronger lift => higher hover => much weaker
    # coil authority (see sim/README.md).
    radius: float = 14.0 * MM  # distance of each stack centre from board centre
    disc_d: float = 6.0 * MM
    disc_h: float = 2.0 * MM
    per_stack: int = 1  # one small disc per corner
    base_z: float = 0.0  # stack bottom above board top
    br: float = 1.22  # T, ~N35 (cheap kit magnets)

    @property
    def top_z(self) -> float:
        return self.base_z + self.per_stack * self.disc_h


@dataclass
class CoilCandidate:
    name: str
    body_d: float  # m (flange / body diameter)
    body_h: float  # m
    inductance: float  # H
    dcr: float | None  # ohm (None = unknown, check LCSC page)
    i_rated: float | None  # A  (rated / saturation current, None = unknown)
    turns: int | None = None  # measured turns if known, else estimated from L

    # Drum-core winding window, as fractions of the body (typical proportions)
    @property
    def r_in(self) -> float:
        return 0.24 * self.body_d  # core (bobbin) radius

    @property
    def r_out(self) -> float:
        return 0.46 * self.body_d  # outside of winding

    @property
    def wind_z0(self) -> float:
        return 0.12 * self.body_h  # winding starts above bottom flange

    @property
    def wind_z1(self) -> float:
        return 0.88 * self.body_h  # winding ends below top flange

    def turns_estimate(self) -> float:
        if self.turns:
            return float(self.turns)
        area = math.pi * self.r_in**2
        length = self.wind_z1 - self.wind_z0
        return math.sqrt(self.inductance * length / (MU0 * MU_EFF * area))

    @property
    def tau(self) -> float | None:
        return self.inductance / self.dcr if self.dcr else None


COIL_RING_RADIUS = 14.0 * MM  # coil centres at (+/-R, 0) and (0, +/-R)

FLOATERS = [
    FloaterOption("disc 20x4 N42", 20 * MM, 4 * MM),
    FloaterOption("disc 25x4 N42", 25 * MM, 4 * MM),
    FloaterOption("disc 30x4 N42", 30 * MM, 4 * MM),
]

COILS = [
    # Values from distributor listings - confirm on the LCSC/Bourns pages.
    CoilCandidate("SXN PK0810-151MB (C400316)", 10.0 * MM, 13.0 * MM, 150e-6, 0.24, 0.72),
    CoilCandidate("Bourns 6100-101K-RC (C3350165)", 11.0 * MM, 13.0 * MM, 100e-6, 0.16, 1.98),
    CoilCandidate("FH VLU1012-221K (C128035)", 12.0 * MM, 14.5 * MM, 220e-6, None, None),
    CoilCandidate("Bourns RLB1314-331KL", 13.5 * MM, 14.0 * MM, 330e-6, 0.42, 0.55),
    CoilCandidate("FH VLU1415-331K (if stocked)", 16.5 * MM, 17.0 * MM, 330e-6, None, None),
]

# --------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------


def build_lift(layout: LiftLayout) -> magpy.Collection:
    col = magpy.Collection()
    r = layout.radius / math.sqrt(2)
    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        for k in range(layout.per_stack):
            z = layout.base_z + (k + 0.5) * layout.disc_h
            col.add(
                magpy.magnet.Cylinder(
                    polarization=(0, 0, layout.br),
                    dimension=(layout.disc_d, layout.disc_h),
                    position=(sx * r, sy * r, z),
                )
            )
    return col


def build_coil(cand: CoilCandidate, center_xy, ampere_turns: float, n_r: int, n_z: int):
    """Multi-layer air-core winding as a grid of current loops carrying `ampere_turns` total."""
    col = magpy.Collection()
    radii = np.linspace(cand.r_in, cand.r_out, n_r + 2)[1:-1]
    zs = np.linspace(cand.wind_z0, cand.wind_z1, n_z + 2)[1:-1]
    i_loop = CORE_GAIN * ampere_turns / (n_r * n_z)
    for r in radii:
        for z in zs:
            col.add(
                magpy.current.Circle(
                    current=i_loop, diameter=2 * r, position=(center_xy[0], center_xy[1], z)
                )
            )
    return col


def build_x_pair(cand: CoilCandidate, ampere_turns: float, n_r: int, n_z: int):
    """+X coil carries +AT, -X coil carries -AT (driven in opposition by one H-bridge)."""
    R = COIL_RING_RADIUS
    pair = magpy.Collection()
    pair.add(build_coil(cand, (R, 0.0), ampere_turns, n_r, n_z))
    pair.add(build_coil(cand, (-R, 0.0), -ampere_turns, n_r, n_z))
    return pair


def make_floater(opt: FloaterOption, meshing: int):
    f = magpy.magnet.Cylinder(
        polarization=(0, 0, -opt.br),  # north DOWN -> repelled by north-up lift magnets
        dimension=(opt.diameter, opt.height),
    )
    f.meshing = meshing
    return f


def force_on(floater, sources, pos) -> np.ndarray:
    floater.position = pos
    ft = getFT(sources, floater, anchor=pos)
    return np.asarray(ft[0])


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def find_hover(floater, opt: FloaterOption, lift, z_floor: float, z_ceil: float):
    """Floater centre height where lift == weight with dFz/dz < 0 (vertically stable)."""
    weight = opt.mass * G
    zs = np.arange(z_floor, z_ceil, 0.5 * MM)
    fz = np.array([force_on(floater, lift, (0, 0, z))[2] for z in zs])
    for i in range(len(zs) - 1):
        if fz[i] >= weight > fz[i + 1]:
            lo, hi = zs[i], zs[i + 1]
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                if force_on(floater, lift, (0, 0, mid))[2] >= weight:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi), zs, fz
    return None, zs, fz


def analyse(quick: bool):
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    meshing = 150 if quick else 600
    n_r, n_z = (3, 4) if quick else (4, 6)
    offsets = np.linspace(0.25, 4.0, 6 if quick else 12) * MM

    layout = LiftLayout()
    lift = build_lift(layout)
    tallest_coil = max(c.body_h for c in COILS)
    obstacle_top = max(layout.top_z, tallest_coil)

    print_header(layout, obstacle_top)
    check_clearance(layout)

    results = []
    for opt in FLOATERS:
        floater = make_floater(opt, meshing)
        z_floor = obstacle_top + 2 * MM + opt.height / 2  # >= 2 mm clearance above parts
        z_hover, zs, fz = find_hover(floater, opt, lift, z_floor, z_floor + 100 * MM)
        res = {"opt": opt, "z": z_hover, "zs": zs, "fz": fz}
        results.append(res)
        if z_hover is None:
            print(
                f"\n[{opt.name}] mass {opt.mass * 1e3:.1f} g: NO HOVER - lift "
                f"{fz.max() * 1e3 / G:.1f} gf max vs weight {opt.mass * 1e3:.1f} gf "
                f"(at >=2 mm above parts). Stronger/closer lift magnets or lighter floater."
            )
            continue

        # Lateral instability from the lift field alone
        f_destab = np.array([force_on(floater, lift, (x, 0, z_hover))[0] for x in offsets])
        k = np.polyfit(offsets[:3], f_destab[:3], 1)[0]  # N/m, small-offset slope
        growth = math.sqrt(k / opt.mass) if k > 0 else 0.0
        b_sensor = lift.getB(SENSOR_POS) + floater_at(floater, z_hover).getB(SENSOR_POS)
        dbx_dx = (
            floater_at(floater, z_hover, 0.5 * MM).getB(SENSOR_POS)[0]
            - floater_at(floater, z_hover, -0.5 * MM).getB(SENSOR_POS)[0]
        ) / (1 * MM)
        floater_at(floater, z_hover)  # reset

        res.update(offsets=offsets, f_destab=f_destab, k=k, growth=growth)
        gap = z_hover - opt.height / 2 - obstacle_top
        print(f"\n[{opt.name}]  mass {opt.mass * 1e3:.1f} g")
        print(
            f"  hover: centre z = {z_hover / MM:.1f} mm, gap above tallest part = {gap / MM:.1f} mm"
        )
        lift_margin = fz.max() / (opt.mass * G)
        print(
            f"  vertical margin: peak lift / weight = {lift_margin:.2f}"
            + ("  <-- LOW: a small knock drops it; aim for >= 1.5" if lift_margin < 1.5 else "")
        )
        if k > 0:
            print(
                f"  lateral instability: k = {k:.2f} N/m, F({X_RECOVER / MM:.0f} mm) = "
                f"{np.interp(X_RECOVER, offsets, f_destab) * 1e3:.2f} mN, "
                f"e-folding time {1e3 / growth:.0f} ms (unstable pole {growth / (2 * math.pi):.1f} Hz)"
            )
        else:
            print(f"  lateral: k = {k:.2f} N/m (laterally STABLE here - unusual, check layout)")
        print(
            f"  sensor: |Bz| = {abs(b_sensor[2]) * 1e3:.1f} mT "
            f"({'OK' if abs(b_sensor[2]) < SENSOR_RANGE_T else 'EXCEEDS'} vs "
            f"+/-{SENSOR_RANGE_T * 1e3:.0f} mT), "
            f"Bx sensitivity to floater offset = {abs(dbx_dx):.2f} mT/mm"  # T/m == mT/mm
        )

        # Coil candidates
        rows = []
        for cand in COILS:
            pair = build_x_pair(cand, 1.0, n_r, n_z)  # 1 ampere-turn per coil
            f_coil = np.array([force_on(floater, pair, (x, 0, z_hover))[0] for x in offsets])
            # choose current sign so that the pair pushes back toward the centre
            sign = -1.0 if np.mean(f_coil) > 0 else 1.0
            f_restore = -sign * f_coil  # restoring force magnitude per AT (positive)
            fr = np.interp(X_RECOVER, offsets, f_restore)
            fd = np.interp(X_RECOVER, offsets, f_destab)
            at_req = MARGIN * max(fd, 0.0) / fr if fr > 0 else math.inf
            n_est = cand.turns_estimate()
            i_req = at_req / n_est
            bx_per_at = pair.getB(SENSOR_POS)[0]  # coil feed-through into the sensor
            rows.append((cand, fr, at_req, n_est, i_req, bx_per_at, f_coil, sign))
        res["rows"] = rows
        print_coil_table(rows)

    plot(results, out_dir)
    print(f"\nPlots written to {out_dir}/")


def floater_at(floater, z, x=0.0):
    floater.position = (x, 0.0, z)
    return floater


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def check_clearance(layout: LiftLayout):
    """Warn if a coil body would overlap a lift stack or a neighbouring coil."""
    r = layout.radius / math.sqrt(2)
    d_coil_mag = math.hypot(COIL_RING_RADIUS - r, r)
    d_coil_coil = COIL_RING_RADIUS * math.sqrt(2)
    for c in COILS:
        if d_coil_mag < c.body_d / 2 + layout.disc_d / 2:
            print(f"  WARNING: {c.name} overlaps a lift stack at this layout")
        if d_coil_coil < c.body_d:
            print(f"  WARNING: {c.name} bodies overlap each other at COIL_RING_RADIUS")


def print_header(layout: LiftLayout, obstacle_top: float):
    print("=" * 96)
    print("maglev-ctrl levitation sizing  (kit-like defaults - edit sim/levitation_sizing.py)")
    print("=" * 96)
    print(
        f"Lift: 4 stacks x {layout.per_stack} x D{layout.disc_d / MM:.0f}x{layout.disc_h / MM:.0f} mm "
        f"(Br {layout.br} T) on diagonals at r = {layout.radius / MM:.0f} mm, top z = {layout.top_z / MM:.1f} mm"
    )
    print(
        f"Coils: on +/-X, +/-Y at r = {COIL_RING_RADIUS / MM:.0f} mm, air-core model x CORE_GAIN={CORE_GAIN}, "
        f"turns estimated with mu_eff={MU_EFF}"
    )
    print(
        f"Criterion: coil pair force >= {MARGIN}x destabilising force at x = {X_RECOVER / MM:.1f} mm, "
        f"tallest part top z = {obstacle_top / MM:.1f} mm"
    )


def print_coil_table(rows):
    print(
        f"  {'coil':34s} {'mN/AT':>7s} {'AT req':>7s} {'turns~':>7s} {'I req':>8s} "
        f"{'I rated':>8s} {'P pair':>8s} {'L/R':>7s} {'Bx fb':>9s}  verdict"
    )
    for cand, fr, at_req, n_est, i_req, bx_per_at, *_ in rows:
        rated = f"{cand.i_rated:.2f} A" if cand.i_rated else "?"
        p = f"{2 * i_req**2 * cand.dcr * 1e3:.0f} mW" if cand.dcr else "?"
        tau = f"{cand.tau * 1e3:.2f}ms" if cand.tau else "?"
        if cand.i_rated is None:
            verdict = "rating unknown"
        elif i_req <= cand.i_rated:
            verdict = f"PASS ({cand.i_rated / i_req:.1f}x headroom)"
        else:
            verdict = f"FAIL (needs {i_req / cand.i_rated:.1f}x rating)"
        print(
            f"  {cand.name:34s} {fr * 1e3:7.3f} {at_req:7.1f} {n_est:7.0f} {i_req:6.3f} A "
            f"{rated:>8s} {p:>8s} {tau:>7s} {abs(bx_per_at) * 1e3:6.3f}mT/AT  {verdict}"
        )


def plot(results, out_dir: Path):
    ok = [r for r in results if r["z"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r in results:
        axes[0].plot(r["zs"] / MM, r["fz"] / G * 1e3, label=r["opt"].name)
        axes[0].axhline(r["opt"].mass * 1e3, ls=":", lw=0.8, color=axes[0].lines[-1].get_color())
    axes[0].set(xlabel="floater centre height z (mm)", ylabel="lift (gram-force)",
                title="Vertical lift vs height (dotted = weight)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for r in ok:
        if "f_destab" in r:
            axes[1].plot(r["offsets"] / MM, r["f_destab"] * 1e3, label=f"{r['opt'].name}: destabilising")
    axes[1].axvline(X_RECOVER / MM, ls="--", color="grey", lw=0.8)
    axes[1].set(xlabel="lateral offset x (mm)", ylabel="force (mN)",
                title="Lateral destabilising force from lift magnets")
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "lift_and_instability.png", dpi=130)

    fig, axes = plt.subplots(1, max(len(ok), 1), figsize=(5 * max(len(ok), 1), 4.5), squeeze=False)
    for ax, r in zip(axes[0], ok):
        if "rows" not in r:
            continue
        for cand, fr, at_req, n_est, i_req, _, f_coil, sign in r["rows"]:
            ax.plot(r["offsets"] / MM, -sign * f_coil * 1e3, label=cand.name.split(" (")[0])
        ax.set(xlabel="lateral offset x (mm)", ylabel="restoring force per AT (mN/AT)",
               title=f"Coil pair force, {r['opt'].name}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "coil_force_per_AT.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="coarser meshing for a fast check")
    analyse(ap.parse_args().quick)
