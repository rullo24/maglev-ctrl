# sim/

Magnetostatic sizing model for the floating-magnet levitator, built on
[magpylib](https://magpylib.readthedocs.io) and `magpylib-force`.

```bash
uv sync                                   # once: create .venv from uv.lock
uv run sim/levitation_sizing.py           # full run (~10 s), plots -> sim/out/
uv run sim/levitation_sizing.py --quick   # coarser, ~4 s
uv run sim/layout_sweep.py                # layout search, ~1 min on all cores -> sim/out/layout_sweep.csv
```

`layout_sweep.py` tries 2,160 layouts:

| Parameter | Values swept |
|---|---|
| Lift magnet size | 6 sizes |
| Magnets per corner | 1–4 |
| Lift ring radius | 14–30 mm |
| Floating magnet | 5 sizes |
| Spacer height | 0/5/10 mm |

It keeps layouts with **vertical margin ≥ 1.5** (counting only lift the magnet can reach above the
coil tops) and a **gap ≥ 3 mm**. Then it ranks each coil candidate by the current it needs, at its best
feasible coil-ring radius.

## What it models

- 4 permanent lift-magnet stacks on the diagonals, north up (passive Z lift)
- 4 radial drum-core inductors on ±X / ±Y, each axis pair driven in opposition (active X/Y)
- Hall sensor at the board centre
- Floating disc magnet above, north down

For each floating-magnet option it reports:

- hover height and gap
- vertical margin (peak lift ÷ weight)
- lateral instability stiffness `k`
- how fast the instability grows
- the field at the sensor

For each coil candidate it reports:

- restoring force per ampere-turn (AT)
- the AT and current needed to push the magnet back from `X_RECOVER` with `MARGIN`
- I²R heat
- L/R
- coil field leaking into the sensor's Bx reading

## Assumptions / limits

- **Coils are modelled without their ferrite core**, so coil force is a conservative lower bound.
  Measure the real force once (scale or pendulum test), then set `CORE_GAIN`.
- **Turns are estimated from inductance** using an assumed core permeability `MU_EFF`. Set
  `CoilCandidate.turns` if you measure the real count.
- **Tilt is not modelled.** The floating magnet is held level and only shifted sideways.
- **Coil ratings come from distributor listings.** Entries marked `?` need their LCSC page checked.
- **All dimensions are kit-like defaults.** Edit the dataclasses at the top of the script.

## Key findings (air-core coils, so pessimistic)

1. **There's a trade-off: margin against coil authority.** Stronger lift gives more vertical margin but hovers
   higher, and coil force falls off steeply with height. A strong kit-like lift (12 × D10×3 mm) floats
   the magnet ~45 mm above the parts, where every coil would need tens of amps.
2. **The best layouts use tiny lift magnets in a tight ring under a wide floating magnet.** One D5–6×2 mm
   disc per corner on a 14 mm radius, with a 25–30 mm × 4 mm disc floating ~8–11 mm above the coils,
   gives margin 1.6–2.1. This is now the default in `levitation_sizing.py`.
3. **Raising the lift magnets on spacers doesn't help.** The hover point rises with them.
4. **Current needed** to push back from a **2 mm** offset with **2.5× force margin** (best layouts):

   | Coil | Air-core estimate | Rated |
   |---|---|---|
   | RLB1314-331KL | ~1.2–1.6 A | 0.55 A |
   | VLU1012-221K | ~1.5–2.0 A | ? |
   | VLU1415-size | ~0.9–1.2 A | ? (needs a wider coil ring) |
   | 10–11 mm parts | 2.5–4 A | |

   Two things bring these numbers down:
   - The ferrite core (`CORE_GAIN`, likely ~2×).
   - A smaller recovery target. Current scales roughly with `X_RECOVER`, so 1 mm halves it.

   With both, the **RLB1314 or a VLU1415-size part is plausible; 10–11 mm parts are not**.
5. **Bigger floating magnets need more coil current** at a given layout, but they buy vertical margin.
6. **The instability is slow.** It grows over ~50–55 ms (unstable pole ~3 Hz), so a 1 kHz loop with L/R < 1 ms
   coils has plenty of bandwidth.
7. **The sensor needs coil compensation.** |Bz| at the centre is ~20–30 mT, within TMAG5170A1 range.
   The floating magnet gives ~0.6–0.9 mT of Bx per mm of offset. The coils add ~0.004–0.007 mT per AT,
   about 0.4–0.6 mT at ~100 AT, which is comparable to the signal, so the firmware must subtract it.

## Next step that matters most

**Measure `CORE_GAIN`.** Do a scale or pendulum test on one RLB1314 or VLU1415 with a 25×4 mm magnet.
That single number decides whether these parts pass outright.
