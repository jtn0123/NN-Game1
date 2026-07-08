# Crystal Caves Level Fidelity Audit

Comparison of the 16 hand-crafted levels (`src/game/crystal_caves_handcrafted_levels.py`)
against how the **original Crystal Caves Episode 1** (Apogee, 1991) is actually designed.

Facts about the original were gathered by two independent research passes (ModdingWiki,
Wikipedia, MobyGames, GameFAQs walkthrough, TV Tropes, fan writeups). Structural numbers
for the real levels come from the level bytes we extracted from the game binary
(`experiments/cc_status/data/cc1_levels_raw.json`). No original level layouts are
reproduced here — only mechanics and aggregate statistics.

> **2026-07-02 addendum:** the structural comparison below describes the ORIGINAL
> hand-crafted set this audit reviewed. The levels were since REBUILT to address
> exactly these gaps: the current set is 40x24 with 30-34 crystals, 6-7 enemies,
> gun/ammo, and hazards per level (see LEVEL_COMPARISON_EVAL.md and the level
> validator). The ladder-heavy vertical structure remains a documented deviation.

## Verdict in one line

The levels are **mechanically authentic to a faithful SUBSET** of Crystal Caves and match
its core loop and level count, but they are **~half the size and much simpler** than the
originals: no enemies, no gun/ammo, no moving platforms, gravity-flips, or power-ups, and
they lean on ladders where the original leans on jumping + platforms.

## What the original does (corroborated by both research passes)

- **16 caves per episode**, entered from a **hub map** in any order; finishing requires
  clearing every cave.
- **Win condition:** collect **every crystal** in a cave, then reach the **exit airlock**.
  Completion is signalled by a sound cue and the screen **border turning red→green**.
- **Standard cave size: 40 × 24 tiles** (levels 7, 8, 14 are 23 rows).
- **Colored levers open the matching-colored door;** other **switches** drive elevators,
  gun turrets, and lights.
- **Vertical travel:** jumping plus **moving platforms / elevators** and **gravity-flip**
  sections (G-Pill). *(Sources conflict on climbable poles/ladders — see caveat below.)*
- **Rocket gun** (starts with 5 rockets, more found in levels) to destroy some enemies and
  obstacles; **some enemies must be avoided, not shot.**
- **Enemies:** bats, robots, Rockmen, birds, dinosaurs, mine carts.
- **Hazards:** spikes (stalactites/stalagmites), stationary turrets, moving lasers, crusher
  hammers, leaking pipes, instant-kill poison mushrooms.
- **Air generator:** one per level; **shooting it kills you** (vacuum/inflate death).
- **Power-ups:** P-Pill (super gun), G-Pill (gravity flip), Stop Signs (freeze enemies).
- **3 HP** per level; full-health completion pays a $50,000 bonus.

### Honest caveat: ladders/climbing
The two research passes **disagreed**. One found the original uses only 4 inputs
(left/right/jump/shoot) with **no climbing** — verticality via jumps, moving platforms, and
gravity flips. The other found walkthroughs that **reference climbing poles**. I can't
resolve it from public sources. Either way, the original's *primary* vertical mechanic is
jumping + platforms, **not** the ladder-heavy structure I used. (Ladders exist in the
*current engine* — added later — so my levels are engine-faithful even if not
original-faithful on this point.)

## Structural comparison (measured)

| | Original Ep-1 (extracted bytes) | My hand-crafted set |
|---|---|---|
| Level count | 16 | 16 ✅ |
| Width | 40 | 40 ✅ |
| Height | 24 (23 for L7/8/14) | 7–12 ❌ (~half) |
| Non-blank tiles / level | ~450–700 | ~100–160 ❌ (much sparser) |
| Enemies | many per level | 0 ❌ |
| Gun / ammo / shootables | central mechanic | none ❌ |

## Aspect-by-aspect fidelity

| Design aspect | Original | Mine | Verdict |
|---|---|---|---|
| Level count (16) | 16 | 16 | ✅ Faithful |
| Grid width (40) | 40 | 40 | ✅ Faithful |
| Grid height | 24 | 7–12 | ❌ ~half size |
| Win = collect ALL crystals → exit | yes | yes (engine-enforced) | ✅ Faithful |
| Colored lever → matching door | yes | yes (switch→door) | ✅ Faithful (I mostly use one color) |
| Switches drive elevators/turrets/lights | yes | door-opening only | 🟡 Partial |
| Vertical via jumps + moving platforms | yes | ladders + jumps | 🟡 Deviation |
| Elevators / moving platforms | yes | none | ❌ Missing |
| Gravity-flip sections | yes | none | ❌ Missing |
| Enemies (bats/robots/Rockmen…) | many | none | ❌ Missing |
| Rocket gun + ammo + shootables | central | none | ❌ Missing |
| Hazard variety | spikes+turrets+lasers+hammers+pipes+mushrooms | spikes only | 🟡 Partial |
| Air generator (don't-shoot death) | 1/level | none | ❌ Missing |
| Treasure / power-ups / stop signs | yes | none | ❌ Missing |
| Guaranteed winnable | shipped winnable | **verified 16/16** | ✅ (mine explicitly proven) |

## Why the gaps exist (deliberate)

The set was built to be **provably winnable** (the imported real levels were 0/16 winnable).
Omitting enemies, guns, moving platforms, gravity-flips, and power-ups keeps the
reachability oracle able to *certify* each level and keeps early training tractable. The
trade-off is lower fidelity to the busy, combat-heavy originals.

## If we want higher fidelity next

In rough order of value vs. risk to winnability:
1. **Grow the levels to the full 40×24** and add more vertical structure.
2. **Add enemies** (engine has crawlers `M` / flyers `F`) — the biggest feel gap; needs the
   oracle extended to model combat/avoidance, or accept enemies as non-blocking for the
   winnability proof.
3. **Add ammo `A` + treasure `$`** pickups (engine supports both) for the gun economy.
4. **Add elevators `=`** for original-style vertical travel (engine supports them) and lean
   less on ladders.
5. **Add hazard variety** (acid, plus the current spikes).
6. Optionally **gravity-flip** sections (engine has a gravity power-up).

All of these are supported by the current engine; each should be re-verified with
`python -m experiments.cc_status.level_reach` after adding.
