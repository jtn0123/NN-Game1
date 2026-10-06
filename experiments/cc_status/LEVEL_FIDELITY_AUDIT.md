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

---

# 2026-07-07 Full-Decode Fidelity Audit (supersedes the structural table above)

The original-level ground truth below comes from a NEW decoder,
`experiments/cc_status/cc1_decode.py`, which decodes ALL 16 original levels
from `data/cc1_levels_raw.json` with **zero unknown bytes** using the
authoritative byte->object mapping (ModdingWiki "Crystal Caves Map Format" +
Camoto libgamemaps `fmt-map-ccaves-mapping.hpp`). The old best-effort decode
(commit `2d4134b`) misread `0x64`/`0x6E` (I-beam / multi-cell continuation) as
ladders/air and did not know the concrete terrain family — that is exactly why
6/16 levels looked "garbled" and all object types were dropped.

**Decode validation anchors:**
- L4 decodes to exactly **34 visible gems**, matching the `00/34` HUD in the
  original-game screenshot referenced in `LEVEL_COMPARISON_EVAL.md`.
- L7/L8/L14 decode as 23-row levels, exactly as ModdingWiki documents.
- The six formerly "garbled" levels all use the concrete terrain family
  (`0x4B/0x4C/0x6B/0x6C`) the old decode lacked; they now decode fully.
- L16's raw extract carries one trailing row of non-map bytes (EXE overrun);
  the decoder detects and trims it.

## Ladder question: RESOLVED (primary source)

The 1991 format has **no ladder tile**. Its only climbables are vines and
hanging chains (codes `0x85`-`0x88`), present in just **3 of 16 levels**
(4-17 cells). Original verticality is jumping + I-beam/pipe platforms +
moving platforms. Both earlier research passes were half right: climbing
exists (vines), ladders do not.

## Original Episode 1 ground truth (stats only; no layouts reproduced)

| lvl | size | gems (hidden) | enemies | hazards | mushrooms | switches | doors | moving plat | vine cells | pickups | dark |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L1 | 40x24 | 27 (13) | 6 | 3 | 0 | 3 | 2 | 2 | 0 | 15 | yes |
| L2 | 40x24 | 32 (7) | 6 | 15 | 1 | 5 | 3 | 1 | 0 | 16 | |
| L3 | 40x24 | 46 | 11 | 7 | 16 | 0 | 0 | 1 | 0 | 17 | |
| L4 | 40x24 | 34 | 8 | 5 | 1 | 3 | 2 | 2 | 0 | 9 | |
| L5 | 40x24 | 65 (6) | 6 | 5 | 7 | 3 | 1 | 2 | 4 | 4 | |
| L6 | 40x24 | 108 | 11 | 11 | 0 | 0 | 1 | 0 | 0 | 10 | |
| L7 | 40x23 | 35 | 6 | 13 | 13 | 2 | 1 | 3 | 17 | 27 | |
| L8 | 40x23 | 60 | 8 | 8 | 1 | 3 | 1 | 1 | 0 | 14 | |
| L9 | 40x24 | 53 (5) | 7 | 12 | 1 | 3 | 3 | 3 | 0 | 22 | |
| L10 | 40x24 | 40 | 9 | 22 | 5 | 4 | 3 | 2 | 0 | 5 | |
| L11 | 40x24 | 28 | 16 | 6 | 2 | 4 | 3 | 2 | 0 | 13 | |
| L12 | 40x24 | 44 (2) | 6 | 6 | 1 | 2 | 1 | 2 | 10 | 15 | |
| L13 | 40x24 | 38 | 14 | 3 | 10 | 1 | 0 | 0 | 0 | 13 | |
| L14 | 40x23 | 47 | 12 | 2 | 0 | 3 | 3 | 2 | 0 | 18 | |
| L15 | 40x24 | 32 | 7 | 2 | 0 | 4 | 3 | 0 | 0 | 28 | |
| L16 | 40x23 | 57 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 23 | |

Notes: "hidden" gems live inside I-beam platform tiles (codes `0x98-0x9A`) —
a mechanic our engine does not have. L1 starts dark with a light switch
(`0xA6`) — also not in our engine. Pipes (green + corrugated) are walkable
platform scaffolding in the original, not decor.

## Fidelity deltas: current handcrafted set vs decoded originals

| Metric | Handcrafted (16) | Original (16) | Verdict |
|---|---|---|---|
| Grid | 40x24 | 40x24 (23 for L7/8/14) | ok match |
| Crystals | 30-34, mean 30.5, near-uniform | 27-108, mean 46.6, high variance | LOW + too uniform |
| Enemies | exactly 6-7 everywhere | 6-16, mean 9.2 | LOW + too uniform |
| Hazards (+mushrooms) | 1-6, mean 3.1 | 0-27, mean 11.1 | ~3.5x LOW |
| Switches / doors | mean 0.8 / 0.8 | mean 2.5 / 1.7 | LOW |
| Climbable cells | ladders 9-80, mean 46.3, every level | vines 0-17, mean 1.9, only 3/16 levels | **~24x OVER — biggest structural deviation** |
| Moving-platform cells | mean 9.2 | mean 1.4 (single-tile movers) | over, different mechanic |
| Pickups | 2-5, mean 3.4 | 4-28, mean 15.6 | ~4.5x LOW |
| Solid fraction | mean 0.40 | mean 0.40 | ok match |
| Fill fraction | mean 0.50 | mean 0.57 | close |
| Per-level variance | near-zero (stat-identical levels) | large (dense gem-farm L6 vs sparse lever-maze L15) | missing character variance |

## Debug pass (level_validator, five independent checks per level)

- Initial run: 15/16 OK. **Cavern of Echoes = JUNK**: a flyer at (24,8) was
  sealed inside a 1x1 pocket (`#F#` with solid above/below) — unencounterable
  dead content. FIXED: moved into the adjacent patrol corridor; validator now
  reports **16/16 OK** and the full handcrafted/validator test suite passes.
- Remaining warnings (not failures): spawn-zone ambush enemies on Ore Shaft,
  The Smelter, Cavern of Echoes, The Mother Lode; Ore Shaft clock usage 0.82 /
  stall-window usage 0.87 (the known DATA-1 tightness).

## Ranked authenticity fix list (stats-only targets; layouts stay original designs)

1. **F1 crystal counts + variance:** per-level targets drawn from the original
   distribution (e.g. 27-65, median ~42; treat L6=108 as the one gem-farm
   outlier) instead of uniform 30-34.
2. **F2 hazards:** mean ~3 -> ~9-11 with high variance (0-22), including
   crystals guarded by adjacent threats (the original guards ~1/3 of gems).
3. **F3 enemies:** mean 6 -> ~9, range 6-16, varied per level.
4. **F4 vertical-structure rebalance (biggest):** cut ladder dependence
   (46 cells/level -> low single digits on most levels) in favour of
   jump-platform chains, I-beam-style thin platforms, and moving platforms;
   keep at most a few vine-like climbs on a minority of levels.
5. **F5 pickups:** 3.4 -> ~10-15 (ammo, treasure) per level.
6. **F6 gates:** switches/doors toward 2-3 per level incl. multi-colour
   chains on some levels.
7. **F7 (engine) hidden crystals** inside platforms, and **F8 (engine) dark
   level + light switch** — original mechanics our engine lacks; decide
   add-or-document before touching level data.

Every fix must keep the `level_reach.py` oracle certification and re-run
`level_validator` (16/16) + the handcrafted test suite.

## 2026-07-07 Rebalance pass: F1-F3, F5, F6 executed; F4 partial

All 16 handcrafted levels were rebalanced level-by-level against the decoded
per-level originals (each handcrafted level i targets original Li's stats).
Every level remains oracle-certified (plain + lock-ordering), every door
gates a real objective, `level_validator` reports **16/16 OK with zero
ambush warnings** (was 4), `compass_audit` stays clean, and the full suite
passes (1484 tests).

| Metric | Before | Now | Original |
|---|---|---|---|
| Crystals | 30-34 (mean 30.5), uniform | **28-93 (mean 46.0)**, incl. a 93-gem Mother-Lode-style farm | 27-108 (mean 46.6) |
| Total crystals | 488 | **736** | 746 |
| Enemies | 6 everywhere | **6-13 (mean 8.5)** | 6-16 (mean 9.2) |
| Hazards | 1-6 (mean 3.1) | **2-12 (mean 7.6)**, incl. guarded crystals | 0-27 (mean 11.1, incl. mushrooms we lack) |
| Switches / doors | 0.8 / 0.8 | **1.4 / 1.4** (engine caps at 2 colours) | 2.5 / 1.7 |
| Pickups | 2-5 (mean 3.4) | **5-11 (mean 8.7)** | 4-28 (mean 15.6) |
| Ladder cells | mean 46.3 | mean 43.4 (4 redundant shafts removed, ~46 cells) | vines mean 1.9 |

Per-level character now varies like the original: hazard-light gem farms
(Mother Lode 57, Elevator Exchange 93), the hazard gauntlet (Cavern Descent
11 hazards), the enemy warren (Cavern of Echoes 13), a pickup-rich spire, and
gated vaults on 12 of 16 levels (up from 8 gates total to 22 switch/door
tiles). Also fixed en route: 6 spawn-ambush layouts (blockers/relocations)
and a gated exit on Cavern of Echoes.

**Still open (F4, the honest gap):** ladder dependence. Only redundant
parallel shafts were removed; converting primary ladder verticality to
jump-platform chains is a per-level structural redesign and remains the one
big deviation from the 1991 game (43 ladder cells/level vs ~2 vine cells).
F7 (hidden crystals) and F8 (dark level + light switch) remain engine
decisions.
