<!-- Synthesized 2026-06-17 from a research workflow (3 web researchers + code analysis). The prototype scripts/gen_crystal_cave.py implements the v0 of this. -->

# Crystal Caves Level Generator — Design Spec

> Authoritative design + concrete algorithm for an AUTOMATIC Crystal Caves level generator, to be built in Python (`scripts/gen_crystal_cave.py` → promoted into `src/game/`) and iterated on. This spec is grounded in our real engine: the tile alphabet, the `CaveSpec` dataclass, the `CAVES` tuple, and the jump-aware `_cave_reachable` checker that our solvability test already uses.

---

## 1. Goals + Hard Constraints

### Goal
Produce an endless supply of **fully playable, authored-feeling Crystal Caves levels** that drop in as `CaveSpec` entries with zero hand-tuning, pass our existing solvability + density tests, and read like the reference game (descend from the top, collect every crystal, flip the switch, leave through the exit).

### The three non-negotiable constraints
1. **Top entrance, descending level.** The player spawn `P` sits in the top band of the grid (`player_row <= 3`, ideally row 2), above all objectives. The level's structure descends from there toward an exit chamber near the bottom. We do **not** adopt the reference game's "start anywhere" rule — the user requires a top entrance.
2. **Everything reachable / "walk through everything."** Every open tile that matters — the switch, **every** crystal `*`, and the exit `E` — must be reachable by the player under our jump-aware physics. Connectivity is **guaranteed by generate-and-verify**: we flood the grid with `_cave_reachable` and **repair** (refill unreachable air to solid, or re-route) until the layout verifies. We never ship a layout the checker rejects.
3. **Themed bases / biomes.** Every level carries a theme (palette + prop vocabulary) chosen from the engine's episode set. A theme sets `CaveSpec.background` / `CaveSpec.accent` and biases hazard/prop selection, but never changes the grid contract — the same 44×18 grid, same tile chars, same door/switch rules across all themes.

### Engine contract this generator must honor
- **Grid:** `44` columns × `18` rows (`COLS=44`, `ROWS=18`), tile size 32px. Layout is `Tuple[str, ...]`, all rows equal width, parsed by `_load_level()`.
- **Tile alphabet (exact chars):** `#` SOLID, `.` EMPTY, `*` CRYSTAL, `E` EXIT, `D` DOOR, `s` SWITCH, `A` AMMO, `$` TREASURE, `p` POWER_SHOT, `g` GRAVITY_POWER, `z` FREEZE_POWER, `O` AIR_TANK, `M` CRAWLER, `F` FLYER, `^` SPIKE, `~` ACID, `P` PLAYER.
- **Reachability physics (`_cave_reachable`, `jump=3`):** walk left/right on ground or in air, fall down through EMPTY, jump up **at most 3 tiles** while grounded (fuel resets on landing). A `D` door blocks unless `doors_open=True`. This is the *only* solvability oracle we trust — the generator imports/reuses the exact same flood.
- **Test gates we must pass for any cave we register:**
  - `test_every_cave_is_solvable`: from `P` with `doors_open=False` reach every `s`; from `P` with `doors_open=True` reach every `*` and `E`.
  - `test_every_cave_is_dense`: `0.45 <= count('#')/total <= 0.85`.
  - First-cave authoring contract (only if a generated cave takes slot 0): `player_col < 6`, `switch_col > player_col`, `exit_col > switch_col`, `crystals >= 8`, `ammo >= 3`, `doors >= 1`, `hazards >= 3`.

---

## 2. Level Structure Model

A level is **one fixed-size, smooth-scrolling playfield** (not flip-screen, not connected rooms) — matching the reference engine and our `_load_level` model.

```
col →  0                                            43
row 0  ##############################################   solid cap
row 1  ##... ENTRANCE BAND (P spawns here, row 2) ...##
row 2  ##.P............ open landing shelf ..........##   ← TOP ENTRANCE
row 3  ───────────────── descent zone A ──────────────   crystals cluster 1
row 5  ───────────────── descent zone B ──────────────   switch s + door D gate
row 8  ───────────────── descent zone C ──────────────   crystals cluster 2 + hazards
row 12 ───────────────── descent zone D ──────────────   crystals cluster 3 + enemies
row 15 ───────── EXIT CHAMBER (E near bottom) ─────────   exit, row >= 12
row 17 ##############################################   solid floor
```

### Coordinate + size conventions
- **Grid:** 44×18. Border ring (`row 0`, `row 17`, `col 0`, `col 43`) is always SOLID. Carving happens in the interior `1..42 × 1..16`.
- **Entrance band:** rows `1–2`. `P` is placed at `(player_col ∈ [2,5], row 2)` on a solid shelf, with EMPTY headroom above. This satisfies both the top-entrance constraint and the first-cave `player_col < 6` contract.
- **Descent zones (difficulty bands):** the interior is partitioned into **4 horizontal bands** top→bottom (A,B,C,D). Hazard density, gap width, and enemy count **increase monotonically** per band (rhythm-group pacing: a hard band is followed by an easier "rest" shelf). This is what makes the descent read as authored rather than uniform noise.
- **Exit chamber:** the lowest reachable standing ledge, `exit_row >= 12` (engine-checked `>= ROWS-6`). Exit is a distinct placed entity, not auto-derived from terrain shape.

### How verticality maps on
- Verticality is **platforms, ledges, and short drops joined so every drop's landing is climbable back out within `jump=3`** — no ladders (engine has none). The descent spine is a zig-zag of horizontal shelves connected by ≤3-tile drops, so the player can always traverse *down* and the reachability flood proves they can also get *back up* to the switch and side crystals.
- **Door gate as a soft lock:** one `s`→`D` pair gates the lower descent. The switch is placed so it is reachable **with doors closed** (constraint 2, closed-door pass), and the door blocks the route to the deepest crystal cluster + exit until pulled. This creates a Crystal-Caves-style "find the lever, then the exit opens" beat without ever creating a circular lock (switch is always upstream of its door on the descent).

### How themes map on
Themes are **reskins over the identical grid**. A `THEMES` table maps an episode/biome to:
- `background`, `accent` RGB (drives `CaveSpec`),
- a **hazard bias** (e.g. *blue rock* → spikes `^`; *rust/industrial* → acid `~`; *gray-tech* → mixed + air tanks `O`),
- a **power-up bias** (which of `p`/`g`/`z` shows up),
- an **enemy mix** (`M` crawlers vs `F` flyers).

The three engine episodes give us our three confirmed bases: **(0) Blue rock/crystal**, **(1) Rust/industrial**, **(2) Gray-tech/moon** — matching `background`/`accent` of the authored `CAVES` and the palette tests (`test_episodes_have_distinct_dominant_hues`). New biomes are added as new `THEMES` rows.

---

## 3. Generation Algorithm (Python, implementable)

We use a **carved-solution-path skeleton + jump-aware verify-and-repair**, *not* raw cellular automata. Justification: our hard constraints are (a) a top-down descending path and (b) total reachability under jump physics. The PCG research is explicit that CA gives **no connectivity guarantee** and needs flood-fill rescue anyway, while a **Spelunky/Downwell-style carved descending solution path guarantees a connected, top-to-bottom solvable spine by construction**. We carve that spine, fatten it into rooms for the density target, then use *our own jump-aware checker* (not plain grid BFS — it must respect the 3-tile jump ceiling) as the final solvability oracle with a repair loop. This is the standard "generate-and-test with a physics-aware solver" fallback, and we already have the solver.

The prototype `scripts/gen_crystal_cave.py` (`generate(seed)`, `reachable()`, `grade()`) is the seed of this; below is the target algorithm it grows into.

### (a) Carve a connected descending cave

1. **Init.** `grid = [['#']*44 for _ in range(18)]` (all solid). Keep the border ring solid forever.
2. **Place the entrance shelf.** Choose `px ∈ [2,5]`, `py = 2`. Carve a small open landing: `grid[2][px..px+2] = '.'`, plus headroom `grid[1][px] = '.'`. Record spine seed `(px, py)`.
3. **Carve the descending spine (Spelunky-style walk, down-biased).** From the cursor, repeatedly:
   - pick a horizontal `direction ∈ {-1,+1}` (reverse it on hitting `col 2` / `col 41`, the Spelunky wall-bounce rule),
   - carve a **horizontal shelf** of length `randint(4,10)` at the current row, carving body-height EMPTY **and one row of headroom above** so the corridor is walkable/jumpable,
   - then **drop** `randint(1,3)` rows (≤ jump height so it's climbable back), carving the column down. Mark these as "descending" — never require an "up" move *out* of a pit deeper than 3.
   - Stop when `cy >= ROWS-3`. Append every carved cell to `spine`.
   This single continuous walk is **fully connected by construction** (every carved tile adjoins the previous).
4. **Fatten into rooms (density + exploration).** For each of the 4 bands, dig `randint(5,8)` **branches** off random spine cells: short horizontal runs (length 3–8) with headroom. Branches add the side pockets that hold off-path crystal clusters and treasure. Carving is clamped to the interior so the border stays solid.
5. **Density check.** Compute `solid_ratio = count('#')/(44*18)`. Target band is `0.45–0.85`. If **too sparse** (`< 0.50`), stop fattening / refill a random branch; if **too dense** (`> 0.82`), dig one extra branch. (Carved descending caves naturally land in-band; this is a guard, not the main mechanism.)

### (b) Guarantee full reachability — verify-and-repair loop

This is the heart of constraint 2. We reuse the **exact** flood from the solvability test (`_cave_reachable(grid, start, doors_open, jump=3)`).

```
def verify_and_repair(grid, px, py, max_iters=8):
    for _ in range(max_iters):
        reach_open = cave_reachable(grid, (px,py), doors_open=True)
        # 1. REFILL: any EMPTY tile the player can't reach becomes solid rock.
        changed = False
        for r in 1..16:
            for c in 1..42:
                if grid[r][c] == '.' and (c,r) not in reach_open:
                    grid[r][c] = '#'; changed = True
        # 2. RECONNECT: if a needed region got orphaned, dig a ≤3-tile
        #    stair from the nearest reachable standing tile toward it.
        if region_orphaned(grid, reach_open):
            dig_stair(grid, from=nearest_reachable, to=orphan)
            changed = True
        if not changed:
            return True   # stable + fully connected
    return False          # give up → caller regenerates with next seed
```

- **Refill** is the connectivity safety net the research prescribes: keep only the component containing the entrance; everything the flood can't reach becomes wall. This *cannot* strand a required objective because objectives are placed **after** this pass (step c) only on reachable standing tiles.
- **Reconnect** handles the rare case where fattening split a useful pocket: dig a staircase whose every step is within the 3-tile jump, then re-flood.
- **Door-closed correctness:** before placing the door, we additionally assert the chosen switch is in `cave_reachable(grid, (px,py), doors_open=False)`. If not, move the switch upstream on the spine and re-verify. This guarantees the closed-door solvability leg.
- **Generate-and-test outer loop:** if `verify_and_repair` returns `False`, **reject the seed and regenerate** (increment seed). We never repair indefinitely.

### (c) Place objectives (top → bottom flow)

All placement draws from **standing tiles** = reachable EMPTY tiles with SOLID directly below (so the player can stand/land there). Computed once from the post-repair flood, sorted top→bottom.

1. **Player `P`** — written at `(px, 2)` (already carved). Top entrance ✔.
2. **Switch `s` + Door `D`** — choose a standing tile in **band B** (upper-middle) that is reachable with doors closed; place `s` there. Place its `D` on the spine **just below** the switch's band, sealing the route to the deepest cluster + exit. Verify: `s ∈ reach_closed` and `D` actually separates `E` from `P` when closed (optional, for a meaningful gate). Matched single pair keeps it simple and never circular.
3. **Crystals `*` (8–12, clustered along the route).** Distribute across all 4 bands so the player must descend through the whole level to 100% them. Place **dense small clusters of 2–3 on the critical path** (pull the player forward) and **larger off-path clusters in branch pockets** gated behind a hazard or a tricky jump (risk/reward). At least one crystal sits in the open near spawn (`player_col < 6` region) for the discoverability contract. Every `*` is asserted in `reach_open`.
4. **Exit `E`** — the **lowest, then left-most reachable standing tile** with `exit_row >= 12`, placed *behind* the door `D` so the level genuinely requires the switch. (Mirrors the research rule "put the exit at the argmax of the distance map" while honoring our bottom constraint.) Assert `E ∈ reach_open`.
5. **Pickups.** `A` ammo ×≥3 spread one-per-band (enough to clear mandatory enemies — never require more shots than ammo available). One theme-appropriate power-up from `{p,g,z}` placed **before** the section it enables. One `$` treasure tucked in a hard-to-reach branch (pure bonus, non-blocking).

Ordering invariant enforced: `player_col < switch_col`-region-reachable, switch upstream of its door, exit downstream of the door, exit near bottom — matching the first-cave contract and the reference "collect all crystals → switch flips border green → exit opens" flow.

### (d) Hazards + props so it reads authored

- **Hazards (`^` spikes, `~` acid):** placed as **short runs (2–5 tiles) on shelves and at pit bottoms**, never as a single impassable wall across the only route. Hazards are *passable but damaging* (1 dmg), so they pace tension without breaking the reachability proof — but we still **re-run `cave_reachable` after hazard placement** (hazards are walkable so they don't change the flood, but powerups/air tanks could change intent). Bias by theme: blue→`^`, rust→`~`, tech→mixed. Density rises per band (≥3 hazards total for the first-cave contract). Place a "rest" shelf with no hazard after each hazardous band (rhythm groups).
- **Enemies (`M` crawler, `F` flyer):** `M` on shelves with clear lanes, `F` over open vertical gaps. Count rises per band. Pair each forced enemy with reachable ammo.
- **Air tanks (`O`):** optional, off the forced firing line (research: shooting one kills the player); if placed in a lane, the renderer/dressing flags it. Never on the critical path's shooting axis.
- **Visual dressing:** generated caves can reuse the engine's authored-prop vocabulary (`DressingPiece` kinds: `beacon`, `mine_sign`, `generator`, `terminal`, `warning_post`, `vacuum`, `zapper`, `elevator_frame`, `clear_blocks`, `room_label`, `eye_turret`, `bat_perch`) anchored to landmark tiles (switch room, exit chamber, hazard corridor) — this is what makes it "read" as hand-built. (Dressing is optional/cosmetic for v1; the tile layout is the deliverable.)

### (e) Apply a theme

```
THEMES = {
  "blue_rock":   dict(background=(9,12,22),  accent=(80,190,255),
                      hazards=["^"], powerups=["p","z"], enemies=["M","M","F"]),
  "rust":        dict(background=(12,10,18), accent=(255,188,80),
                      hazards=["~","^"], powerups=["p","g"], enemies=["M","F"]),
  "gray_tech":   dict(background=(8,15,13), accent=(120,255,155),
                      hazards=["^","~"], powerups=["g","z"], enemies=["F","M"]),
}
```
Pick a theme by seed (or caller request). The theme sets `CaveSpec.background`/`accent` and biases steps (d). Grid contract is identical across themes, so all themes pass the same solvability + density tests; only the palette and prop/hazard flavor differ (satisfies constraint 3 and the engine's distinct-hue palette test).

### Final emission
Return `CaveSpec(name=<themed name>, layout=tuple("".join(row) for row in grid), background=theme.background, accent=theme.accent)` — drop-in for `CAVES`.

---

## 4. Grading Rubric (0–100, self-grade vs gold)

Each generated level is auto-scored by `grade(rows, theme)` (the prototype's `grade()` grown to this rubric). **A level must score ≥ 85 AND pass both hard gates (solvable + density) to be accepted.** Below 85 → reject + regenerate.

| # | Criterion | Weight | Measure (pass condition) |
|---|-----------|:---:|---|
| 1 | **Solvability** | 25 | `s ∈ reach(doors_closed)` for every switch; every `*` and `E ∈ reach(doors_open)`. All-or-nothing (the hard gate). |
| 2 | **Density** | 12 | `0.45 ≤ count('#')/792 ≤ 0.85`. Full marks in-band; 0 if out. Bonus shape: peak at 0.60–0.70. |
| 3 | **Top entrance** | 12 | `player_row ≤ 3` (full at row 2) AND `P` above all `*`/`E` rows. |
| 4 | **Descent** | 8 | Crystals distributed across ≥3 of 4 bands; mean crystal row increases vs entrance; `exit_row ≥ 12`. |
| 5 | **Connectivity** | 12 | `fraction of open tiles in reach(doors_open) == 1.0` (everything walkable). Linear partial credit. |
| 6 | **Objective placement & flow** | 10 | `player_col < switch-region < exit_col`; switch upstream of its door; exit behind door; ≥1 crystal near spawn; `8 ≤ crystals ≤ 12`; `ammo ≥ 3`; `doors ≥ 1`. |
| 7 | **Hazard / prop authoring** | 8 | `≥3` hazards; hazard density rises per band; ≥1 "rest" shelf; no hazard fully blocks the sole route; ammo ≥ enemies requiring it. |
| 8 | **Theme coherence** | 6 | `background`/`accent` match chosen theme; hazard/powerup/enemy mix drawn from theme table; distinct dominant hue vs other themes. |
| 9 | **Reference resemblance** | 7 | Single scrolling 44×18 playfield; one `P`, one `E`, ≥1 `s`/`D` pair, ≥8 `*`; descend-then-switch-then-exit beat present; reads like a carved cave room (not platforms over void). |

**Letter bands:** A ≥ 90, B 80–89, C 70–79, D 60–69, F < 60. **Acceptance threshold = 85 (A-/B+) + both hard gates green.**

This rubric is also the **gold comparison**: scoring the three authored `CAVES` should land them in A territory; a generated level that scores within ~5 points of the authored mean is "indistinguishable enough" to register.

---

## 5. Integration + Iteration Plan

### Where it lives
- **v0 (now):** `scripts/gen_crystal_cave.py` — standalone prototype with `generate(seed)`, `reachable()`, `grade()`, `render_png()` (already present). It imports nothing from the game except the renderer, so it runs fast in a loop.
- **v1 (promotion):** extract the verified algorithm into `src/game/crystal_caves_gen.py` exposing:
  - `generate_cave(seed: int, theme: str = "blue_rock") -> CaveSpec` — returns a fully-verified, theme-stamped `CaveSpec`.
  - `grade_cave(spec: CaveSpec) -> dict` — the rubric scorer.
  It imports `CaveSpec` from `crystal_caves_entities` and the shared reachability flood (factor `_cave_reachable` into a small shared module so generator and test use **one** implementation — no drift).

### How it plugs into `CaveSpec` / `CAVES`
- Output is a `CaveSpec`, identical shape to authored caves. To use generated caves:
  - **Append mode:** extend `CAVES` with N generated specs (the engine cycles `level_index % len(CAVES)`), or
  - **Replace mode:** swap in a generated list behind a config flag (`Config.PROCEDURAL_CAVES = True`), seeding from `Config.SEED` for reproducibility.
- Because the output respects every grid invariant, `_load_level()`, `get_state()` (119-dim), the vec env, and the renderer all work unchanged. Generated caves automatically inherit dressing only if we register `CAVE_DRESSING` entries (optional).

### Render + grade samples
- `scripts/render_crystal_caves_gallery.py` already renders authored caves; add a `--procedural --seed N` path that renders generated specs to PNG via the real `CrystalCaves` renderer (the prototype's `render_png` does this). Produces side-by-side gold-vs-generated sheets for eyeballing theme coherence.
- A batch harness (`gen_crystal_cave.py --batch 200 --seed-start 0`) generates N levels, scores each with the rubric, and prints distribution (accept rate, mean score, worst failing criterion). This is our regression dashboard.

### The refine loop
1. **Generate-and-test gate (per level):** `generate → verify_and_repair → grade`; reject < 85 or hard-gate fail, advance seed, retry (cap ~20 attempts/level, else log the seed for inspection).
2. **Test integration:** add `test_generated_caves_are_solvable_and_dense` that generates K seeds × M themes and asserts each passes `_cave_reachable` (closed + open) and the 0.45–0.85 density band — the **same** assertions as the authored-cave tests, so generated levels are held to the identical bar.
3. **Iterate on rubric failures:** run the batch harness, find the lowest-weighted-average criterion (e.g. "hazard authoring" or "theme coherence"), tune that placement stage, re-run. The rubric is the objective function; each refinement should raise mean score and accept rate without regressing the hard gates.
4. **Difficulty / variety knobs:** expose designer knobs (crystal count, band hazard slope, door-gate on/off, theme) so the same engine yields easy intro caves and hard deep caves — mirroring the reference's 16-cave episode arc.

**Definition of done for v1:** `generate_cave(seed, theme)` produces, for ≥95% of seeds across all themes, a `CaveSpec` that (a) passes `_cave_reachable` closed+open, (b) lands in the 0.45–0.85 density band, (c) spawns the player at the top with the exit near the bottom behind a switch-gated door, and (d) scores ≥85 on the rubric — registerable into `CAVES` with no manual edits.

---

## 6. Surface-start entrance (user directive, 2026-06-17)

Reference: Crystal Caves HD opens a level on the **planet surface under space** — Mylo
stands on rocky ground with Earth + the Moon and stars above, a `MINE →` sign points to
the shaft, transmitter pylons line the surface, and a cut-stone-block roof separates the
surface from the mine below.

The generator and renderer model this with `CaveSpec.sky_rows` (default 0):
- The top `sky_rows` rows render as **outer space** (black, parallax stars, Earth + Moon)
  with a **horizon glow band** fading into the planet surface.
- `P` spawns on the **surface ground** (a solid band just below the sky), not inside the
  cave. A **mine-shaft gap** in the surface/roof lets the player descend; the `MINE →`
  sign and pylons dress the surface.
- Below the stone-block roof, the carved descending cave from §3 takes over.

Reachability is unchanged: the surface is walkable, the shaft is a fall into the cave, and
the verify-and-repair flood still proves every objective reachable from the surface spawn.

---

## 7. Platform-network pivot (implemented, 2026-06-17)

User feedback on the first carved output: "more random and less blocky." The
carve-and-fill model left large uniform solid masses. We pivoted the *implemented*
generator (`src/game/crystal_caves_gen.py`) from **subtractive carve** to an
**additive platform-network** model — and it is now the shipping generator:

- The interior starts **mostly OPEN**; the dense textured back-wall carries the
  "full" look, so thin platforms over it read full, not void.
- A varied web of **thin platforms** is threaded across staggered rows with
  randomized length / spacing / vertical jitter / thickness and occasional support
  pillars; gaps widen toward the bottom (difficulty bands).
- **Connectivity** is redefined for this model: every *standing tile* (platform
  top) + objective must be reachable, verified with the shared `cave_reachable`
  flood; floating unreachable platform bits are pruned back to open background.
- **Density target lowered to 0.22-0.50** (was 0.45-0.85) — the level is open by
  design; density is in the platforms + border, not big masses.
- The exit sits in a **sealed chamber whose only entry is a switch-gated DOOR**
  (the prune pass protects the chamber walls), so the door genuinely gates the exit.
- **Three themes** (blue_rock / rust / gray_tech) bias hazard / power-up / enemy
  selection and set the palette.

Result: mean rubric **100/100** across 80 seeds, every level solvable, density ~0.26,
fully connected, door-gated — and it reads like the reference platform maze. The
shared flood now backs both the authored-cave solvability test and the generator.
