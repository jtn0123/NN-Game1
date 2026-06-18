# Crystal Caves Visual Fidelity Plan

**Status:** Active execution plan
**Created:** 2026-06-17 · synthesized from a research workflow (4 web researchers on the 1991 Apogee original + 2 code gap-analysis agents)
**Owner:** Lead designer (visual/UX)
**Targets:** `src/game/crystal_caves.py`, `src/game/crystal_caves_art.py`, `config.py`
**Relationship to backlog:** This document is the research-grounded **execution layer** on top of `.Codex/crystal-caves-reference-visual-backlog.md` (V001-V110). The backlog enumerates *what* the game is missing against the 1991 DOS references; this plan is the *ordered, addressable, file-anchored "how"* an engineer or agent picks up next. Every task here cross-references the backlog IDs it advances. **Do not duplicate the backlog** — when a task is fully covered there, this doc only adds the concrete code target, the exact RGB values, and a testable acceptance criterion.

> **Progress note (2026-06-17):** CCV-08's back-wall fill is already implemented (`_draw_wall_fill` in `crystal_caves.py`), replacing the black starfield with a dense, dim, theme-colored parallax masonry layer across all three episodes. The remaining first-sprint items (CCV-01..07) refine on top of it. See the "Execution log" at the bottom.

## 1. Purpose

Make the Crystal Caves mode read like an authored 1991-Apogee EGA cave room instead of a sparse prototype of thin platforms floating in a black void. The work is clean-room: we capture the **visual grammar and EGA palette conventions** (a hardware color set and unprotectable layout ideas), never original assets, sprites, fonts, character names, or level geometry.

Three things drive the entire plan, in priority order:

1. **Kill the black void.** The room must read as solid rock with carved tunnels, not islands over emptiness.
2. **Period-authentic HUD.** A thin bottom bar with score, ammo, and hearts — not a 6-compartment debug dashboard.
3. **Foreground/background separation + terrain mass.** Chunky carved terrain with strong outlines and bright top lips, sitting in front of a dim back-wall layer.

## 2. Visual North Star (clean-room style targets)

### 2.1 Palette — EGA 16 (hardware standard, free to replicate)

Author every color by stepping between these 16 values; never anti-alias, keep hard pixel edges.

| Idx | Name | RGB | Role examples |
|---|---|---|---|
| 0 | black | `0,0,0` | outlines, void seam, HUD bg |
| 1 | blue | `0,0,170` | blue-cave wall body |
| 2 | green | `0,170,0` | girder shade |
| 3 | cyan | `0,170,170` | metal trim shade |
| 4 | red | `170,0,0` | danger plate, lava body |
| 5 | magenta | `170,0,170` | authentic cave void fill |
| 6 | brown | `170,85,0` | rust slab, earth |
| 7 | light-gray | `170,170,170` | gun/pipe body, stone |
| 8 | dark-gray | `85,85,85` | shadow/mortar |
| 9 | bright-blue | `85,85,255` | blue stone highlight |
| 10 | bright-green | `85,255,85` | HUD numerals, girder face, grass lip |
| 11 | bright-cyan | `85,255,255` | metal ledge lip |
| 12 | bright-red | `255,85,85` | hearts, rust top-lip, red gem |
| 13 | bright-magenta | `255,85,255` | dashed ledge lip on magenta void |
| 14 | yellow | `255,255,85` | lava crust, sign text, gem |
| 15 | white | `255,255,255` | `$`, specular, gem sparkle |

The current `_episode_palette()` uses off-grid tones (e.g. `18,34,150`, `255,158,52`). These read fine but should be **migrated toward EGA-true anchors** and **extended with named role keys** (`wall_fill`, `wall_accent`, `ledge_lip`, `edge_dark`) so the whole room — not just pipes/lips — is themed.

**Three theme registers** (from research): (a) blue cobblestone + magenta void; (b) gray stone-brick back wall + brown/rust slabs; (c) green-girder industrial over black. Map these onto the three existing episodes.

### 2.2 Tile grammar

- **Rooms are a filled mass, carved.** Start solid, cut corridors. The carved passage is a flat fill **color**, never large black regions: magenta `170,0,170` (blue theme), brick back-wall showing through (stone theme), or black only for the girder theme.
- **Walkable top-lip rule (universal "you can stand here"):** every standable surface gets, top-down: (a) 1px black outline along the very top, then immediately above the void (b) a 1px **dashed bright lip** — bright-magenta `255,85,255` on magenta void, else bright-cyan `85,255,255` / bright-green `85,255,85`. Exposed side/bottom edges get a 1px black outline.
- **Edge vs interior distinction is what makes a mass read solid.** Interior fill tiles carry no outline; only exposed edges/corners do. The code already computes `open_left/right/top/bottom` — the rendering just needs to lean on it harder.
- **Platform thickness:** a standable block fills ≥50% of tile height visually, not a 5-10px trim strip.
- **Back-wall layer:** a dim, non-colliding masonry/brick fill behind everything (`_draw_wall_fill`). Foreground solids draw on top; where there's no solid, this fill is what shows — so there is no black.
- **Hazards occupy real volume:** spikes and lava/acid fill the full tile height with a dark base, bright crust/warning line, and a black outline matching solid tiles.

### 2.3 HUD layout (sparse, period-authentic)

One thin bottom strip on solid black. Left→right, **no word labels**: `$` + bright-green score digits ... gem icon + count ... gun icon + bright-green ammo digits ... up to 3 bright-red hearts. No CAVE/EXIT/MYLO labels, no compartment dividers, no always-on controls line. Numerals are chunky monospaced pixels (`85,255,85`). Reserve ~6 score digits, ~2-3 ammo digits, exactly 3 heart slots (lost hearts → dark outline).

### 2.4 Sprite / prop language

- 1px pure-black outlines on every sprite; small dark drop-shadow so objects sit in the world.
- **Gems are the brightest, largest non-character objects** — faceted diamonds with a white upper-left highlight facet, mid-tone body, dark lower-right shade, black outline. 4-5 color variants for visual variety.
- Props cluster in groups of 3-8 (barrels+tools+sign, machinery zone, supply cache) — never one isolated object per screen.
- Signs are diegetic decals baked into the wall (red plate, black border, yellow/white text), not floating UI overlays.

### 2.5 Room composition rules

- 35-55% (lean denser in hard rooms; backlog targets up to ~85% built) of tiles are solid environment; thread a 1-2 tile air corridor through the mass.
- Read order: foreground solid terrain → midground props/hazards → goal-layer gems leading the eye toward the exit/airlock chamber.
- Crystals routed in clusters of 3-8 through risk pockets; collect-all opens the exit.
- Hazards get a warning-sign landmark 2-4 tiles before them.
- Exit/airlock is a distinct, recognizable tech-framed chamber, inert until full clear.

## 3. Current State vs Target

| Dimension | Current (code) | Target |
|---|---|---|
| Background | `_draw_wall_fill` masonry **added**, parallax dim fill; `_draw_cave_depth` shapes are near-invisible (alpha 22-54). | Dense themed fill edge-to-edge; readable depth layer; no black void exposed on camera pan. |
| Palette | 3 palettes, off-EGA-grid, only feed pipes/lips/sparse depth. | EGA-anchored, extended with `wall_fill`/`wall_accent`/`ledge_lip`/`edge_dark` permeating the whole room. |
| Terrain mass | Surface tiles render a ~10px top strip + thin interior bands; reads as a rail. | ≥50% tile-height solid body, beveled edges, 3px black outline, dashed bright top-lip. |
| Hazards | Spikes occupy bottom ~13px; acid a 17px mid-strip. Both have dead `return` + unreachable code below. | Full-tile-height hazard volume, dark base, bright crust, black outline, denser teeth. Remove dead code. |
| HUD | 6-compartment SCORE/AMMO/CRYSTAL/CAVE/EXIT/MYLO dashboard + always-on controls line; `HUD_HEIGHT=52`. | Clean label-less black bar (score, crystals, ammo, hearts), no dividers, controls moved to pause/title, `HUD_HEIGHT≈38`. |
| Levels | Sparse ASCII grids: `#` platforms over `.` void. | Authored dense rooms, 60-85% terrain, prop clusters. |
| Gems / enemies | Gems ~24px diamond icons; enemies 2x sprites, no outline/shadow. | 32px gems with shadow + halo; enemies with 2px black outline + drop shadow. |

**Net:** the visual-identity reset is *started* (wall fill exists) but not finished — the depth layer is too faint, terrain still reads thin, and the HUD is still a dashboard. The highest-leverage moves are finishing the background/terrain/HUD trio.

## 4. Execution Plan

Ordered by impact. **Visual-identity-reset items first.** Each item is independently shippable. Navigate by `file:method` (line numbers drift).

### P0 — Visual identity reset

#### CCV-01 — Make the cave-depth back layer readable
- **Change:** In `_draw_cave_depth`, raise alphas (dark `54→90`, mid `22→50`, accent `38→70`) and add a high-contrast `rock_light` second pass on ~2 of 6 polygons so carved-wall silhouettes read.
- **Target:** `crystal_caves.py:_draw_cave_depth`
- **Impact:** High · **Effort:** Small · **Backlog:** V001, V019, V021
- **Accept:** No contiguous pure-black `(0,0,0)` region larger than 1 tile (32×32) outside hazards/outlines on a paused ep-0 frame.

#### CCV-02 — Extend episode palettes with room-role colors
- **Change:** Add `wall_fill`, `wall_accent`, `ledge_lip`, `edge_dark` to each palette and migrate body tones toward EGA anchors. Ep-0 blue: `wall_fill=(8,28,80)`, `ledge_lip=(106,150,255)`, `edge_dark=(0,0,40)`. Ep-1 brown/rust: `wall_fill=(64,32,16)`, `ledge_lip=(255,85,85)`, `edge_dark=(40,16,0)`. Ep-2 gray/tech: `wall_fill=(40,48,64)`, `ledge_lip=(85,255,255)`, `edge_dark=(20,24,34)`. Feed `wall_fill` into `_draw_wall_fill`'s base.
- **Target:** `crystal_caves.py:_episode_palette` (consumed in `_draw_wall_fill`)
- **Impact:** High · **Effort:** Small · **Backlog:** V002, V003, V008, V017
- **Accept:** Each palette exposes ≥10 named keys including the four new roles; three episodes render visibly distinct dominant hues.

#### CCV-03 — Thicken terrain into carved masses with strong outlines
- **Change:** In the `is_surface` branch, fill the whole tile with `edge_dark`, build a body ≥50% tile-height, bump the outline 2px→3px, add a 2px light bevel top-left + `edge_dark` bottom-right. Texture interior tiles so masses read behind surfaces.
- **Target:** `crystal_caves.py:_draw_solid_tile`
- **Impact:** High · **Effort:** Medium · **Backlog:** V005, V006, V015, V018, V020
- **Accept:** Standable tile body covers ≥16 of 32 vertical px; exposed edges carry a 3px black outline; interior tiles carry no top-edge outline.

#### CCV-04 — Simplify HUD to a period-authentic bottom bar
- **Change:** Replace the 6-compartment dashboard with clean label-less clusters on solid black (score `$`+digits, crystal icon+count, gun+ammo, 3 hearts). Remove divider loop and CAVE/EXIT/MYLO labels. Drop the always-on controls line. `HUD_HEIGHT` 52→38.
- **Target:** `crystal_caves.py:_draw_hud`, `:HUD_HEIGHT`
- **Impact:** High · **Effort:** Medium · **Backlog:** V022, V023, V024, V096
- **Accept:** No `CAVE`/`EXIT`/`MYLO`/`ARROWS` strings render in the play HUD; 0 vertical divider lines; `HUD_HEIGHT == 38`; HUD tests green.

#### CCV-05 — Move controls off the play HUD into pause/title
- **Change:** Render the controls reference only when paused or on the title screen, not in `_draw_hud`.
- **Target:** `crystal_caves.py:_draw_hud` controls block, `render_title_screen`
- **Impact:** High · **Effort:** Small · **Backlog:** V024, V109
- **Accept:** No controls text during normal play; controls visible on pause overlay; test asserts absence in play HUD.

#### CCV-06 — Spikes: full-tile hazard volume + remove dead code
- **Change:** Dark-red base, ≥6 teeth from 50% tile-height up, yellow warning stripe, 3px black outline. Delete unreachable code after the `return`.
- **Target:** `crystal_caves.py:_draw_spike_tile`
- **Impact:** High · **Effort:** Medium · **Backlog:** V071
- **Accept:** ≥24px hazard height, black outline, ≥6 teeth; no unreachable statements.

#### CCV-07 — Acid/lava: full-tile pool + remove dead code
- **Change:** Maroon base, 8px bright crust, 4-5 orange bubbles, 3px black outline. Delete unreachable block after the `return`.
- **Target:** `crystal_caves.py:_draw_acid_tile`
- **Impact:** High · **Effort:** Small · **Backlog:** V042, V072
- **Accept:** Full 32px height with bright crust line + black outline; no unreachable statements.

#### CCV-08 — Restructure `_draw_background` to lead with theme — **DONE (baseline)**
- Back-wall fill leads the background; star pixels removed. Refinement: ep-0 column/beam framework and ordering pass remain.
- **Target:** `crystal_caves.py:_draw_background`, `_draw_wall_fill`
- **Backlog:** V001, V019

### P1 — Rooms that read as authored

- **CCV-09** — Lean on edge detection for carved interior-wall variants (`_draw_solid_tile` interior branch). V015, V016.
- **CCV-10** — Triple prop density and cluster props (`_draw_level_dressing`, `CAVE_DRESSING`). V050, V051, V061, V087.
- **CCV-11** — Make gems screen anchors: 32px, drop-shadow, accent halo, highlight facet (`_draw_pickups`). V074, V075, V076.
- **CCV-12** — Outline + shadow enemies (`_draw_enemies`). V020, V063.
- **CCV-13** — Redesign cave layouts toward 60-85% density with threaded corridors (`CAVES`). V004, V005, V032, V081. *(Large.)*
- **CCV-14** — Per-episode theme previews on the title/menu (`render_title_screen`). V078, V079, V093.

### P2 — Polish & regression protection

- **CCV-15** — Sprite-table rectangularity validation test (`crystal_caves_art.py` SPRITES). V091.
- **CCV-16** — Centralize theme color aliases in the art palette (`LAVA_CRUST`, `GRASS_LIP`, `RUST_TOP`). V008.

## 5. Measurable Acceptance (test-ready metrics)

Snapshot/pixel tests over a rendered headless frame (render to an offscreen surface, sample with numpy).

| Metric | Current | Target | How to measure |
|---|---|---|---|
| Black-void ratio | High on void rooms | **< 0.10** | `(0,0,0)` non-outline pixels ÷ play-area pixels |
| Background-fill coverage | Partial | **≥ 0.85** | 1 − black-void ratio (HUD excluded) |
| Foreground terrain density | ~0.2-0.35 | **0.55-0.85** per cave | `solid_tiles / total_tiles` |
| Props per visible screen | ~3 | **8-15** | Count prop draws in one viewport |
| HUD simplicity | 6 + controls | **no labels, 0 dividers** | label strings absent in `_draw_hud` |
| HUD height | 52px | **38px** | `HUD_HEIGHT == 38` |
| Terrain outline | 2px | **3px** exposed edges only | inspect `_draw_solid_tile` |
| Gem footprint | 24px | **≥ 32px** + shadow + halo | bounding-box on rendered gem |
| Hazard volume | ~13-17px | **≥ 24px** + black outline | pixel-height on hazard tiles |
| Dead code | unreachable in spike/acid | **0** | coverage/lint |
| Episode distinctness | blue-dominant | 3 distinct hues | mean play-area hue per episode |

## 6. Suggested First Sprint

1. **CCV-01** — Boost the cave-depth layer (Small).
2. **CCV-02** — Extend palettes with role colors (Small).
3. **CCV-04** — Simplify the HUD (Medium).
4. **CCV-05** — Move controls off the play HUD (Small).
5. **CCV-03** — Thicken terrain into carved masses (Medium).
6. **CCV-06 / CCV-07** — Full-tile spikes and acid + delete dead code (Medium/Small).

CCV-13 (layout redesign, Large) is the natural follow-on sprint.

## 7. Execution log

First sprint shipped 2026-06-17 — black-void ratio measured at **0.07-0.08** across all three episodes (target < 0.10), all 779 tests green, lint + mypy clean.

| Date | Item | Status | Notes |
|---|---|---|---|
| 2026-06-17 | CCV-08 | ✅ baseline | `_draw_wall_fill` added; black starfield replaced with dim parallax masonry across all 3 episodes; star pixels removed; restructured `_draw_background` to lead with the themed fill. |
| 2026-06-17 | CCV-02 | ✅ done | Each `_episode_palette` gained `wall_fill`/`wall_accent`/`ledge_lip`/`edge_dark`; `_draw_wall_fill` now drives off these role colors. |
| 2026-06-17 | CCV-01 | ✅ done | `_draw_cave_depth` alphas boosted (54→86 / 22→60 / 38→74) + a brighter `rock_light` rim on every third recess so carved silhouettes read. |
| 2026-06-17 | CCV-04 | ✅ done | `_draw_hud` rebuilt as a label-less period footer (score / crystals / ammo / lock icon / hearts), dividers + CAVE/EXIT/MYLO labels removed, `HUD_HEIGHT` 52→38; test `test_hud_uses_fixed_compartments` rewritten → `test_hud_is_clean_period_footer`. |
| 2026-06-17 | CCV-05 | ✅ done | Always-on "ARROWS MOVE…" controls line removed from the play HUD (controls remain on the title screen). |
| 2026-06-17 | CCV-06 | ✅ done | `_draw_spike_tile` rebuilt with a full-tile dark base, warning crust, 6 teeth, black frame; unreachable post-`return` code deleted. |
| 2026-06-17 | CCV-07 | ✅ done | `_draw_acid_tile` rebuilt as a full-tile molten pool (maroon body, bright crust, bubbles, black frame); unreachable post-`return` code deleted. |
| 2026-06-17 | Tests | ✅ added | `test_background_fill_kills_the_black_void` (< 0.15 ratio guard) and `test_episodes_have_distinct_dominant_hues` regression tests added. |
| 2026-06-17 | CCV-17 | ✅ done | Grass/vine organic detailing: `grass` role color added to EP1 (teal moss) + EP2 (lime); new `_draw_ledge_growth` paints a moss fringe, swaying blades, and hanging vines on walkable ledges. EP3 (tech) stays clean metal. |
| 2026-06-17 | CCV-18 | ✅ done | `_draw_switch_wires` draws a taut cable from each switch to its nearest door — green core once thrown, amber while armed — making the switch→target relationship readable (ref frame 1). |
| 2026-06-17 | CCV-19 | ✅ done | `_draw_gravity_overlay`: while the gravity power is active, a violet edge vignette, upward-floating debris, and a REVERSE GRAVITY banner signal the altered field (ref frame 2). |
| 2026-06-17 | CCV-20 | ✅ done | Audio: new `crystal_caves_audio.py` (`CrystalCavesAudio`) — procedural chiptune SFX (numpy→`Sound`), headless/CI/training-safe (self-disables under `SDL_VIDEODRIVER=dummy`). Wired to gem/pickup/win/door/switch/gravity/jump/land/shoot/damage/lose events + a quiet looping music bed. 5 audio tests. |

**Updated grades after CCV-17..20:** Organic detailing **D → B+** (grass/vines on 2 themes); Mechanic visuals **C- → B** (switch wires + gravity treatment); Audio **F → B** (full procedural SFX + music, headless-safe). Overall UI/UX vs reference **B- → B/B+**.

### Remaining first-sprint / follow-on

- **CCV-03** (terrain into carved masses, Medium) — surface tiles already carry lips/bolts/outlines and read well against the dimmer wall; a 3px-outline + thicker-body pass is the next refinement.
- **CCV-09..12** (interior edge variants, prop density, gem anchors, enemy outlines) — P1 polish.
- **CCV-13** (dense layout redesign, Large) — the natural follow-on sprint; pairs with CCV-03/09.

## 8. UI/UX grade vs references (2026-06-17, satisfies backlog V110)

Graded against three user-supplied reference frames from the modern Crystal Caves
(purple-brick room w/ grass + switch-wire + minimap; reverse-gravity black-room;
red-industrial low-gravity room). Bar = "how close to that look/feel," noting we
render with pygame primitives in a clean-room educational/NN-training project.

| Dimension | Grade | Evidence vs reference |
|---|---|---|
| HUD / status bar | **A-** | References (frames 2-3) use exactly `$ <score>` green + gun+ammo green + 3 red hearts; our new footer matches and cleanly adds crystal count + lock icon. |
| Background fill / density | **B** | Dense themed masonry now fills the screen; references carry richer per-tile art and integrate grass into the fill. |
| Terrain mass & readability | **C** | Biggest gap: ours are still thinnish platform strips; references are thick masses you stand *on top of*. (CCV-03/13) |
| Organic detailing (grass/vines/moss) | **D** | Signature reference look (frame 1: bright green grass lips + hanging vines) is entirely absent in our renderer. (new: CCV-17) |
| Gems / pickups | **B-** | Good faceted diamonds; references are larger/brighter with stronger white highlight + drop shadow. (CCV-11) |
| Hazards | **B** | Full-tile spike walls + molten acid pools landed this sprint. |
| Props & clustering | **B-** | Rich set (chests, lamps, barrels, terminals); references add value labels (1K/2K) and tighter clusters. (CCV-10) |
| Enemies | **B-** | Varied roster exists; references show bigger, more iconic silhouettes (dino, robot, shark). |
| Signs / diegetic UI | **B** | DANGER/ACID/LOW-G signs present and embedded, close to reference REVERSE/LOW GRAVITY decals. |
| Theme variety / palette | **B** | 3 distinct registers; references hint at more (purple-brick, red-industrial, reverse-g black). |
| Mechanic visuals (clear blocks, switch wires, gravity rooms) | **C-** | `clear_blocks` is decorative only; no switch→target wire (frame 1); gravity rooms lack a distinct visual treatment. (new: CCV-18/19) |
| **Audio / music** | **F** | None. No `pygame.mixer`, no SFX, no music anywhere in the project. (new: CCV-20) |

**Overall UI/UX vs reference: B-** (up from ~C+ pre-sprint). HUD is the standout; dragged down by terrain organic-detailing and the total absence of audio.

### New tasks surfaced by this grade

- **CCV-17** (P0) — Grass/vine lips: bright green moss fringe on walkable surfaces + hanging vines under ledges (frame 1). The single highest-impact "feels like Crystal Caves" addition. Target `_draw_solid_tile` surface branch + a new `_draw_ledge_growth`.
- **CCV-18** (P1) — Switch→target wire: draw a connecting line from each switch to the door/elevator it drives (frame 1). Target `_draw_lever_switch` + door render.
- **CCV-19** (P1) — Gravity-room visual treatment: reverse/low-gravity rooms get a distinct backdrop + flipped hazard cues (frames 2-3).
- **CCV-20** (P1) — Audio layer: optional, headless-safe `pygame.mixer` with procedurally-generated chiptune SFX hooked to existing `visual_events` (pickup/shoot/door/damage/switch/gravity/exit); silent under headless/CI/training.
