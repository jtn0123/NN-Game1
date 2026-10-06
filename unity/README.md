# Crystal Caves

A game-first Unity 6 experience for Crystal Caves. Start at the title screen,
pick one of sixteen caves, and explore with a small HUD over the full-window
cave. Pause, cave selection, options and results have their own menus. Cleared
caves and best scores are saved locally for entirely human runs. AI-assisted
attempts do not earn cleared-cave badges.

Unity uses the 1991 Crystal Caves visual grammar: native 32-pixel artwork,
vivid EGA color anchors and selected extra shades, continuous beveled platforms, patterned panel and masonry walls,
a small red-helmeted miner in magenta overalls, mixed-color diamond gems,
ribbed pipes and industrial equipment. Seven room families distinguish the
sixteen caves. The black bottom HUD uses pixel numerals, a raygun and hearts;
controls live in the pause/options menus. Artwork is authored from code, with
point filtering, no mipmaps, integer display scaling and pixel-snapped motion.
The miner remains 24x32 pixels, with finer helmet, face, clothing and equipment
detail. Its run cycle retains four discrete frames at the existing cadence.
The miner carries a visible raygun in standing, running and airborne poses.
Firing in midair keeps its bent-leg pose, with unchanged gun and muzzle position.
The active roster now includes a hunched olive-green creature with a heavy jaw,
tiny arms and curling tail, a dark thin-winged bat with tiny yellow eyes, a green
two-eyed flying slime, an upright pink snake and a gray walking rock with purple legs. Appearances stay fixed throughout
a patrol. Generators, terminals, signs and hanging lamps dress clear room bays;
AIR vessels and exposed switches have separate silhouettes. A pixel title scene
uses the actual miner and green creature.
The red rooms use quiet slate-blue platforms and 64-pixel wall panels spanning
two collision cells. The other room materials use diagonal steel plates,
dark masonry behind timber, staggered brick behind green platforms, and blue cobble against gray diamond-patterned walls.
Modern polish stays within that arcade style: animated poses, crystal glints,
pixel action feedback, acid bubbles, bright cave previews, readable menus and
a reduced-motion option.

Play now starts in a traversable main mine, with sixteen numbered cave doors,
mixed-size angular stone walls, worn braced timber platforms, chain shafts and animated torches.
Walk to a door and press E, Down or Enter to explore its cave. Returning to the
mine keeps Mylo at that doorway. Cleared entrances turn green using the same
human-only completion records as cave selection. The grid remains an optional
shortcut. This is a new connecting mine, not the original overworld map.

The view uses twenty native tiles across, integer pixel scaling and a compact
black footer, with centered margins on wider displays. The camera follows
without the previous lead and smoothing. Gun pickups now resemble dropped
rayguns; treasure uses a chest, P a capsule, G a green lettered block and freeze
a STOP sign. Service pipes connect into AIR machines through ribbed bends.

Human cave sessions now give the tall green creature its full 64px collision
body, five ordinary-hit health and a faster charge when it sees Mylo or is hit.
A powered shot defeats it immediately. The gray rock sleeps until it sees Mylo,
wakes, then patrols; ordinary shots cannot defeat it, while a powered shot can.
Walls block awareness. Bats reverse their horizontal flight at irregular
intervals. These reference-informed rules live in `classic_game.py`, separate
from the training game's simplified creatures. The original enemy-hurt and
empty-gun effects are used. The complete original enemy roster and original
cave layouts have not been reproduced.

Python advances the repository's game simulation and optionally runs a trained PyTorch policy.
Unity sessions use a classic control profile: 140 world pixels per second,
immediate stopping on release and an approximately 80-pixel, one-second jump.
This is reference-informed tuning, not a claim of exact 1991 engine parity.
The training engine's class defaults and checkpoint files are unchanged.
This migration still requires the local Python runtime; the built app is not yet
a standalone simulation. The rejected painted experiment is archived outside
Unity's Assets directory in `ArtSource/RejectedModern`; it is not imported or
included in the player. The direction is documented in `ArtSource/STYLE.md`. Source art is
`src/unity_bridge/visuals.py`, exported with
`python -m src.unity_bridge.export_assets`.

## Launch on this Mac

From the repository root, using the project's Python 3.10–3.12 environment:

```bash
python scripts/unity_pilot.py
```

The launcher builds the player if needed, starts the loopback bridge, then starts
Unity. Closing the game or pressing Ctrl-C in the launcher stops both processes.
Unity 6000.3.25f1 is the version used for this project. An installed editor is required
for the initial build; an existing player only needs Python and the project dependencies.

```bash
# Rebuild after changing Unity code or the source artwork/levels.
python scripts/unity_pilot.py --build

# Watch the existing compatible CNN checkpoint, or take control of it.
python scripts/unity_pilot.py --cnn-state --model models/crystal_caves/crystal_caves_best.pth

# Record completed human episodes in the existing demonstration format.
python scripts/unity_pilot.py --record-demos .Codex/artifacts/unity-pilot/demos
```

The AI Lab is a separate optional menu, accessible from the title or pause screen.
F2 opens its overlay during play. Watch the agent is enabled only when an explicitly
supplied checkpoint loads successfully. Take the controls pauses for a deliberate
human resume. Starting or restarting a cave always returns to human control.
It runs inference; it does not train, save or overwrite model weights. The default
observation is the rich 295-feature state; `--legacy-state` selects the older
119-feature state. `--cnn-state` selects SpatialDQN; otherwise the current default
dueling network is used. Checkpoints with different experimental observations or
network settings require matching configuration and are rejected when incompatible.
The bridge uses the repository's existing trusted-local-checkpoint loading rules.

Controls: A/D or left/right to move; Space, W or up to jump/climb; J/X to shoot;
E/down to interact; Enter also enters a nearby mine door; Esc/P to pause; M to mute. Restart is in Settings, accessible
from the pause menu; R has no restart action. Simultaneous jump and
shoot follows the game's discrete action space: shooting takes priority. Losing
window focus pauses the cave. Level changes and restarts begin a fresh human run.
Up climbs a mine chain, Down descends, and releasing holds position. Arrow keys and Enter also operate the cave-selection menu. Sound volume and last
selected cave are saved locally. The fullscreen setting is available in Options.
C opens cave selection, O opens Options, and H returns home from menus. Enter
activates the focused menu action. Arrow keys navigate menus. In Settings,
Up/Down selects a row, Left/Right adjusts it, Tab/Shift-Tab changes pages, and
Escape goes back. Controllers use the stick, A/B and LB/RB. F11 opens the
fullscreen Keep/Revert confirmation; F12 saves a screenshot.
During play, the outer border is red until the last crystal is collected, then
green alongside `ALL CRYSTALS / EXIT OPEN`. The exit still needs to be reached.
Health and the all-crystals-then-exit win condition remain in effect. Normal
Unity play removes the AI training time and inactivity cutoffs and their HUD
countdown. Two caves have small ladder connections for the lower jump; the
training maps retain their original layouts. Geometry checks reach every
crystal, switch and exit in all sixteen caves with the classic profile; these
checks exclude enemies, hazards and full winning-route ordering.

Classic sound effects are rendered from the original Episode 1 shareware tone
programs: crystal, jump, shot, ammo, treasure, damage, switch, power-up and
level cues. The player uses one speaker voice with the source priorities,
pauses and rapid gating. WAVs import as uncompressed mono PCM. The unrelated
two-second music loop and invented landing thud are removed. The reference
[LGR video](https://www.youtube.com/watch?v=_WQTGCBZ1FM) shows Crystal Caves HD;
its soundtrack and commentary are not included. Source sound files and their
provenance are in `ArtSource/ClassicSounds/PROVENANCE.md` and retain the original
game copyright.

## Presentation settings

Display, Graphics, Sound and Accessibility pages share a live cave preview.
Opening Settings pauses Python simulation; changing render FPS never changes
its 60 Hz step clock. Settings migrate the old HUD, motion and volume choices,
then save to the versioned `cave-settings-v1` PlayerPrefs record. Defaults reset
presentation, audio and accessibility; the default display is staged separately
and still requires Apply/Keep. Test and replay modes do not write this record.

| Page | Controls |
| --- | --- |
| Display | Window/fullscreen, remembered window size, 15-second Keep/Revert, VSync, 30–360 FPS or unlimited, FPS counter, automatic or 1–6× integer scaling, adaptive/classic/wide framing |
| Graphics | Current Retro / Enhanced Retro / Low Power presets, independent shake/particles/glints, Off/Low/Normal density, environmental motion, local light, contact shadows and strength, brightness, original/warm/cool palette, CRT scanlines and phosphor |
| Sound | Master/effects volume, mute, paused-menu preview of the actual raygun sound |
| Accessibility | Reduced motion, damage flashes, top/bottom HUD, standard/large HUD, safe margin, high contrast, letter/symbol item cues, screenshot HUD preference |

Current Retro preserves the accepted artwork, animation cadence and 60 FPS
presentation. Lighting, environmental dressing motion, grading and CRT are off.
Enhanced Retro adds restrained gem/torch light and sparse drips, service lamps,
dust and vapor. Low Power caps rendering at 30 FPS and reduces decoration.
Manual graphics overrides become Custom; display resolution, audio and HUD
preferences survive preset changes. Reduced motion suppresses decorative
motion while lifts, bullets, enemy poses and essential hazard cues continue.
World grading and CRT run before the HUD and menus, so their colors stay crisp.

On macOS, VSync uses the app's `CAMetalLayer.displaySyncEnabled` and display-refresh
pacing, avoiding the Unity VSync semaphore stall reproduced on this Mac.
[Apple documents the layer synchronization behavior](https://developer.apple.com/documentation/quartzcore/cametallayer/displaysyncenabled).
The universal arm64/x86_64 native helper is compiled from `Native/CaveDisplay.mm`
by `CaveNativeBuild` during a macOS build, using Xcode Command Line Tools. Other
platforms use Unity's standard VSync setting. VSync is off by default.

Settings > Screenshot pauses the cave, lets H / controller X toggle the HUD,
and saves with F12 / A. Escape / B returns to the originating screen. F12 also
saves during normal play. PNGs go to `Application.persistentDataPath/Screenshots`
with a visible saved-path confirmation; capture controls and notifications are
excluded from the saved image. On this Mac the directory is normally
`~/Library/Application Support/NN Game1/Crystal Caves/Screenshots`.

For an isolated native presentation review, the player accepts
`--settings-smoke OUTPUT_DIR --settings-case menus|display|render|vsync`, optionally
with `--settings-state SNAPSHOT_JSON`. `--presentation-settings JSON_FILE` loads
an ephemeral configuration. Stage CLI inputs/outputs outside Documents for
native reviews; copy the evidence back afterward. Editor checks run through
`CrystalCaves.Pilot.Editor.CaveSettingsChecks.Run`. Review modes do not prove a
physical controller; joystick mappings still need testing on the target device.

A compact [native visual review](../docs/unity-review/README.md) includes captures,
42 settings/native checks and portable reproduction commands. Run
`python scripts/review_unity.py menus|display|render|vsync` with one case at a time
after building the player. Fullscreen requires an unlocked active desktop.

## Open in Unity Editor

Open the `unity/` directory as a Unity project, open
`Assets/Scenes/CrystalCaves.unity`, and press Play after starting:

```bash
python -m src.unity_bridge
```

The editor defaults to port 8766. The standalone player also accepts
`--bridge-port PORT`; the launcher forwards `--port PORT` to both processes.
The initial cave visible while disconnected is a static preview; gameplay starts
only after the bridge connects. Reconnection starts a fresh session.

The runtime tilemap is constructed from the existing Python-authored levels.
Painting a Unity tilemap does not edit the authoritative level data in this version.
A standalone simulation and an authoring/export workflow are future migration
steps; Python remains authoritative, including the Unity session's control profile.

## Assets and build

```bash
python -m src.unity_bridge.export_assets
python scripts/unity_pilot.py --build-only
```

The exporter uses `src/unity_bridge/visuals.py` for the Unity game's authored
stone, characters, pickups, mine props and backdrops. It reuses the original
pixel-lettered title. `classic_audio.py` exports the source speaker programs
and their priorities. Terrain follows the
canonical grid, including spike and acid hazards, and is sliced into individual Unity
Tilemap cells. Texture imports preserve original dimensions and point filtering;
acid surface animation is a separate presentation layer. Dressing has no colliders.
The classic profile also has seven green proximity thorns in six caves. They rise
through five authoritative poses when Mylo enters the column above them, retract
when he leaves, and damage only their emerged body. Static silver spikes remain.
All thirteen vertical lifts use the square gauge housing and cyan underside
struts seen in the reference. Mylo can ride either direction and jump off; shafts
no longer act as invisible ladders. Obstructed lifts reverse instead of embedding
the player in a floor or ceiling. These rules are isolated from the training engine.
Settings > Accessibility > HUD Position switches the money, gun/ammo, hearts and crystal counters between
classic bottom placement (default) and modern top placement. The saved choice
moves the reserved camera strip, hints and world text with the layout. Select
the row and use Left/Right or Enter to switch placement.
The Python research renderer and simulation rules remain separate from this art.
The video comparison pass adds two distinct room kits: pale lavender blocks over
black copper pipe walls in Dripstone Hollow (which has reverse gravity), and quiet
slate platforms over vertical gray ribs in Twin Vaults. Blue cobble ledges retain
their green tops and gain a thin cyan side rim. All solid footprints stay on the
same grid. Six orange ceiling stalactites in Stalactite Chasm and Sunken Grotto
release when Mylo enters their unobstructed downward column. Their narrow tips
fall in discrete 17 Hz poses, use swept collision, and break at solid surfaces.
They use original speaker program 24; restarting the cave restores them.
Resources, scenes, package manifests and `.meta` files are kept with the project;
Library, Logs, UserSettings and Builds are ignored. Build/player logs live under
`.Codex/artifacts/unity-pilot/`.

Menu typography uses Chakra Petch Regular and SemiBold from the
[Google Fonts source](https://github.com/google/fonts/tree/main/ofl/chakrapetch),
licensed under the SIL Open Font License. The license is included in
`Assets/Resources/Fonts/OFL.txt`.

## Verification

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m pytest tests/test_unity_bridge.py tests/test_classic_fidelity.py tests/test_classic_mechanisms.py tests/test_classic_stalactites.py tests/test_reference_*.py -q
```

The bridge tests compare observations, positions and rewards with direct calls to
the authoritative engine under both classic and training profiles, measure walk
pace and jump height/duration, cover every cave and its assets, check terrain/hazard alignment
and dressing placement, validate AI takeover, and
verify pickup-specific sound events, original sound decoding/PCM exports,
absence of training cutoffs and the two ladder repairs. They also
exercise malformed requests over a real socket. Commands are newline-delimited
JSON objects: `snapshot`, `mine` with an optional `cleared` cave-index array, `reset` with `level`, `mode` with `human`/`ai`, or `step`
with an array of 1–8 discrete actions. Invalid action batches do not advance any
frames. The bridge listens only on 127.0.0.1; one client owns one session. It never
steps autonomously, so renderer delays cannot silently advance the game.

The built player has an opt-in native smoke run:

```bash
# Start the bridge in another terminal first, then run this from the repo root.
"unity/Builds/Crystal Caves.app/Contents/MacOS/Crystal Caves" \
  --smoke-report /tmp/crystal-caves-smoke.json \
  --capture /tmp/crystal-caves-smoke.png \
  -logFile /tmp/crystal-caves-player.log
```

The real Unity player connects, steps, loads all sixteen caves, tests AI/takeover
when a checkpoint is available, captures title/caves/options/play/pause views
plus representative cave palettes, firing feedback and the optional AI overlay,
checks single-speaker sound priorities and the results transition after an actual
game-over, verifies restart and captures explicitly labelled all-gems border and completion UI fixtures,
walks to a main-mine entrance, enters the real cave and returns to the same doorway, then exits. Add `--visual-smoke` to capture a sequence of actual movement frames.
Add `--mechanism-smoke` to board and ascend the Freight Lift using real inputs,
capture its active floor thorn, verify both HUD/camera placements, and check that
the top layout leaves the full world visible. The mechanism tests also cover all
thirteen authored lifts, inverted gravity, obstruction reversal, thorn activation,
retraction and collision. They do not prove complete winning routes through every cave.
Add `--reference-smoke` to walk and jump from the real Stalactite Chasm spawn,
trigger a ceiling trap, dodge left, and verify its floor impact without taking
damage. It captures every four simulation steps, with matching JSON snapshots,
and captures every cave from 02 through 16 (plus the default cave 01 play capture).
It also verifies gravity-relative orientation, airborne poses, ceiling/floor
contact shadows and normal-state restoration, and captures closed/open tall
exits. Those gravity and door close-ups are explicitly presentation fixtures,
not gameplay routes. On macOS,
launching with `open -n "unity/Builds/Crystal Caves.app" --args ...` uses Launch
Services. The second five-pass reference loop completed five fresh windowed smokes
and fresh player captures, including the previous shadow correction. Earlier
launches stalled intermittently before the first frame; this pass does not claim
to fix that startup issue. See `.Codex/ui-ux-audit/2026-10-04-five-pass-2/`
for the current captures, comparison and validation. `-force-gfx-direct` selects direct Metal graphics
submission. The screenshot hook needs a windowed player;
`-batchmode` does not provide its display backbuffer on this Mac.
The fixtures do not complete a gameplay route or save achievements. For manual
captures, start a player with `--capture /absolute/path/review.png` and press F12;
it saves the screen and a JSON state snapshot alongside it. Rendering and connection
updates continue in the background; losing focus pauses ordinary gameplay.

The current artwork includes seven active room families. The Freight Lift, Acid
Vents and Switchback Spire use purple planks/slotted silver beams. Both blue
cobble caves contain looped pink vines and purple mushrooms; those caves, the
purple rooms and Twin Vaults also have silver supports. These fixtures are
non-colliding scenery. Their complete footprints stay clear of authored objects,
ladders, hazards and AIR pipes. The equipment pass adds seven two-tile AIR vessels
where headroom is clear; four tight sites retain the compact vessel. Their
readable AIR labels and two gauge poses preserve the original pickup base and
current interaction rules. Reduced motion holds the gauge pose. Red, green,
gray and purple rooms use twenty-four worn cyan/magenta barrels. Twenty-three
DANGER plates and two REVERSE GRAVITY plates sit near actual authored markers;
their complete two-cell footprints stay clear of other equipment and pipes.
Green platforms now use quieter muted faces, pale rims and warm rust-brick walls.
The asset review accounts for 150 exported sprite files in 186 labeled entries,
including all sixteen cave previews. The native comparison and full asset board
are in `.Codex/ui-ux-audit/2026-10-04-five-pass-2/`.

The five-pass follow-up adds four square silver ventilation grilles, corrects
Mylo's upside-down reverse-gravity presentation, gives thirteen clear-headroom
exits tall green/yellow-trimmed door art, redraws both blue caves with larger
irregular cobbles, and replaces the green bat face/G pickup with the reference's
dark-winged bat/upward gravity arrow. Three tight exits and all thirteen current
keyed gates keep compact dimensions. Gate colors, lock rules, pickup sites,
collision grids, observations and model files retain their existing behavior.
Expanded door headroom stays clear of scenery and AIR pipes. All 91 targeted
tests, style/type checks, five fresh macOS builds and five fresh native smokes
passed. The per-iteration report records the evidence and scope limits.

The second five-pass series adds a thirty-step white hit flash and four
seventy-two-step bone breakup poses to the green creature. It removes 629
unsupported dangling pipe placements while retaining the AIR service runs,
adds four barrels and two DANGER plates to clear main-mine bays, and replaces
the gold shot streak with four gray capsule poses. Capsule artwork is 16x8,
centered on the unchanged 10x4 projectile body; shot direction controls its flip.
The bridge filters only muzzle sparks from Unity presentation, leaving the
engine's events and wall impacts intact. Pickups remain anchored while their
discrete glints animate. The native smoke also checks hit/bone/capsule fixtures,
pixel snapping, effect expiry, restoration and authored pickup positions.
All 115 final tests, Ruff, Black and mypy checks passed, along with five fresh
macOS builds and five fresh native smoke runs. The report records the separate
manual-launch startup limitation and distinguishes presentation fixtures from
the real-input gameplay checks.

The native player also accepts an opt-in `--render-replay <json>` with
`--replay-output <folder>` for video capture. The JSON contains `fps` and a
`frames` array of complete protocol snapshots recorded from `CaveSession`.
This mode renders the existing game, camera and HUD without a bridge or changes
to gameplay preferences, writes numbered PNGs and a capture report, then exits.
Use a temporary local folder for its input and output: this macOS session stalled
opening replay files directly in the workspace, while temporary-file capture
succeeded. The 31.2-second native gameplay video, recorded inputs and encoding
scripts are in `.Codex/artifacts/unity-pilot/game-video-2026-10-04/`.

The third five-pass reference series adds broken timber grain and shaded support
faces, irregular dark mine rocks, a green two-eyed flying slime, an upright pink
snake, and a bent-leg airborne firing pose with warmer HUD hearts. The snake's
24x32 art shares the unchanged 24x24 patrol body's floor anchor. Existing
creature aliases, movement, damage, observation data and model files remain
intact. All 143 focused tests and style/type checks passed, with five fresh
macOS builds and thirteen matched native snapshot views per build. These art
captures use bounded room batches and omit the startup splash only in replay
mode; they do not verify ordinary startup or live room transitions. The existing
ordinary-startup stall remains unresolved. Before/after views, the full 151-sprite
sheet, inputs and exact limits are in
`.Codex/ui-ux-audit/2026-10-04-five-pass-3/reference-pass-report.md`.

The fourth five-pass series adds cool tapered torch sockets, a broad divided
lift instrument face, clean lifted-boot silhouettes, a four-stage red slime
defeat pulse, and pose-matched white damage flashes for Mylo. The lift's landing
lip and cyan strut poses remain exact. Slime feedback retains its 36-step life,
score and fatal shot direction; Mylo retains the original 70-step immunity.
Render metadata exposes that existing timer without changing the simulation.
Damage/score labels sit above the affected artwork, and the old gold damage
sparkle no longer covers the player. Reduced motion holds a white action pose
and the first pulse stage instead of blinking or drifting.

All 173 focused tests, Ruff, Black and focused mypy checks passed. Five ordered
passes produced six fresh macOS builds: the fifth was rebuilt after a native
close-up exposed the overlapping damage sparkle. Each build rendered the same
22 real-input snapshots; 11 exact native pixel checks verify selections,
animation phases and anchors. Terrain/environment textures remain byte-identical.
The full 174-sprite sheet, matched views and validation are in
`.Codex/ui-ux-audit/2026-10-04-five-pass-4/reference-pass-report.md`.
These isolated replay captures verify art, not ordinary startup or live room
transitions; the previously observed startup stall remains unresolved.
