# Crystal Caves: retro fidelity direction

The game should retain the feel and animation style of the original 1991
Crystal Caves while increasing fidelity. The approved direction is a detailed
pixel remaster of an industrial arcade cave, with the game as the main focus.

- Author collision tiles at native 32x32 and the compact miner at 24x32. Keep
  the red helmet with yellow trim, warm yellow face, magenta overalls, red boots
  and small arcade proportions. Wall panels span 64x64 pixels, or two tiles.
- Keep chunky continuous platforms, patterned room walls, mixed-color cut
  diamonds, ribbed pipes, air machinery and the black bottom HUD.
  Red rooms have quiet slate-blue platform faces with narrow turquoise rims.
  Other rooms use diagonal steel plates, timber/dark masonry, green/brick and blue cobble with gray diamond walls.
- Anchor colors in the original vivid EGA families. Use a small selection of
  extra shades for each material and gem; build hard pixel clusters and bevels.
  Retain quiet areas so fine details do not compete with hazards and actors.
- Show use and age where it belongs: chipped paint at metal joints, rubbed
  ladder rungs, tarnish around bolts, scuffed machinery, timber grain/nail holes
  and small stone fractures. Draw a few purposeful clusters rather than random
  noise over every surface. Keep active indicators, gems and HUD symbols clear.
- The miner's helmet, boots and clothes can carry small stable scuffs across
  animation frames. Creatures gain membranes, segments and natural markings;
  organic surfaces do not receive the equipment's corrosion treatment.
- Prioritize recognizable silhouettes: the green creature has a hunched back,
  sloping forehead, heavy toothed jaw, tiny arms, belly, curling tail and clawed
  feet. Author its 24x64 art directly, without squeezing a wider sprite. It is
  anchored at the existing ground patrol's feet, in corridors with full-route
  headroom, with a matching 64px body collider. The bat, green two-eyed flying slime,
  upright pink snake and gray walking rock use separate four-pose cycles.
  Assign appearances at spawn; retain them while moving.
- The flying slime is a tapered green 24x24 body with two stacked eyes and quiet
  gray lower nubs. The snake is pink 24x32 art with an upright S-neck, yellow eye
  and curled tail. Anchor all four snake poses to the unchanged 24x24 patrol's
  feet; preserve clear headroom and horizontal facing. Legacy `eye_flyer` and
  `slug_enemy` filenames remain stable.
- Mylo's airborne firing pose keeps its standing shot's head, torso and muzzle,
  with bent legs in the same 24x32 canvas. Retain hurt priority and the original
  shot window; this is presentation rather than changed jump or gun mechanics.
- The bat has thin near-black scalloped wings, a small body and tiny yellow eyes.
  Keep its four native 24x24 poses and a restrained cool edge for dark-wall contrast.
- Surviving hits turn the green creature into a hard white/ink silhouette for
  thirty simulation steps, preserving its four poses and floor anchor. Its
  defeat uses four discrete white bone poses over seventy-two steps, with small
  signed pixel drift from the fatal shot. Keep the bones opaque until expiry and
  leave enough room above them for the existing point text.
- The main mine uses mixed-size angular dark rocks with small uneven fillers and
  restrained mortar, pink-brown timber with broken grain and darker undersides,
  worn posts and shaded diagonal braces, narrow chain shafts, gray doors with yellow windows and red/green
  indicators, and short orange torch animations. It connects all sixteen caves.
  Sparse cyan/magenta barrels and red DANGER plates occupy quiet timber bays;
  reserve all entrances, labels, torches, chains, spawn and support arms first.
- Pickup silhouettes follow the reference: a dropped raygun, domed chest,
  lettered P capsule, framed red upward gravity arrow with yellow outline, and
  red STOP sign. Ribbed service runs have
  horizontal sections and elbows connected to AIR machines.
  Use purposeful connected service runs instead of repeated dangling pipe stamps.
- Shots use four native 16x8 gray/silver capsule poses, centered on the existing
  10x4 projectile body and flipped with shot direction. Keep the discrete tail
  vane and white highlight; remove the unsupported gold muzzle star/fan from
  Unity presentation while retaining wall-impact feedback.
- Gems and power pickups remain anchored to their authored sites. Animate their
  hard pixel glints from simulation steps without wall-time sine bobbing.
- Distinguish equipment by form: exposed switch levers, pressure vessels for AIR
  and tall door frames. Place generators, terminals, signs, crates and hanging
  lamps in clear bays. The title illustration shares the actual game artwork.
- Cave exits have green panels, yellow trim, a small dark upper window and a
  red/green lock lamp. Use two-tile art only with clear headroom, preserving the
  original base; tight exits and the current red/blue keyed gates stay compact.
  Reserve the expanded top cell before placing any scenery. Main-mine entrances
  retain their separate numbered-door art and dimensions.
- Silver wall grilles have tall dark vertical slots in a worn square frame.
  Reserve their full 2x2 footprint, side clearance and an empty row below; they
  are non-colliding machinery scenery.
- AIR vessels use a two-tile silver body where the original base has clear
  headroom; tight sites retain the compact vessel. Keep the red AIR lettering,
  two-pose red gauge and red/yellow/green lower indicator readable. Gauge poses
  use the simulation clock and hold in reduced-motion mode. These are visual
  dimensions; the existing pickup site and interaction rules remain unchanged.
- Use cyan/magenta barrels with silver bands in red, green, gray and purple
  machinery rooms. Dents and tarnish gather around bands and edges. Wide red
  plates have yellow DANGER or REVERSE GRAVITY lettering and occupy clear bays
  near actual hazards or gravity pickups. Reserve both cells of the sign.
- Green rooms use broad muted green platform faces with narrow pale upper rims,
  occasional exposed-edge chips and warm rust-brick walls. Keep material wear
  sparse, and let the colored gems, machinery and actors carry the fine detail.
- Keep point filtering, no mipmaps, integer display scaling and pixel snapping.
  Lighting remains bright and readable. The shading is drawn into the artwork.
- Use a four-frame run cycle at the existing ten poses per second and two climb
  poses. Animation uses discrete sprites and the authoritative state. Keep the
  direct camera follow and small pixel feedback effects. The world view is twenty tiles wide with integer scaling and centered margins. Mylo visibly carries his
  raygun; firing raises it and climbing stows it.
- Reverse gravity flips only Mylo vertically. Select jump/fall relative to the
  authoritative gravity sign and mirror contact shadows, landing/shot effects
  and helmet points around the sprite center. Returning to normal gravity or
  the main mine restores the normal presentation.
- Play uses a red outer border that changes to green when all crystals are
  collected, alongside a written exit-ready cue. Restart belongs in Settings.
- HUD placement is a saved choice: bottom by default for the classic presentation,
  or top for the modern layout. Money, gun/ammo, hearts and crystal counters stay
  together in a reserved 32px strip. Camera bounds and hints follow that choice.
- Moving lifts have a square gray gauge housing, a worn yellow landing lip and
  paired cyan struts. Use four discrete mechanical poses in a 32x32 footprint.
  Green floor thorns are single crooked organic spikes with five rising poses;
  their authoritative state controls both visibility and collision, including
  reduced-motion mode. Keep them distinct from the static silver spike clusters.
- Pale rooms use mixed lavender block sizes with bright upper facets, scuffed
  corners and black walls edged by thin copper pipes. Machinery rooms use broad
  quiet slate platforms with cyan stitched edges and gray vertical wall ribs.
  Keep these patterns distinct from the red panels and blue cobbles.
- Purple machinery rooms use long staggered purple wall planks, recessed slots
  in silver beams, and narrow reflective columns. Keep the slots smaller than
  the miner; the existing solid tile remains fully opaque and colliding. Worn
  joints, small edge chips and tarnish belong around the metal's seams.
- Blue cobble rooms have paired crooked magenta vines ending in rounded loops
  and angular purple mushroom caps above green curled roots. These are scenery,
  as are the reference's silver columns, not enemies or climbable ladders.
  Flared columns belong in ribbed machinery rooms. Reserve every cell of tall
  scenery, with a clear cell on either side, and keep AIR service runs visible.
- Blue cobble platforms use fewer, larger irregular rounded rock clusters,
  bright upper facets, quiet navy joins and pale worn upper rims. Paint in world
  coordinates so rocks continue across 32px tile boundaries. All authored solid
  cells remain completely opaque and keep their original collision shape.
- Orange ceiling stalactites are narrow, crooked cones, bright on their left
  facet. Their positions come from Python: waiting, dropping in discrete 17 Hz
  poses, then breaking on a solid surface. Reduced motion must not freeze them.
- Use the classic speaker sound programs and one voice with source priorities.
  Preserve their frequencies, pauses and rapid gating; import mono PCM without
  lossy compression. Keep the original material's provenance. The rejected
  generic music loop is removed; no HD soundtrack or video narration is shipped.
- Cave previews show the actual game art. Menu typography and layout can retain
  modern clarity while title/headings and play counters stay recognizably pixel.
- Keep decoration independent of collisions and learning observations. The
  level layouts, checkpoint data and progress eligibility rules remain separate.
  The explicitly requested control correction uses `playfeel.py` to tune Unity
  sessions to a slower walk and lower, slower jump; Python remains authoritative
  and the training engine's defaults remain intact. Normal expeditions remove
  training cutoffs. `classic_levels.py` supplies small ladder repairs in two
  Unity caves so the lower jump preserves access to their objectives.

`src/unity_bridge/visuals.py` defines the main art and cave layers,
`visual_materials.py` contains shared shading/wear, and `visual_props.py` defines
the additional native props. `visual_creatures.py` defines the active creature
roster and its animations.
`visual_rooms.py` supplies the pale, ribbed, purple girder and green/brick materials;
`visual_scenery.py` supplies cave plants, silver supports, warning placement and
their clear footprints. `visual_equipment.py` defines the tall AIR vessels,
two-tone barrels and lettered warning plates.
`visual_doors.py` defines colored cave doors and their clear-headroom reservation.
`visual_mine_dressing.py` reserves and paints the main mine's sparse equipment;
`visual_projectiles.py` supplies the gray capsule poses; `visual_mine_rocks.py`
supplies deterministic, fully opaque rock-wall variation.
`visual_mechanisms.py` supplies lift, thorn and stalactite art; `classic_mechanisms.py`
contains their human-profile rules, leaving training defaults unchanged. Export with
`python -m src.unity_bridge.export_assets`. `RejectedModern/` preserves the
superseded painted experiment outside Unity Assets; it is not part of the build.

Style references, not source assets:
[Crystal Caves (1991)](https://store.steampowered.com/app/358260/Crystal_Caves/)
and [Crystal Caves HD](https://store.steampowered.com/app/1330890/Crystal_Caves_HD/).
The user's [LGR reference video](https://www.youtube.com/watch?v=_WQTGCBZ1FM)
shows the HD remaster. Its frames inform proportions and colors; its recording
and soundtrack are not runtime assets.

The fourth reference series keeps torch flames unchanged above a cool tapered
silver/lavender wall socket. Hover lifts have a broad dark split instrument face,
pale divider and cool worn bevel within their existing footprint; the landing
lip and cyan strut bytes stay exact. Jump/climb boots have no detached sole
pixels below their lifted shapes.

Flying slime defeat uses four authored red stages on the existing 36-step event,
with nine steps per stage and five native pixels of signed drift per stage.
Mylo's existing immunity timer drives alternating twelve-step white/color phases
of his current action pose. White variants retain each pose's exact alpha and
INK outline; the damage sparkle and feedback labels must not obscure them.
Reduced motion keeps an opaque white action pose and holds the first pulse stage.
