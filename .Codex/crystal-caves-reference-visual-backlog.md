# Crystal Caves Reference Visual Backlog

**Created:** 2026-06-18  
**Scope:** Clean-room visual and UX backlog for making the NN-Game1 Crystal Caves mode look much closer to the provided 1991 DOS Crystal Caves references.  
**References:** User images #1-#7 from the Codex clipboard.  
**Current comparison gallery:** `/tmp/nn_game1_crystal_caves_reference_compare`

## Short Verdict

The references are not just "more detailed" than the current build. They use a different visual system:

- Screens are dense tile worlds, not sparse platforms over a black backdrop.
- Each level has a complete theme: wall fill, edge tiles, ledge caps, props, hazards, pickups, and enemies all match.
- Terrain is chunky and architectural. Walkable areas are thick, readable shapes with strong black outlines and bright top lips.
- Props are part of level composition: signs, switches, crates, barrels, pipes, doors, grates, keys, levers, elevators, torches, and machines are placed in clusters.
- Gems are large, shaded, outlined, and embedded into the room design.
- The HUD is simple and period-authentic: black footer, chunky pixel numerals, ammo icon/count, hearts. The current compartmented HUD is readable but too modern/debuggy.
- The current build still reads as a space/cave prototype because of the large black voids, thin blue platforms, isolated props, and repeated column shapes.

The list below is deliberately large. P0 items are the visual identity reset; P1 items make screens read like authored Crystal Caves rooms; P2 items are polish or regression protection.

## Reference Observations

| Image | Key lessons |
|---|---|
| #1 | Gray metal/base theme with vertical ribs, thick blue-gray terrain, white pillar platforms, gems placed under ledges, small clear hazards, bottom HUD. |
| #2 | Brick/green cave theme, full brick background, green terrain fill, low-gravity sign, exit/machine cluster, grates, pipes, crates, pickup clusters. |
| #3 | Red panel industrial theme, chunky rounded wall tiles, cyan ledge outlines, gray grates, sign near danger, large monster silhouette, pipes below. |
| #4 | Dark rock/lava theme, stone-wall texture, molten pools with bright yellow crust, falling-rock sign, torches, ladders/cages, boulder, robot hazards. |
| #5 | Blue rock theme, diagonal gray patterned background, rocky blue terrain, grass lips, vines/drips, crates, mushrooms, small monsters, secret-looking gold block. |
| #6 | Reverse-gravity/dark tech theme, black room fill, white block tiles, orange/brown pipe seams, many gem pockets, sign embedded in level path. |
| #7 | Tool/debug reference still shows authentic tile grammar: purple plank walls, item toolbar, minimap, switch/laser relationship, barrels, spikes, doors, inventory icons. |

## Massive Work List

| ID | Priority | Area | What to change | Evidence / reason |
|---|---|---|---|---|
| V001 | P0 | Visual identity | Replace the current black starfield/space backdrop in normal caves with dense room-specific wall fills. | References #1-#5 almost always fill the whole screen with tile texture; current start/hazard shots are mostly black void. |
| V002 | P0 | Tileset system | Build a real per-episode tileset catalog instead of one procedural rock/platform renderer. | Every reference theme has its own full visual kit: gray metal, brick, red panels, stone/lava, blue rock, white tech, purple plank. |
| V003 | P0 | Theme completeness | For each episode, define wall fill, solid edge, ledge top, underside, corner, platform column, prop accent, hazard, and background pattern. | References do not mix random props onto one generic cave tile. The theme carries the whole screen. |
| V004 | P0 | Terrain density | Make screens 60-85 percent built environment by area, with carved corridors rather than floating fragments. | References are dense map slices. Current gallery leaves large unused black regions. |
| V005 | P0 | Terrain silhouettes | Replace thin platform strips with chunky terrain masses and cutouts. | References #2, #3, #5, #6 are made of thick carved shapes; current platforms look like rails. |
| V006 | P0 | Ledge readability | Add strong top lips, black outlines, and underside shadows to every walkable edge. | Reference ledges have bright green/cyan/yellow lips and dark underside separation. |
| V007 | P0 | Tile scale | Standardize all tile art to a chunky 16x16 or 32x32 pixel-grid look with integer scaling. | Reference sprites and tiles share a consistent pixel density; current sprites/tiles sometimes feel from different scales. |
| V008 | P0 | Palette control | Create named palette families for each level theme and stop using mostly blue/cyan/black. | Current build is dominated by black/blue; references use red, brick, green, gray, purple, lava, and white-tech worlds. |
| V009 | P0 | Wall fill | Add brick wall fill like image #2 with staggered rounded brick courses. | This is a signature repeated screen texture missing from current game. |
| V010 | P0 | Wall fill | Add red rounded panel tiles like image #3. | The red-panel theme is visually distinctive and would immediately move the game closer. |
| V011 | P0 | Wall fill | Add dark stone wall tiles like image #4. | Current gray/blue cave blocks do not capture the black-rock/lava episode feel. |
| V012 | P0 | Wall fill | Add blue rounded rock tiles like image #5. | Current episode 3 gray/purple still does not resemble the blue rock texture. |
| V013 | P0 | Wall fill | Add white/lavender tech-block tiles like image #6. | Reverse-gravity/tech screens need the white block + black room identity. |
| V014 | P0 | Wall fill | Add purple plank/metal strip wall tiles like image #7. | Gives another strong episode or test-room visual identity. |
| V015 | P0 | Edge tiles | Implement edge-specific tile variants: top, bottom, left, right, inner corner, outer corner, vertical shaft edge. | References have hand-authored edges; current solid tiles often look like repeated rectangles. |
| V016 | P0 | Corners | Add curved or beveled corner tiles instead of square cutouts everywhere. | Image #3 red panels and image #5 blue rock use softened/rounded surfaces. |
| V017 | P0 | Ground lips | Add theme-specific lip colors: green grass, cyan metal, yellow lava crust, gray stone, purple trim. | Reference walkable boundaries are readable through bright top-edge strips. |
| V018 | P0 | Platform thickness | Make platforms at least one full tile thick visually, even when collision is one tile. | Reference floors feel solid; current collision surfaces can read as thin floating trim. |
| V019 | P0 | Interior backgrounds | Add non-colliding background wall panels behind foreground solids. | Reference #1 vertical ribs and #5 diamond-pattern walls show depth behind the playable terrain. |
| V020 | P0 | Black outlines | Increase black outline consistency around terrain, props, gems, player, enemies, and hazards. | The references rely heavily on black outlines for DOS readability. |
| V021 | P0 | Camera framing | Compose each screenshot-sized area as a room with foreground, midground, and goal object. | References show self-contained rooms; current scenes feel like a broad scrolling test map. |
| V022 | P0 | HUD | Replace the current verbose HUD in human mode with a period-authentic black footer. | References show score, ammo icon/count, and hearts without labels like CAVE/EXIT/MYLO taking space. |
| V023 | P0 | HUD text | Use chunky green/yellow pixel numerals and icons; avoid small modern debug labels. | Reference HUD numerals are big, simple, and bright. |
| V024 | P0 | HUD layout | Move training/objective help text out of the always-visible play HUD. | Current "ARROWS_MOVE..." line makes the game feel like a prototype. |
| V025 | P0 | Player sprite | Redraw Mylo larger and cleaner with helmet/hair/face/gun silhouette readable at screenshot scale. | Reference player is small but iconic; current player is improved but still a little low-contrast and small. |
| V026 | P0 | Player animation | Add obvious idle, walk, jump, shoot, hit, climb/fall poses with stronger frame differences. | Reference player poses are readable even in busy scenes. |
| V027 | P0 | Gem sprites | Replace current crystal-bubble look with larger cut-gem sprites with black outline, white facets, and colored shadow. | References use big diamond/ruby/emerald sprites as primary screen anchors. |
| V028 | P0 | Gem placement | Place gems in authored pockets, ledges, and risk/reward chains rather than evenly scattered. | References use gems to guide routes and secrets. |
| V029 | P0 | Enemy silhouettes | Build enemy sprites with large distinct shapes: green cyclops, pink worm, robot walker, bat/flyer, mushroom, rock monster. | Current enemies are small/generic compared with images #2-#5. |
| V030 | P0 | Hazard identity | Make lava/acid/spikes occupy real tile pools and set pieces, not only thin strips. | References #3 and #4 use hazards as major screen shapes. |
| V031 | P0 | Signs | Use large red/yellow pixel signs embedded into the map for DANGER, LOW GRAVITY, FALLING ROCKS, REVERSE GRAVITY. | References #2, #3, #4, #6 use signs as visual landmarks. |
| V032 | P0 | Level authoring | Redesign the first cave into screen-by-screen authored rooms matching the reference density. | Current first cave is playable but still reads generated/sparse. |
| V033 | P1 | Gray metal theme | Add vertical ribbed wall panels and blue-gray upper terrain like image #1. | Useful for an intro/tech base room. |
| V034 | P1 | Gray metal theme | Add white pillar platforms with black top caps and angled support shadows. | Image #1 pillars are clearer and more dimensional than current posts. |
| V035 | P1 | Gray metal theme | Add gray-blue wall speckles and subtle vertical stripe variation. | Reference #1 avoids flat fills while staying readable. |
| V036 | P1 | Brick theme | Add green terrain fill with rough edges cutting through brick walls. | Image #2's green caves are a major missing visual pattern. |
| V037 | P1 | Brick theme | Add thick lime platform strips and yellow highlight lines. | Image #2 platforms pop strongly against brick. |
| V038 | P1 | Brick theme | Add vertical chain/seam columns to break large brick areas. | Image #2 uses dark chain seams for structure and rhythm. |
| V039 | P1 | Red panel theme | Add red rounded panels as repeated background blocks with orange highlights. | Image #3 is dominated by this specific wall language. |
| V040 | P1 | Red panel theme | Add cyan dash trim along ledges and platforms. | Image #3 uses cyan trim to separate foreground from wall fill. |
| V041 | P1 | Stone/lava theme | Add gray stone floor blocks with lighter bevels and dashed edge highlights. | Image #4 gray terrain reads as solid stone architecture. |
| V042 | P1 | Stone/lava theme | Add lava pools with dark red body, yellow crust, orange bubbles, and black outline. | Current acid/lava is less theatrical than references #4 and #3. |
| V043 | P1 | Stone/lava theme | Add dark rock wall mosaic tiles behind lava rooms. | Image #4's wall is full of irregular stones, not empty background. |
| V044 | P1 | Blue rock theme | Add blue boulder-tile fill with repeated rounded rocks and dark seams. | Image #5's blue rock is a strong episode identity. |
| V045 | P1 | Blue rock theme | Add gray diamond-pattern back wall where rock is cut away. | Image #5 has a second background layer behind the rock. |
| V046 | P1 | Blue rock theme | Add grass/acid-green top fringe on walkable blue rock. | Image #5 uses green grass strips to make ground readable. |
| V047 | P1 | White tech theme | Add white block clusters with lavender shadows and black void behind them. | Image #6 needs a dedicated reverse-gravity tileset. |
| V048 | P1 | White tech theme | Add orange/brown pipe seams around black rooms. | Image #6 uses colored pipe outlines to define empty corridors. |
| V049 | P1 | Purple theme | Add purple plank wall with rivets and black outlines. | Image #7 would be a good later lab/control-room style. |
| V050 | P1 | Props | Add barrels in stacked clusters with metal bands and shadow. | References #1, #2, #5, #7 use barrels as room filler and cover. |
| V051 | P1 | Props | Add wooden crates with diagonal planks and black outline. | Image #3 and #5 crates have strong period style. |
| V052 | P1 | Props | Add blue doors/elevator doors as large two-tile structures. | Images #2 and #7 use doors as important landmarks. |
| V053 | P1 | Props | Add metal grates/vents in wall and floor. | Images #2 and #3 include grates that break up walls. |
| V054 | P1 | Props | Add torches/fire bowls and wall-mounted lights. | Image #4 has torch/lava lighting that current build lacks. |
| V055 | P1 | Props | Add cages/ladders/arches for mine-specific decoration. | Image #4 uses hanging/cage shapes to create a mined-cavern feel. |
| V056 | P1 | Props | Add pipes that run through walls and turn corners. | References #3 and #4 use pipes as composition lines. |
| V057 | P1 | Props | Add monitors/terminals with tiny pixel displays. | References #1, #4, #7 include readable machine faces. |
| V058 | P1 | Props | Add switch panels with ON/OFF text and visual connection lines. | Image #7 shows switch-to-device relationships; current switches are isolated. |
| V059 | P1 | Props | Add raygun pickups that look like side-view guns from the references. | Guns in #1, #2, #3, #5, #6 are highly recognizable. |
| V060 | P1 | Props | Add key/lever/objective items as map landmarks. | Image #1 key-like object and #2 levers help rooms feel game-like. |
| V061 | P1 | Object clustering | Place props in clusters of 3-8, not one isolated object per screen. | References often have barrels + tools + signs + pickups grouped together. |
| V062 | P1 | Foreground/background | Separate collision tiles from decorative props so rooms can be dense without changing pathing. | Needed to match reference density while preserving NN-friendly physics. |
| V063 | P1 | Enemy scale | Increase enemy sprite footprint and contrast. | Reference monsters occupy about a player-height or more; current enemies can disappear in the room. |
| V064 | P1 | Enemy animation | Add 2-4 frame cycles for worm crawl, robot hover/walk, mushroom idle, rock stomp, bat flap. | Reference enemies feel alive from pose and silhouette. |
| V065 | P1 | Enemy placement | Create enemy set pieces: monster guarding gem, robot over lava, flyer in corridor, worm near barrels. | References use enemies as room composition, not random patrols. |
| V066 | P1 | Giant enemy | Add at least one larger green monster silhouette like image #3. | This would immediately improve screenshot authenticity. |
| V067 | P1 | Falling rock | Add falling-rock/boulder visual object and warning sign. | Image #4 has a clear "FALLING ROCKS" setup. |
| V068 | P1 | Reverse gravity | Add visual treatment for reverse-gravity rooms: black background, white blocks, upside-down readable hazards. | Image #6 is a signature Crystal Caves visual missing from current gameplay. |
| V069 | P1 | Low gravity | Add low-gravity sign and room treatment with vertical shafts/gem pockets. | Image #2 explicitly calls this mechanic out visually. |
| V070 | P1 | Laser/switch | Add switch-linked laser/barrier visuals. | Image #7 shows a diagonal line relationship; useful for puzzles and screenshots. |
| V071 | P1 | Spikes | Redraw spikes as integrated hazard tiles with dark base, not just white teeth on open black. | References place spikes in terrain pockets and tunnels. |
| V072 | P1 | Acid/lava labels | Make hazard signs part of the wall, not floating overlay labels. | Reference signs are tile props in the world. |
| V073 | P1 | Secret language | Add hidden-looking gem pockets, breakable blocks, and suspicious wall patterns. | Reference screens invite exploration through visual hints. |
| V074 | P1 | Collectible variety | Add red, yellow, green, blue gems with distinct facet layouts. | References use color-coded treasures heavily. |
| V075 | P1 | Pickup shadows | Add small black/drop shadows under gems and pickups so they sit in the world. | Current crystals can look like UI icons floating in space. |
| V076 | P1 | Pickup size | Tune pickup sizes so gems dominate more than small props, but enemies/player still read. | Reference gems are large screen anchors. |
| V077 | P1 | Room labels | Use signs for room mechanics only; reduce generic labels like LANDING/POWER unless visually styled like reference signs. | Current signs are improving but still feel instructional/prototype in spots. |
| V078 | P1 | Human-mode UI | Add a period-authentic title/menu/instructions/high-score flow using the same tile language. | The menu should feel like the game, not only a modern wrapper. |
| V079 | P1 | Episode select | Show episode tiles/previews with the actual theme art. | Makes the three-episode identity visible before gameplay. |
| V080 | P1 | Score feedback | Keep floating score text minimal or pixel-font; do not let modern effects dominate. | Reference moment-to-moment UI is sparse and grounded. |
| V081 | P1 | Screen composition | Author first cave as 8 rooms: intro, gem pocket, switch tutorial, hazard tunnel, enemy room, secret, exit room, bonus route. | This maps directly to the repeated room structures in references. |
| V082 | P1 | Negative space | Avoid big empty areas unless they are intentional dark/tech rooms like image #6. | Current negative space makes screenshots look unfinished. |
| V083 | P1 | Navigation clarity | Use terrain and gem placement to guide the player instead of HUD text. | References guide through object placement. |
| V084 | P1 | Exit rooms | Build large exit/elevator chambers with machine panels and clear door sprites. | References #2 and #7 make exits/doors visible and important. |
| V085 | P1 | Vertical shafts | Add climb/drop shafts with repeated side tiles, gems, and hazards. | References #1, #2, #6 use verticality heavily. |
| V086 | P1 | Tile repetition | Make repeated patterns intentional and theme-specific, not procedural noise. | Reference repetition looks authored because the tile art is strong. |
| V087 | P1 | Decorative density | Target at least 8-15 visible decorative objects per authored screen. | Current visible prop count is much lower than the references. |
| V088 | P2 | Renderer structure | Extract a renderer/tile atlas layer so visual passes do not keep bloating `CrystalCaves`. | The next visual work is large enough to need separation. |
| V089 | P2 | Level specs | Move level layouts and dressing to data files or typed level specs. | Hand-authoring reference-like screens inside one class will become painful. |
| V090 | P2 | Tile catalog | Define tile IDs for collision, render sprite, state encoding, and theme variant. | Needed for clear blocks, hidden blocks, grates, lava, reverse gravity, and doors. |
| V091 | P2 | Sprite atlas | Convert current sprite dictionary into organized sprite sheets or sprite groups by theme. | Current catalog is useful but will get hard to manage. |
| V092 | P2 | Clean-room guardrail | Use the references as style targets but redraw original sprites and tiles. | Keeps the project safe while making it feel authentic. |
| V093 | P2 | Screenshot gallery | Add reference-target gallery states for each theme: brick, red panel, lava, blue rock, reverse gravity, purple lab. | The current 12-shot gallery does not yet prove these target looks exist. |
| V094 | P2 | Visual metrics | Add tests for background fill ratio, foreground tile density, prop count, gem count, and black void budget. | This directly catches the current biggest mismatch. |
| V095 | P2 | Theme tests | Add tests that each episode uses a distinct dominant palette and tile family. | Prevents all caves from drifting back to one blue/black look. |
| V096 | P2 | HUD tests | Add tests for footer simplicity: score digits, ammo count, hearts, no debug help line in human mode. | Protects the period-authentic HUD. |
| V097 | P2 | Object tests | Add screenshot tests for signs, doors, grates, barrels, crates, and switches appearing in authored rooms. | Props are a major part of the reference look. |
| V098 | P2 | Hazard tests | Add screenshot tests for lava pool, spike tunnel, falling-rock room, and reverse-gravity room. | These are high-attention visual states. |
| V099 | P2 | Player tests | Add sprite catalog tests for each player pose and a render metric that the player is high-contrast against each theme. | Player readability is fragile in dense rooms. |
| V100 | P2 | Enemy tests | Add sprite/placement tests for at least five distinct enemy silhouettes. | The current enemy variety is not enough for the target look. |
| V101 | P2 | Palette snapshots | Store lightweight palette histograms for reviewed screenshots. | Cheaper than full golden images and useful for CI. |
| V102 | P2 | Golden screenshots | Once a theme reaches B-level, store small golden reference images under tests fixtures. | This prevents backsliding after a big visual pass. |
| V103 | P2 | Side-by-side review | Add a script that renders current gallery next to labeled target notes, not copyrighted images. | Helps future passes stay aligned without embedding user-provided reference art. |
| V104 | P2 | Acceptance checklist | Add a per-screenshot checklist: dense background, chunky terrain, 3+ props, 3+ gems, readable hazard, clean HUD. | Makes subjective visual review less hand-wavy. |
| V105 | P2 | Performance guard | Make decorative rendering optional/headless-safe and cheap enough for training. | Visual density should not hurt NN training mode. |
| V106 | P2 | Data/state safety | Keep added visual-only props out of state unless they affect gameplay. | NN compatibility remains important. |
| V107 | P2 | Accessibility | Keep hazards readable by shape as well as color. | References do this with spikes, signs, lava crust, and black outlines. |
| V108 | P2 | Sound-ready hooks | Add event hooks for pickup, shoot, door, damage, lava, and gravity switch sounds later. | Audio will matter once visuals get closer. |
| V109 | P2 | Minimap/dev overlay | If a modern minimap/debug overlay is kept, make it a toggled dev mode like image #7, not the default play UI. | The target gameplay HUD should stay period-authentic. |
| V110 | P2 | Grade rerun | Regrade after P0 and first 15 P1 items, using screenshots before code-only claims. | The visual grade should be evidence-based. |

## Suggested Fix Order

1. **P0 identity reset:** V001-V008, V022-V032. This is the fastest path from C+ toward B- visually.
2. **Theme pack 1:** Build one high-quality brick/green or red-panel theme before spreading effort across every episode.
3. **First cave reauthor:** Apply the new tiles to 6-8 screen-sized rooms with dense props, gems, signs, enemies, and a real exit room.
4. **HUD pass:** Make human mode use the simple black footer and move modern help/training text to pause/dev overlay.
5. **Hazard and enemy pass:** Add lava/spike/falling-rock/reverse-gravity visuals plus 4-5 large readable enemy silhouettes.
6. **Regression guard:** Add black-void budget, tile-density, prop-count, HUD, and theme-palette tests before another broad visual pass.

## Validation Run

- Rendered current gallery: `python scripts/render_crystal_caves_gallery.py --output-dir /tmp/nn_game1_crystal_caves_reference_compare`
- Inspected current gallery shots: `02_start.png`, `06_hazard_corridor.png`, `09_shooting_frame.png`, `12_episode3_moon.png`
- Inspected all seven user-provided reference images in this turn.
- Checked repo status before writing this audit so existing untracked Crystal Caves work was not confused with this report.

