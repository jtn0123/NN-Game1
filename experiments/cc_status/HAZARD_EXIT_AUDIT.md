# Hazard-walkability + exit audit (pre-playtest, all 16 levels)

Method: physics-faithful macro BFS from spawn (doors open) where any spike/acid
touch CUTS the move — measuring what is collectible with ZERO hazard damage.
Conservative: the 12-macro movement vocabulary is coarser than pixel-perfect
human play, so "forced damage" means "no clean route found at this granularity",
not proof one doesn't exist. Hazards cost 1 HP + knockback + 70 invuln frames
(same as enemy hits) — never instant death from full health.

| level | spikes | acid | hazard-free objectives | exit (col,row) |
|---|---:|---:|---|---|
| Ore Shaft | 4 | 0 | 17/33 — spike choke gates a whole wing | top-middle (21,2) |
| Dripstone Hollow | 2 | 3 | ALL clean | top-left (3,3) |
| The Freight Lift | 2 | 0 | ALL clean | top-middle (20,3) |
| The Lockworks | 1 | 0 | ALL clean | top-middle (15,2) |
| Stalactite Chasm | 2 | 2 | ALL clean | top-middle (20,4) |
| Elevator Exchange | 1 | 0 | ALL clean | top-middle (20,3) |
| The Smelter | 3 | 2 | 26/33 — right wing behind a hazard choke | top-middle (14,3) |
| The Acid Vents | 2 | 3 | 30/31 — one crystal (16,21) | top-left (3,2) |
| Twin Vaults | 1 | 0 | 22/33 — left vault behind a choke | top-middle (20,2) |
| Cavern Descent | 0 | 2 | ALL clean | top-right (32,5) |
| Cavern of Echoes | 3 | 2 | 23/32 — right/upper-right wing | top-right (36,2) |
| Cascade Keep | 1 | 0 | ALL clean | top-right (37,2) |
| Scaffold Reactor | 0 | 2 | ALL clean | top-left (12,3) |
| The Sunken Grotto | 3 | 2 | ALL clean | top-left (3,2) |
| The Switchback Spire | 1 | 0 | ALL clean | top-middle (20,3) |
| The Mother Lode | 4 | 2 | 11/35 — spike chokes gate most of the map | top-left (3,2) |

Findings:
- 10/16 levels are fully collectible with zero hazard damage.
- 6 levels (Ore Shaft, Smelter, Acid Vents, Twin Vaults, Echoes, Mother Lode)
  have at least one hazard CHOKE the coarse mover can't pass cleanly — costing
  ~1 HP per crossing at worst. With 3 HP + invuln this is classic
  spend-a-heart design, not unwinnability; the winnability oracle already
  proves touch-reachability of every objective.
- These 6 overlap heavily with where the demo planner died or gave up
  (Twin Vaults failed at 61 steps; Mother Lode plan_failed at hp=1) —
  corroborating hazard chokes + camped corridors as the levels' real teeth.
- ALL 16 exits are in the TOP band (rows 2-5): the game loop is descend,
  collect, then CLIMB HOME. The endgame the agent never reaches is always an
  ascent — exactly what demo-prefix (mid-route) training starts drill.
- Exit perception: dedicated tile code in the window (locked=0.38 vs open),
  a global exit-unlocked state flag, and the compass targets it in the final
  phase — fully visible to the agent in both states.
