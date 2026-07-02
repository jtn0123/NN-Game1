# Decision brief: course of action after RUN-24 (2026-07-02)

Synthesis of three independent analyses: a fresh-eyes strategy review, an
adversarial red-team review (both read RUN_LOG.md, IMPROVEMENT_PLAN.md,
EVAL_PIPELINE_AUDIT.md, demo_extract.py, config.py; the red-team also ran
live code checks), plus the session's own running analysis.

## Where the analyses AGREE (high confidence)

1. **"Train longer" and "reward tuning" are closed.** RUN-24 (16k episodes,
   best config) plateaus at crystal_frac 0.60-0.66 with unsustained win blips.
2. **Human demo recorder is the cheapest unblocking move.** `--human` mode
   exists (src/app/interactive.py); the engine is deterministic; the open-loop
   verifier (`demo_extract.verify_stored`) already accepts any action source.
   A ~50-line recorder + one play session yields (a) ground-truth difficulty
   calibration and (b) winning demos that unblock backward-curriculum and
   DQfD-lite — the two highest-evidence techniques for fixed sets.
3. **Enemy-motion observation was built and then never used.** RUN-23/23b/24
   all ran with `CRYSTAL_CAVES_ENEMY_MOTION=False` while enemy deaths dominate
   (0.396 of deaths vs hazard 0.042). Must be an arm in the next run.
4. **Tiered win-at-K curriculum is principled.** Exit opens at K crystals
   during training (K=10→20→all), real rule for reporting. Same move class as
   RUN-13's breakthrough (change what the net sees, not the reward scalar).
   Greedy tour lengths (193-324 tiles ≈ 1500-2500 frames walking) show the
   3000-step budget is tight-but-feasible; the agent's effective horizon is
   720 (stall clock), so it never gets to practice the endgame.
5. **Deprioritize capacity swings** (CNN, bigger nets, distributional) as the
   *next* move — legit later A/B, but the failure signature isn't perception
   of static content or capacity.

## Where the red-team OVERTURNS prior conclusions

6. **The planner's 0/16 does NOT prove the levels are unfair.**
   `demo_extract.py` prunes ANY planned enemy contact as death and replans on
   any HP loss — it is a 0-damage speedrun planner. The game gives 3 HP + 70
   invuln frames: HP is a spendable resource the planner refuses to spend.
   Live runs confirmed failures are early planning collapse (L09: plan_failed
   at step 400 with 29/30 crystals left), not the 3000-step budget. The
   fresh-eyes diagnosis "task/level-design problem" is therefore **unproven**.
   Human playtest resolves this properly (and cheaper than more planner work).
7. **The compass instrument may be manufacturing the failure taxonomy.**
   - `_geodesic_distance_field` is 4-connected with symmetric vertical edges:
     after a one-way drop it happily points back UP an unclimbable wall
     forever. That mechanism alone can produce both the trapped=0.354 stalls
     and RUN-19's "far, oscillating" signature.
   - The 720-step stall clock resets on *euclidean* new-best to the
     *euclidean-nearest* target, while the compass descends a *geodesic*
     field — an agent faithfully following its compass around a wall can be
     killed as "stalled" and charged −6 while making real progress.
   - Fixing the *instrument* (directed, jump-aware distance field — the
     jump-aware oracle already exists in `cave_reachable`) preserves the
     benchmark; repairing *levels* mid-experiment voids RUN-24 comparisons.
8. **Promotion hygiene failed despite the audit.** D′ became "best-so-far" on
   within-noise soft metrics while having 0 wins and MORE enemy deaths than
   A′; RUN-24's 3/48 @ep11k is an argmax over 16 checkpoints (max order
   statistic of p≈0.02 noise) and no checkpoints were saved. New rule: a
   checkpoint is only "winning" if the SAME checkpoint repeats on a fresh
   seed set; decision runs always save checkpoints.
9. **Unmeasured compound mode:** knockback (vy=−5.5, vx=−2·facing) near a
   ledge can convert a −3 hit into a one-way-drop trap, logged as "stalled".

## Recommended course of action

**Phase 0 — this week, zero training cost:**
- **P0a. Stall autopsy (agent-free, ~half day):** over recorded stall
  episodes measure (i) fraction where the compass-suggested route from the
  stuck tile is physically infeasible under jump-limited directed
  reachability, (ii) fraction where geodesic distance was still shrinking
  when the 720 clock fired, (iii) hit→trap conversions (knockback crossing a
  one-way edge within 100 steps before a trapped-stall). This decides
  instrument-flaw vs level-flaw BEFORE any level repair.
- **P0b. Human demo recorder + playtest (~1 day):** record action ids per
  frame in `--human`, dump in `demo_extract` JSON format, gate with
  `verify_stored`. Output: difficulty ground truth + demo dataset.

**Phase 1 — next M4 run (shaped by Phase 0):**
- Jump-aware directed compass field (if P0a confirms) — instrument repair.
- `CRYSTAL_CAVES_ENEMY_MOTION=True` A/B arm.
- Win-at-K=15 tier arm.
- Checkpoints saved; wins must replicate on fresh seeds before promotion.

**Phase 2 — demo-powered (once P0b lands):**
- Backward curriculum from human-demo states and/or DQfD-lite margin loss.
- Level repair (return ladders) ONLY for levels the human playtest + no-trap
  audit both flag — with re-verified oracle suite and a version bump so runs
  aren't silently compared across level versions.

**Explicitly deprioritized:** more planner engineering (optionally one cheap
retry with an HP-budget: contact = high edge cost, tolerate 2 hits before
replanning), CNN/capacity swings, immediate level repair.

## Phase 0 results (same day — the autopsy landed immediately)

**P0a executed as a static physics audit** (`compass_audit.py`): a
physics-faithful motion graph (the oracle's macro simulator) over every
resting cell of every level, doors open, cross-checked against both the
compass field and the tile oracle. Three verdicts:

1. **The levels contain ZERO real traps.** From every one of the 4,368
   physics-reachable resting cells across all 16 levels, every crystal, every
   switch and the exit remain reachable. There are no one-way-drop dead ends.
   **Level repair is cancelled** — nothing to repair.
2. **The trapped=0.354 stat was an instrument artifact.** The live trapped
   detector (`game._oracle_reachable` → `cave_reachable`) had NO LADDER
   support — 'H' was plain air, nothing above a ladder shaft was reachable.
   Cross-check: the oracle called 35.3% of (cell, crystal) pairs unreachable
   (45–76% on the nine ladder-heavy levels, 0% on the seven others); physics
   truth is 0%. FIXED: `cave_reachable` now climbs ladders (shaft traversal +
   grounded grip, mirroring elevators); regression tests added. The same fix
   un-restricts the FAR reverse-exit curriculum placement pool, which used the
   same blind oracle.
3. **The compass tells no hard lies on this set** (statically, doors open):
   its nearest-labeled objective is physically reachable from every resting
   cell — red-team hypothesis #5's worst case (infinite pursuit of an
   unreachable objective) cannot occur on these levels. Distance distortion
   (4-connected symmetric vs real route length) remains possible and is left
   to RUN-25 telemetry.

**P0b delivered:** `--record-demos` human demo recorder
(`src/app/demo_recorder.py`, wired into `--human` mode + `--imported` CLI
flag), episode JSONs replay-verified via `demo_extract.verify_stored`,
summary/verify CLI in `experiments/cc_status/human_demos.py`, 4 tests.
Owner playtest: `python main.py --human --imported --record-demos`.

**Consequence for RUN-25:** stall failures were ~50% of episodes and a third
of those carried a false "trapped" label — the true residual stall mass is
far/oscillating + clock semantics. RUN-25 arms stay as planned (enemy-motion
ON, win-at-K tier, checkpoint hygiene) plus stall-clock geodesic telemetry;
level repair is off the table.
