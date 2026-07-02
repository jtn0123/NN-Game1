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

## RUN-25 verdict (both Phase-1 levers disconfirmed; instruments validated)

Full result in RUN_LOG.md (M4 commit `0de02eb`). 4 arms x 3 seeds x 6k episodes,
zero canonical wins and zero exit unlocks anywhere; the winner's-curse guard
never even triggered.

- **Enemy-motion observation: closed.** +0.013/+0.055 crystal over control,
  no death reduction (killed 0.500 vs 0.458), no completions. The "agent dies
  because it can't see movers" theory doesn't survive contact: it sees them
  now and dies the same.
- **Win-at-K=15 tier: closed in this form.** C had LOWER crystal than control
  and the highest killed rate (0.646); D collected best (0.508) but had the
  worst target progress. The tier never converts to eval unlocks because eval
  (correctly) keeps the real rule and the agent doesn't reach 15-crystal
  states reliably enough for the training tier to reshape behaviour.
- **Phase-0 instrument fixes CONFIRMED live:** trapped_frac = 0.000 and
  clock_mislabel_frac = 0.000 in every arm. The old trapped=0.354 is formally
  buried, and the red-team's stall-clock-mislabel hypothesis (#4) is
  disconfirmed — stalls are genuinely far-from-objective oscillation
  (far-stall 0.82-1.00), not measurement error.

**Strategic position after RUN-25:** observation-side, reward-side, task-tier,
train-longer, and level-repair families are ALL now closed with evidence. The
failure is stable across everything tried: ~50% killed by enemies mid-route,
~40-50% far-oscillation stalls, crystal plateau ~0.5 at 6k / 0.6-0.66 at 16k.
What has never been tried on this fixed set: **demonstration data** (DQfD-lite
margin loss / backward curriculum from demo states — Tier 2 of the improvement
plan, blocked only on demos existing). The demo recorder is built; the owner
playtest is the demo source (planner retry with an HP budget is the fallback).
RUN-26 should be the demo run.

## DATA-1 verdict: the harness timers own 35-54% of all endings, and learning does not fix them

M4 ran analyze_end_reasons over RUN-25 (older run dirs no longer on disk).
Per-milestone end-reason percentages, 48 evals per point, 6 milestones:

- Control (A): killed FLAT (0.46->0.46), stalled FLAT (0.50->0.50), timeout
  FLAT (0.04). Harness-timer share at final: **0.542** — and it never moved
  across 6k episodes of learning.
- B/C/D: stalled SHRINKS with learning (0.58-0.71 -> 0.29-0.46) but killed
  GROWS in lockstep (0.29-0.35 -> 0.50-0.65); wins stay 0.000 everywhere.
  Learning converts stall-executions into deaths, not completions.
- Note: train/test tables are identical by construction in imported mode
  (both splits cycle the same 16 levels deterministically).

Reading, combined with the validator's geometry data (best-case tours use
47-82% of the episode clock; Ore Shaft's longest leg uses 87% of the stall
window at perfect pace):
1. The stall executioner is a material, permanent tax (half the control's
   endings) that the levels' own geometry makes near-unavoidable.
2. BUT loosening it alone is unlikely to create wins — the B/C/D pattern
   shows episodes that escape the stall clock die to enemies instead. The
   binding failure is still "doesn't know how to finish", which is the demo
   path's job.
3. RUN-26 shape: demos are the primary lever; stall-window widening (720 ->
   ~1440) + truncation-aware bootstrapping ride along as fidelity fixes so
   timer endings stop training as deaths and mid-route journeys stop being
   executed. Requires making MAX_STEPS_WITHOUT_PROGRESS configurable (a
   class constant today).
