# Improvement Plan: from 0% wins to completing the hand-crafted set

Context: RUN-20 proved the 16 hand-crafted levels are learnable but unsolved —
crystal_frac climbed 0.28→0.52, zero official wins, ends dominated by killed
(~60%) then stalled (~35%). RUN-21 (hazard-aware compass) is trending negligible.
This plan synthesizes two literature reviews (safe-RL/hazard-avoidance and
sparse-reward completion) with the repo's own run history.

## The regime changed — old disconfirmations need re-reading

RUN-01..18 fought a GENERALIZATION problem (hold-out procedural levels), and many
levers were tested on levels later proven unfair/unwinnable. The new task is 16
FIXED, verified-winnable levels where memorization is allowed. Techniques that
excel at fixed deterministic sets (demonstrations, backward resets, per-level
prioritization) were never really on the table before.

## Key asset we already own

`experiments/cc_status/level_reach.py` proves every level winnable by SIMULATING
a winning exploration with the engine's exact physics. Extending it to record
the action sequence gives us a WINNING DEMONSTRATION for every level — which
unlocks the two highest-evidence techniques in the completion literature
(DQfD demo-seeding, Salimans & Chen backward curriculum) essentially for free.

## Ranked plan

### Tier 1 — cheap, high-evidence, do first

**A. ~~Safe-action shield~~ — VETOED by owner** (not true to the real game:
the player gets no move-blocking protection, so neither should the agent).
Recorded for history; do not build. The death rate must be earned via fair
perception (B) and learning, not enforcement.

**B. Enemy motion in the observation (BUILT — RUN-22, `CRYSTAL_CAVES_ENEMY_MOTION`;
FOV-limited per owner: only enemies inside the perception window)** — a single-frame tile
window provably cannot dodge movers (two mirror states look identical). Add:
per-enemy relative (dx, dy, vx, vy) for the 3 nearest enemies, and/or an
"enemy-next-cell" map channel rolled 1-2 ticks forward (exact — dynamics are
deterministic). Same computation as the shield, exposed to the net. Evidence:
frame-stack ablations (velocity is one of DQN's biggest known effects);
Lample & Chaplot's game-features result.

### Tier 2 — the completion breakthrough path (needs demo extraction first)

**C. Demo extraction (prereq, ~a day)** — extend the oracle BFS with parent
tracking to reconstruct a full action-level winning trajectory per level; replay
each in the live engine and assert it WINS (also the strongest possible
winnability proof). Store trajectories as (state, action, reward, done) in the
agent's observation format.

**D. Backward curriculum from demo states (RUN-23 candidate)** — reset training
episodes to states along the winning trajectory, starting near the exit, moving
the pointer earlier as the agent masters each suffix (>50-70% success). Reuses
the existing reverse-curriculum machinery (`CRYSTAL_CAVES_REVERSE_CURRICULUM*`),
but anchored to demo states WITH the correct crystal/lever inventory. Evidence:
Salimans & Chen beat all Montezuma results from ONE demo this way; it is the
cheap half of Go-Explore. NOTE: RUN-10/12 reverse curricula were disconfirmed —
but on broken procedural levels, without demos, chasing generalization; the
demo-anchored fixed-set version is a different animal. The failure signature to
watch: train-competence dropping while probe flat (RUN-12's signature).

**E. DQfD-lite demo-seeding (RUN-24 candidate)** — separate never-overwritten
demo buffer; add large-margin supervised loss on demo samples + n-step (have) +
PER priority bonus (have); demo ratio SMALL (R2D3: 1/256-1/64 at scale; start
~1/16 here) and annealed. Pre-train briefly on demos. Evidence: DQfD/R2D3 solved
zero-reward-from-scratch tasks; ablations show the margin loss is what makes
demos work — naive buffer-dumping fails.

### Tier 3 — cheap multipliers, fold into any run

**F. PLR-lite level sampling** — sample next training level ∝ (1 - win rate) +
TD-error with ~20% uniform mixing (16 levels → ~30 lines). Evidence: PLR.
**G. Route-ordered compass** — point the compass at the next objective in the
planner's route order (not euclidean-nearest), add phase one-hot + remaining
count. Derive demo order and compass order from the SAME planner route so D/E
don't fight it.

### Explicitly NOT doing (evidence-backed)

- **PPO switch** — lateral move at this scale; replay + demos are off-policy
  advantages we'd lose.
- **Bigger death penalties / hazard-proximity penalties** — with an agent that
  already stalls, they push toward freezing (ROSARL bound; our own RUN-19 stall
  diagnosis). Keep death ≈ -(completion reward), handle danger via observation +
  shield.
- **Full HRL / Go-Explore archive** — overkill; planner already solves
  exploration, backward resets ARE the robustification phase.
- **Re-running geodesic PBRS shaping** — disconfirmed three ways (RUN-06/07/08)
  with a clean mechanism story; the compass observation replaced it.

## Suggested run order

1. **RUN-22:** shield + enemy-velocity features (Tier 1 A+B together — both are
   observation/selection-side, independent of reward). Gate: killed rate drops
   materially AND crystal_frac holds; watch stalls.
2. **Demo extraction** lands meanwhile (C).
3. **RUN-23:** backward curriculum from demos (+ PLR-lite F).
4. **RUN-24:** DQfD-lite margin loss if RUN-23 leaves wins on the table.
5. Re-evaluate; curriculum-of-levels remains a fallback (owner option 1).

Decision gates at each step; every level change re-verified by the oracle suite
(winnable / gated / door-value) before training.


## Post-RUN-24 addendum (long-horizon verdict + new lead)

RUN-24 (16k episodes, best-so-far config) is decision-grade: longer training
lifts crystal collection to a ~0.60-0.66 plateau and produces OCCASIONAL
canonical wins (first ever: 2-3/48 around ep9-11k) but never sustains them —
final checkpoint 0/48. "Train longer" is closed as an excuse for this config
class.

NEW LEAD from the final stall trace: **trapped = 0.354** — a third of stalls
end with the remaining objective PHYSICALLY UNREACHABLE from where the agent
stands, typically holding ~70% of the crystals. The winnability oracle proves
all objectives reachable FROM SPAWN, but several levels contain deliberate
one-way drops — after taking one, parts of the level can become unreachable.
Two candidate responses, complementary to the demo path:
1. Extend the oracle with a NO-TRAP audit (from every standable cell, all
   remaining objectives + exit stay reachable) and repair the levels that fail
   it (add return ladders). Design-side, verifiable, honest.
2. Teach the route order that avoids trap commitment (exactly what planner
   demos encode).
