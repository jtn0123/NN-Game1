# Crystal Caves NN Performance Grade — 2026-07-07 (rev 2, incl. PR #37)

## Scope

A focused grade of the neural network **as it relates to performance** — how well
the learned policy actually plays, how fast it is improving, and what is capping
it. This updates the 2026-06-24 grade (`.Codex/nn-crystal-caves-grade-2026-06-24.md`).

**Rev 2 correction:** rev 1 graded `main` only. Open **PR #37**
(`claude/cc-reverse-curriculum`, 125 commits, active 2026-06-26 → 2026-07-02)
contains a large parallel workstream — 16 handcrafted authentic CC1 levels, a
corrected metrics pipeline, RUN-01…RUN-25, and the single biggest performance
lever found to date. Several rev-1 claims were stale because that evidence had
not been merged; this revision incorporates it. No new training was run for
this grade.

## The repo now has TWO diverged NN tracks

| | Track A (main / tracker) | Track B (PR #37 branch) |
|---|---|---|
| Benchmark | procedural tutorial-tier pool, **first-crystal** surrogate | 16 handcrafted authentic CC1 levels + procedural, **full-win** metric |
| Harness | `cc_status_session.py` recipes, B-series | `diagnose_gap.py` / `lever_ab.py`, RUN-NN series |
| Best result | B21: 23/60 first-crystal validation wins | RUN-13 compass: held-out tutorial win 0.033 → **0.483** |
| Last run | 2026-06-25 | 2026-07-02 |

Neither track's tracker/docs reference the other's findings. Track A's
experiment tracker does not know the compass breakthrough exists; Track B does
not use Track A's demo-BC/B3s machinery (it built its own DQfD-lite instead).
This divergence is itself a top performance finding: evidence is not flowing
to where decisions are made. (Rev 1 of this grade fell into the same trap.)

**Update 2026-07-08:** PR #37 is merged (`c03eddf`), so both tracks now share
one tree, and `--geo-compass` is wired into the Track A harness. The two
tracker documents (`CC_NN_EXPERIMENT_TRACKER.md` and
`experiments/cc_status/RUN_LOG.md`) still need folding into one
decision surface.

## Executive Grade

| Area | Grade | Basis |
|---|---:|---|
| Navigation/routing | **B-** | RUN-13 geo-compass: held-out tutorial wins 0.033 → 0.483 (14×), every seed; survives on easy/normal for collection |
| First-crystal surrogate (Track A) | **C+** | real, replicated gains (B3s 19/60 → B21 23/60 validation) |
| Full-level completion (the actual goal) | **F** | 0 wins on held-out normal across ALL of RUN-14…RUN-25; exit-unlock 0.000; crystal plateau ~0.5–0.66 |
| Learning dynamics / credit assignment | **C** | reward-clip now controllable and tested (RUN-23b); n-step truncation partially mitigated; NoisyNet+PER stack healthy (meanQ bounded) |
| Improvement velocity | **C** | Track B ran 25 rigorous A/Bs in ~6 days incl. one breakthrough — but most families closed with nulls, and the two tracks don't share evidence |
| Evidence/infra quality | **A-** | both tracks have strong promotion/audit discipline; the metric-bug find (IQM flooring) and instrument audits were excellent |
| **Overall NN performance** | **C** | up from C- on the strength of RUN-13, held down by zero full-game completion |

## What PR #37 already answered (rev-1 claims corrected)

1. **"No runs in 12 days" — wrong.** RUNs 19–25 ran through 2026-07-02 on the
   branch. The *merged* evidence base was idle; the work wasn't.
2. **Rev-1 red flag "Euclidean compass points through walls" — confirmed and
   FIXED on the branch.** RUN-11 isolated it (NEAR probe 0.73 vs FAR 0.12),
   RUN-13 fixed it as an *observation* (4 geodesic route scalars,
   `CRYSTAL_CAVES_GEO_COMPASS`): held-out tutorial win 0.033 → 0.483. This is
   the largest single lift in project history.
3. **Rev-1 red flag "reward clamp crushes terminals" — independently found
   (RUN-23) and A/B'd (RUN-23b).** The clamp WAS masking death-scale
   experiments (−12/−30 both trained as −5). But raising the clip to 35 did
   **not** unlock completion, and death-penalty −30 failed its decision rule.
   The clamp is real and now controllable (`--reward-clip`), but it is not the
   binding constraint.
4. **Rev-1 "run geodesic PBRS (reward)" — already run and DISCONFIRMED**
   (RUN-06/07/08: hurt learnability at every weight, even gated after-unlock).
   The winning form of the geodesic signal was observation, not reward.
5. **Rev-1 "run reverse curriculum" — family already run and DISCONFIRMED**
   on Track B (RUN-10/12: fixed-p tax lowered train competence, no transfer;
   even the corrected FAR variant lost to control).
6. Also closed with evidence on Track B: pool size/diversity, longer budget
   (16k episodes), fresh-level regeneration, difficulty warm-start curriculum,
   hazard-aware compass, enemy-motion observation, win-at-K training tier,
   NGU novelty bonus (movement up, wins not), CNN+GAP, level repair (physics
   audit proved zero real traps).

## Root causes that still stand (updated)

### 1. Full-chain completion is the wall, and every cheap family is now closed

On held-out normal: ~50% of episodes end killed (enemy-dominated), ~40–50%
stall far from the objective while oscillating, crystals plateau ~0.5 (6k ep)
to 0.66 (16k ep), exit essentially never unlocks. RUN-15a/15b proved levels
are structurally winnable (perfect tour uses 37% of budget) — the gap is
behavioral. Observation, reward, tier, budget, and level-repair families all
closed with nulls. Track B's own synthesis: the one high-evidence family never
tried is **demonstration data** (DQfD-lite margin loss / backward curriculum
from demo states). The recorder is built
(`python main.py --human --imported --record-demos`); RUN-26 is blocked on a
human playtest producing demos.

### 2. The stall clock is a permanent ~35–54% tax that learning does not fix

DATA-1: harness-timer endings (stall/timeout) stay flat across 6k episodes on
control; levers that reduce stalls convert them into enemy deaths, not wins.
Ore Shaft's longest leg uses 87% of the 720-step no-progress window at perfect
pace. RUN-26 should carry the fidelity fixes (stall window 720 → ~1440,
truncation-aware bootstrapping; `MAX_STEPS_WITHOUT_PROGRESS` needs to become
configurable) alongside the demo lever.

### 3. Two tracks, no evidence flow

Track A's promoted baselines (B3s/B21) never got the compass; Track B never
tried Track A's conservative demo-Q recipe (its closest analogue, DQfD-lite,
was built independently on the branch). Cheapest cross-pollination test: the
compass on Track A's B3s recipe surface. Biggest process fix: merge PR #37 (CI
green, mergeable) or explicitly declare Track B the canonical benchmark, and
fold both trackers into one document.

### 4. Selection-score weighting (unchanged from rev 1)

`selection_score` weights `exit_unlocked` equal to `win` (evaluator), so
keep-best can lock onto non-winning policies once unlocking starts happening.
Low urgency while unlock-rate is ~0, but fix before the demo era makes
unlocking common.

## What good looks like from here (revised)

1. ~~Decide PR #37~~ — **done 2026-07-08**: merged to main; PR #39 rebased on
   top; post-merge verification passed (1484 tests, compass audit clean).
2. **Record human demos** (owner playtest, ~an hour:
   `python main.py --human --imported --record-demos`) — the single input that
   unblocks RUN-26, the highest-evidence untried lever.
3. **RUN-26: DQfD-lite / backward-curriculum from demos**, with the stall
   window widening and truncation-aware bootstrapping riding along (the stall
   window is still a class constant, `MAX_STEPS_WITHOUT_PROGRESS = 720`; it
   must become configurable as RUN-26 prep).
4. **Cross-pollination A/B:** geo-compass on the Track A B3s recipe (now
   runnable: `run-recipe b3s_conservative_demo_q --geo-compass`) — if the
   14× tutorial lift shows up as first-crystal gains there too, Track A's
   whole promoted lineage gets a step-change for four extra state dims.
5. **Rebalance `selection_score`** so winning strictly dominates unlocking.
6. Stop spending on (now with branch evidence): reward scalar tuning of any
   kind, start-distribution curricula without demos, observation add-ons
   beyond the compass, longer same-family runs, contact-head threshold
   variants.

## Bottom line

Rev 1 said "the harness improves, the policy doesn't." With PR #37 in view,
the corrected statement is: **the policy DID take its first real step (RUN-13
compass, 14× on tutorial), the wall moved downstream to full-chain completion
under hazards/enemies, and the branch systematically closed every cheap family
attacking that wall.** The path forward is unusually clear and unusually
concrete: merge/canonicalize PR #37, record demos, run the demo lever. The
grade rises C- → C on real performance progress; it stays out of B territory
because the actual game — collect everything and leave — still has never been
completed on a held-out normal level.
