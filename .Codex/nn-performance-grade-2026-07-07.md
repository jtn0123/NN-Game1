# Crystal Caves NN Performance Grade — 2026-07-07

## Scope

A focused grade of the neural network **as it relates to performance** — how well
the learned policy actually plays, how fast it is improving, and what is capping
it. This updates the 2026-06-24 grade (`.Codex/nn-crystal-caves-grade-2026-06-24.md`)
with everything the B-series produced (B3l through B30) plus a fresh code audit
of the training stack. No new training was run for this grade; run-evidence gaps
are enumerated in `CC_NN_RUN_INFO_TASKS.md` for a follow-up execution agent.

## Executive Grade

| Area | Grade | Trend since 06-24 |
|---|---:|---|
| First-crystal routing (the tracked surrogate) | **C+** | ↑ real, replicated gains (B3s 19/60 → B21 23/60 validation) |
| Full-level completion (the actual goal) | **F** | → still ~0; not even measured in the recent eval ladder |
| Learning dynamics / credit assignment | **C-** | → two confirmed structural handicaps still in place (see findings 1–2) |
| Improvement velocity | **D+** | ↓ ~30 experiments in 3 days bought +4/60 on a surrogate; then 12 days with zero runs |
| Evidence/infra quality | **A-** | ↑ promotion gates, metric audit, paired A/B, near-miss ladder |
| **Overall NN performance** | **C-** | flat |

The C- headline is unchanged from two weeks ago, and that is itself the finding:
the experiment harness improved substantially, the policy barely did.

## Where performance actually stands

- **Pure learned policy (B3s):** 10/30 selected first-crystal wins (seed 0),
  19/60 expanded validation, 8/30 on seed 1. Route depth ~60%.
- **Best learned adapter (B21):** frozen B3s trunk + offline contact head,
  13/30 selected / 23/60 validation (seed 0), 10/30 / 18/60 (seed 1).
- **Best overall outcome (B10):** B3s + eval-time advantage-gated oracle option,
  30/60 — but ~half its margin comes from a hand-coded simulator gate, not the NN.
- **Full game (collect all crystals → unlock → exit):** every recent number above
  is a *first-crystal* metric. Full-level wins were ~0 at last measurement and no
  recent artifact even evaluates them.
- **Trajectory of the B-series:** of ~30 labelled experiments (B3l–B30), two were
  promoted (B15, B21) and one controller (B10). B22–B30 were all 9–10/30 —
  threshold/label tuning inside the noise band. The tracker's own conclusion is
  correct: that lane is exhausted.

## Root causes of the stagnation (ranked, with code evidence)

### 1. The learn-time reward clamp crushes exactly the signals the game design relies on — **confirmed**

`src/ai/agent.py:666-667` clamps every learned reward to `min=-REWARD_CLIP`
(`-5.0`, `config.py:529`). But the environment's designed terminal penalties are
death **−12**, timeout **−8**, stall **−6** — after the clamp all three become an
identical **−5**, and since the clamp is applied to the *n-step accumulated*
return, negative PBRS shaping sums get truncated too, which quietly breaks the
telescoping/policy-invariance argument the PBRS work depends on. The agent
cannot distinguish "died" from "wandered until timeout"; the two failure modes
that dominate end-reason counts (stall/timeout) are rewarded identically to the
one that should be feared most. This nullifies a whole class of reward-tuning
experiments — several "reward changes did nothing" conclusions in the tracker
may be artifacts of the clamp.

### 2. The n-step pipeline delivers ~3.5-step credit, not the configured 6 — **confirmed**

`src/ai/prioritized_n_step.py:86-115` (`_flush_buffer`): the per-env buffer is
flushed whenever it reaches `n_steps` *without* a terminal, and the flush emits
returns for **every** index then clears — so in a full 6-flush only index 0 gets
a true 6-step return; indices 1–5 get 5,4,3,2,1-step spans. Since the 2026-06-25
fix the *discounts* match the actual spans (targets are no longer wrong), but
the average bootstrap horizon is ~3.5, roughly half of what `N_STEP_SIZE=6` was
raised to specifically for the long collect→…→exit chain. A sliding-window
emit (flush only the oldest transition once it is n steps old) would deliver the
intended horizon.

### 3. The two levers built for the actual failure mode have never been run — **confirmed**

The documented wall is chain completion, and two remedies were merged on
2026-06-26: the geodesic PBRS potential (PR #35) and reverse-curriculum
mid-solution starts (PR #36). Both are `False` by default in `config.py`, no
runner or recipe references them, and `set_reverse_curriculum_p()` has zero
callers. The newest artifact on disk predates both merges. Twelve days of
potential A/B evidence on the highest-leverage ideas simply doesn't exist.
*(This PR adds the missing `--geodesic-potential`, `--reverse-curriculum-p`,
`--show-locked-exit`, and `--reward-clip` plumbing to the status-session
runner.)*

### 4. The state's only global heading signal points through walls

The target compass features (`target_dx/dy/distance`,
`src/game/crystal_caves_logic.py:390-441`) are straight-line Euclidean. On any
level where the route is not line-of-sight the compass actively misleads —
consistent with the dominant observed failure ("gets within a few tiles, then
loops/stalls"). The BFS geodesic field already exists for the potential; feeding
route distance (or a next-waypoint direction) into the state is the natural
follow-up if the Phase 1 geodesic A/B shows signal.

### 5. Checkpoint selection scores "unlocked the exit" as highly as "won"

`src/ai/evaluator.py:288-294`: `selection_score` weights `win_rate` and
`mean_exit_unlocked_rate` at 1.0 each. A policy that reliably unlocks but never
enters the exit is nearly indistinguishable from a winner at keep-best time, and
this score also feeds the plateau/early-stop and curriculum gating. As soon as
policies start unlocking exits at all, this will actively select against
finishing.

### 6. Strategy: the eval ladder optimizes a surrogate the wins don't live on

Every promotion decision since B3g is on first-crystal terminal evals. That was
the right scaffold to get routing signal, but the last five promoted/held
candidates moved the surrogate without any measurement of full-level behavior.
There is no standing full-level eval row in any recent artifact, so nobody can
say whether B21 is closer to a real win than B3d was. The surrogate has quietly
become the goal.

## What good looks like from here

1. **Run the two dormant levers** (geodesic PBRS, reverse curriculum) as clean
   A/Bs against frozen B3s — Phase 1 of `CC_NN_RUN_INFO_TASKS.md`.
2. **A/B the reward clamp** (`--reward-clip 0` vs default) — if the clamp is
   hurting, every prior negative-reward experiment deserves an asterisk.
3. **Fix the n-step window** (code change, small, testable) and re-baseline.
4. **Stand up a full-level eval row** next to the first-crystal row in every
   artifact so surrogate drift is visible.
5. **Rebalance `selection_score`** so winning strictly dominates unlocking.
6. Stop spending on: contact-head threshold variants, label aggregation
   re-mixes, terminal-reward tuning (already dead per tracker, and doubly dead
   while the clamp is in place).

## Bottom line

The infrastructure grade keeps going up; the policy grade does not move. The
harness can now detect a +1/60 improvement with confidence — what it has not had
is a candidate worth detecting, and two structural handicaps (clamped terminal
signal, halved credit horizon) plus two never-run remedies are the most likely
reasons why. Close the evidence gap in `CC_NN_RUN_INFO_TASKS.md` before any new
method work.
