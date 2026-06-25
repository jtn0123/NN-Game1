# Crystal Caves NN Metrics and Progress Review

**Date:** 2026-06-23
**Purpose:** answer whether the current metrics are detailed enough, whether small
improvements may be getting missed, and whether the project is actually progressing.

## Short Verdict

The concern is valid. Our metrics are now good enough to avoid obvious false positives,
but they are not detailed enough to confidently detect small useful improvements. A
change that moves the agent from "wanders nowhere" to "gets within two tiles of the
crystal and then loops" can still look like a failure in the top-line metrics because
both runs show `0` full wins and often `0` trace crystals.

We are progressing, but slowly and not yet on the final outcome. The strongest progress
so far is diagnostic and first-objective progress:

- The runner now saves comparable artifacts and live checkpoint data instead of relying
  on one-off impressions.
- We found that direction-label supervision is learnable but not enough.
- Action-level tutorial demos are a better lever than direction labels.
- The best route-gate candidates improved selected first-crystal eval from B3d's `6/30`
  to B3g/B3h's `7/30`, with better depth and target-distance approach.

That is real, but it is small. It is not enough to claim the NN is close to full Crystal
Caves completion. We still have no reliable full-level completion signal and the latest
selected trace diagnostics still show `0/4` games collecting a crystal.

## Current Evidence Snapshot

| Run | Main idea | Selected expanded first-crystal eval | Expanded depth | Trace depth | Trace crystal rate | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| B3d | direct tutorial route | `6/30` | `28.6%` | `26.8%` | `0%` | baseline route gate |
| B3f | route direction auxiliary head | `4/30` | `21.7%` | `10.7%` | `0%` | learned label, worse behavior |
| B3g | tutorial demo behavior cloning | `7/30` | `36.2%` | `48.2%` | `0%` | weak positive/current best route gate |
| B3h | online demo margin loss | `7/30` | `35.7%` | `53.6%` | `0%` | mechanically works, no better than B3g |

The subtle part is in the trace diagnostics:

| Run | Mean best target-distance improvement | Mean final target-distance improvement | Mean unique tiles | Mean max tile-visit share | Main trace failures |
|---|---:|---:|---:|---:|---|
| B3g | `7.77` tiles | `5.76` tiles | `23.75` | `66.1%` | no crystal, stall, tile loop, some idle/interact |
| B3h | `8.84` tiles | `6.81` tiles | `23.25` | `61.5%` | no crystal, stall, tile loop |

B3h may have slightly improved "approach the target and avoid interact spam", but the
top-line selected eval ties B3g at `7/30`. Our current reporting makes that look like
"nothing changed." In reality it might be a small behavioral improvement, but it is not
actionable yet because we do not know whether the agent got close enough to matter.

## Why It Feels Like Nothing Helps

This is a sparse, chained-control problem. The NN has to learn a sequence:

1. move to first crystal,
2. collect it,
3. move to the next objective,
4. unlock the exit,
5. reach the exit without stalling.

If any link is unreliable, the final win metric stays near zero. That makes progress feel
flat even when a lower-level behavior improves. A `30`-game eval also has coarse
resolution: one extra success is only `+3.3` percentage points and may just be noise.

The current pattern is not "every idea is useless." The better interpretation is:

- reward-only changes do not matter because the agent rarely reaches the terminal event;
- representation-only changes can be learned but do not force useful control;
- action guidance helps a little, but the current demo source is weak;
- the agent still collapses into stalls/loops before reliable first-crystal collection.

## Metrics We Are Missing

These are the metrics most likely to expose small but real improvements:

1. **Per-level eval matrix**
   Record the same fixed held-out levels every checkpoint with one row per level. This
   tells us whether a change solves new levels, loses old ones, or just moves noise.

2. **Near-miss distance bands**
   Track whether the agent ever gets within `10`, `5`, `3`, and `1.5` tiles of the
   current objective. A run with `0` crystals but many `<=3` tile approaches is very
   different from a run that never gets near the target.

3. **Best-approach timing**
   Record the step where the agent gets closest to the objective, then what happens
   afterward. This separates "cannot navigate" from "gets close and then loops."

4. **Close-zone action distribution**
   When near the objective, record action shares: idle, left/right, jump, interact,
   shoot, and repeated-action streaks. This tells us whether the missing behavior is
   jumping, final alignment, platform approach, or action spam.

5. **Objective geometry**
   For failures, record relative target geometry at the closest point: same platform,
   target above, target below, horizontal gap, vertical gap, and line-of-sight if cheap.

6. **Per-checkpoint confidence flags**
   Mark tiny deltas as weak evidence. For `30` games, `6/30` versus `7/30` is not a
   promotion by itself. A candidate should either win by several games or tie wins while
   improving multiple near-miss metrics.

## Recommended Measurement Upgrade

Before the next NN/model tweak, add a "first-objective near-miss eval" to
`experiments/cc_status_session.py`.

Minimum useful output:

- `per_level_eval.jsonl`: one row per eval level/checkpoint.
- `near_miss_rate_10`, `near_miss_rate_5`, `near_miss_rate_3`, `near_miss_rate_1_5`.
- `mean_min_target_distance_tiles`.
- `mean_target_distance_best_delta_tiles`.
- `mean_target_distance_final_delta_tiles`.
- `mean_step_of_best_approach`.
- `stuck_after_close_rate`.
- `close_zone_action_counts`.
- `close_zone_jump_rate`.
- `close_zone_idle_or_interact_rate`.
- `loop_after_close_rate`.

Implementation status: **done** in `experiments/cc_status_session.py`.

The runner now writes a `near_miss_eval` artifact for final candidate summaries and a
`selected_checkpoint_near_miss_eval` artifact when selected-checkpoint eval is enabled.
Each artifact writes:

- `near_miss_eval/<label>/summary.json`
- `near_miss_eval/<label>/per_level_eval.jsonl`

The markdown report now prints the most important near-miss signals directly: `<=3`
tile rate, `<=1.5` tile rate, mean minimum target distance, best/final target-distance
delta, step of best approach, close-zone jump rate, close-zone idle/interact rate,
stuck-after-close rate, and loop-after-close rate.

2026-06-24 follow-up: correction fine-tune sessions now also expose
`avg_correction_action_loss_100`, `avg_correction_action_accuracy_100`, and
`correction_action_samples_100` in live metrics, summary JSON, and `report.md`. This
gives an earlier check that a DAgger-style correction run is actually sampling the
policy-visited correction dataset before we wait for noisy held-out wins. The sample
count matters because a hinge loss can legitimately reach `0.0`; zero loss is only
trustworthy when sample count is positive.

Verified with tests and a one-episode `first-crystal-direct` smoke run:
`.Codex/artifacts/cc_sessions/20260623_205802_near_miss_smoke/`.

Promotion rule for future candidates:

- Strong promote: selected first-crystal eval improves by at least `+3/30` over the saved
  baseline.
- Conditional promote: selected eval ties or improves by `+1/30`, and at least two
  near-miss metrics improve materially: `<=3` tile rate, best-distance delta, loop rate,
  or close-zone jump/alignment behavior.
- Do not promote: training loss, Q-values, demo-action accuracy, or source eval improve
  while held-out near-miss metrics do not.

## Are We Progressing?

Yes, but the type of progress matters:

- **Instrumentation progress:** strong. We can now compare runs, inspect checkpoint
  behavior, and avoid being fooled by training-only gains.
- **Skill diagnosis progress:** strong. We know the agent can learn isolated skills and
  that first-objective routing is still the practical wall.
- **First-objective held-out progress:** weak but real. B3g/B3h are better than the
  direct route baseline on selected expanded eval and/or approach diagnostics, but the
  gain is too small to trust as a solved direction.
- **Full game completion progress:** not enough yet. The agent still does not reliably
  complete full levels.

The next smart step is not another reward tweak. It is to make the eval tell us whether
the agent is failing far away from the crystal, near the crystal, at the jump/alignment
moment, or after collecting. Once we can see that, the next intervention can be much
more targeted.

## Next Action

Implement the near-miss/per-level eval upgrade, then rerun the current best candidate
style on the same saved baseline comparison surface. After that, choose the next model
change based on the newly exposed failure mode:

- far from target: improve route planner/demo source or objective-map supervision;
- close but no collect: add close-zone action guidance/jump alignment curriculum;
- gets close then loops: add anti-loop/novelty only in close-zone states;
- collects first crystal but fails afterward: move the same metric ladder to next
  objective and exit unlock.
