# Crystal Caves NN Tracker Archive: Findings 2026-06-24 Part 1

Archived from `CC_NN_EXPERIMENT_TRACKER.md` during cleanup on 2026-06-24.

### 2026-06-24 B3m. Two-Stage Demo Route + Light Bridge Fine-Tune

**Status:** completed; not promoted.

**Why this is next:** B3l showed that one bridge lane can preserve some close-zone
action benefits, but bridge interleave from scratch still trails B3g because route
approach weakens. B3m tests the more specific hypothesis: bridge skill practice may
only help *after* the B3g demo-BC policy already knows how to approach the objective.

**Implementation:** added `tutorial-demo-bridge-finetune`. Stage 1 runs B3g-style
tutorial demo BC and selects the best route checkpoint by source eval. Stage 2 starts
from those selected route weights, runs a short bridge interleave fine-tune with
first-crystal tutorial lanes plus full bridge lanes, and selects again by held-out
source eval. The selected eval can choose the initial transfer policy if bridge
fine-tuning hurts.

**Validation before run:** focused status-session tests passed (`36 passed in 2.89s`),
touched-file ruff passed, and the full suite passed (`1020 passed in 22.63s`).

**Artifact:** `.Codex/artifacts/cc_sessions/20260624_060412_tutorial_demo_bridge_ft100_b125_pool512_select30`

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-bridge-finetune \
  --episodes 300 \
  --bridge-finetune-episodes 100 \
  --seed 0 \
  --eval-games 16 \
  --selected-eval-games 30 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --cave-pool-size 512 \
  --route-demo-levels 128 \
  --route-demo-max-steps 800 \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --interleave-bridge-ratio 0.125 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label tutorial_demo_bridge_ft100_b125_pool512_select30
```

**Promotion rule:** compare against B3g (`7/30`) and B3l (`6/30`). Promote if the
selected fine-tune policy beats B3g or ties B3g while improving close-zone jump /
stuck-after-close without losing depth. Reject if selected policy falls back to initial
transfer or if fine-tuning lowers selected wins/depth.

**Route-stage source evals:**

| Episode | Source wins | Crystal rate | Depth | Selection score | Mean score | Ends |
|---:|---:|---:|---:|---:|---:|---|
| after BC | `1/16` | `6.3%` | `32.6%` | `0.076` | `10.9` | `{'first_crystal_goal': 1, 'stalled': 15}` |
| 50 | `0/16` | `0.0%` | `13.4%` | `0.000` | `0.0` | `{'stalled': 5, 'timeout': 11}` |
| 100 | `2/16` | `12.5%` | `25.4%` | `0.154` | `35.9` | `{'first_crystal_goal': 2, 'stalled': 9, 'timeout': 5}` |
| 150 | `0/16` | `0.0%` | `32.6%` | `0.002` | `18.8` | `{'stalled': 11, 'timeout': 5}` |
| 200 | `1/16` | `6.3%` | `17.9%` | `0.076` | `6.3` | `{'first_crystal_goal': 1, 'stalled': 10, 'timeout': 5}` |
| 250 | `3/16` | `18.8%` | `40.2%` | `0.229` | `42.2` | `{'first_crystal_goal': 3, 'stalled': 8, 'timeout': 5}` |
| 300 | `3/16` | `18.8%` | `39.3%` | `0.229` | `37.5` | `{'first_crystal_goal': 3, 'stalled': 8, 'timeout': 5}` |

Route stage selected ep250 for transfer. This route seed was weaker than the prior B3g
artifact selected policy (`7/30` expanded), which is a confound when judging the
fine-tune.

**Bridge fine-tune source evals:**

| Episode | Source wins | Crystal rate | Depth | Selection score | Mean score | Ends |
|---:|---:|---:|---:|---:|---:|---|
| transfer | `3/16` | `18.8%` | `40.2%` | `0.229` | `42.2` | `{'first_crystal_goal': 3, 'stalled': 8, 'timeout': 5}` |
| 50 | `5/16` | `31.3%` | `15.2%` | `0.378` | `31.3` | `{'first_crystal_goal': 5, 'stalled': 5, 'timeout': 6}` |
| 100 | `4/16` | `25.0%` | `19.2%` | `0.303` | `25.0` | `{'first_crystal_goal': 4, 'stalled': 6, 'timeout': 6}` |

Fine-tune selected ep50 for expanded eval. Bridge practice increased first-crystal
hits on the small source eval, but it also made the policy much shallower.

**Selected expanded comparison:**

| Metric | B3g tutorial demo BC | B3l bridge 12.5% | B3m route + bridge fine-tune | Read |
|---|---:|---:|---:|---|
| Selected expanded eval | `7/30` | `6/30` | `7/30` | B3m tied B3g on hits |
| Selected crystal rate | `23.3%` | `20.0%` | `23.3%` | tied B3g |
| Selected depth | `36.2%` | `26.0%` | `16.4%` | severe depth loss |
| Near-miss <=3 tiles | `40.0%` | `33.3%` | `33.3%` | worse than B3g |
| Near-miss <=1.5 tiles | `23.3%` | `20.0%` | `23.3%` | tied B3g |
| Mean min target distance | `5.44` tiles | `7.51` tiles | `8.30` tiles | worse approach |
| Close-zone jump rate | `2.0%` | `14.5%` | `5.1%` | some action-shape gain |
| Close-zone idle/interact rate | `13.5%` | `3.6%` | `7.4%` | better than B3g |
| Stuck-after-close rate | `10.0%` | `3.3%` | `6.7%` | better than B3g |
| Loop-after-close rate | `20.0%` | `20.0%` | `23.3%` | slightly worse |

**Finding:** the two-stage idea improved the local action profile compared with B3g
but did not improve the task outcome. It tied B3g on `7/30` first-crystal hits while
falling from `36.2%` depth to `16.4%` and increasing mean min target distance from
`5.44` to `8.30` tiles. The fine-tune is acting like a narrow behavior shaper, not a
better route learner.

**Decision:** do not promote B3m. Keep B3g as the current route baseline. The bridge
lane remains useful evidence that jump/idle/stuck behavior can be shaped, but the route
policy is fragile and loses broad navigation when skill practice is mixed in.

**Next recommendation:** shift away from bridge-skill mixing and attack the route
fragility directly. The highest-signal next test is an objective-distance / progress
curriculum or a level-set simplification that grades route difficulty before the
first-crystal target, because the selected failures are still mostly `no_crystal`,
stalled, and tile-loop traces rather than close-zone execution failures.

### 2026-06-24 B3n. Tutorial Demo BC + Contextual Shoot Pressure

**Status:** completed; not promoted.

**Why this is next:** B3m closed the bridge-mixing line, but B3g's selected per-level
rows show another concrete action pathology: several failed held-out games spend large
fractions of the episode on `SHOOT`, `LEFT_SHOOT`, or `RIGHT_SHOOT`. B3g selected
failures split as 5 far failures (`>10` tiles), 9 mid failures (`5-10` tiles), and 9
near failures (`<=5` tiles), with 5 failures getting within `3` tiles and still missing.
That means route learning is still weak, but action spam may be hiding smaller
improvements. This probe keeps B3g's current-best training recipe and adds only
contextual pressure against shots that cannot fire or have no plausible target in the
firing lane.

**Implementation:**

- Added opt-in `CRYSTAL_CAVES_INVALID_SHOOT_PENALTY`.
- Default game behavior is unchanged.
- When enabled, no-ammo/cooldown shoot attempts and fired shots with no plausible
  enemy/air-tank target in the firing lane receive an extra `-0.04`.
- Added `invalid_shoot_count`, `invalid_shoot_penalty_total`, and `shoot_action_frac`
  to trace and first-objective near-miss diagnostics.
- Exposed the flag as `--invalid-shoot-penalty` for `tutorial-demo-bc`.

**Validation before run:** focused game/status-session tests passed
(`111 passed in 8.10s`), ruff passed, and the full suite passed
(`1022 passed in 22.40s`).

**Artifact:** `.Codex/artifacts/cc_sessions/20260624_064246_tutorial_demo_bc_invalid_shoot_pool512_select30_300`

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-bc \
  --episodes 300 \
  --seed 0 \
  --eval-games 16 \
  --selected-eval-games 30 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --cave-pool-size 512 \
  --route-demo-levels 128 \
  --route-demo-max-steps 800 \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --invalid-shoot-penalty \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label tutorial_demo_bc_invalid_shoot_pool512_select30_300
```

**Promotion rule:** compare against B3g selected expanded eval (`7/30`, `36.2%` depth,
`5.44` mean min target distance). Promote only if selected wins beat B3g, or if wins
tie while depth/min-distance and shoot-action metrics improve without increasing
idle/interact, stuck-after-close, or loop-after-close. Reject if the penalty merely
suppresses shooting while lowering route success or depth.

**Source eval history:**

| Episode | Source wins | Crystal rate | Depth | Selection score | Mean score | Ends |
|---:|---:|---:|---:|---:|---:|---|
| after BC | `1/16` | `6.3%` | `32.6%` | `0.076` | `10.9` | `{'first_crystal_goal': 1, 'stalled': 15}` |
| 50 | `0/16` | `0.0%` | `19.2%` | `0.000` | `0.0` | `{'stalled': 5, 'timeout': 11}` |
| 100 | `0/16` | `0.0%` | `5.4%` | `0.000` | `0.0` | `{'stalled': 16}` |
| 150 | `1/16` | `6.3%` | `20.5%` | `0.076` | `6.3` | `{'first_crystal_goal': 1, 'stalled': 15}` |
| 200 | `1/16` | `6.3%` | `21.4%` | `0.078` | `25.0` | `{'first_crystal_goal': 1, 'stalled': 15}` |
| 250 | `1/16` | `6.3%` | `26.8%` | `0.076` | `6.3` | `{'first_crystal_goal': 1, 'stalled': 10, 'timeout': 5}` |
| 300 | `3/16` | `18.8%` | `39.3%` | `0.231` | `60.9` | `{'first_crystal_goal': 3, 'stalled': 7, 'timeout': 6}` |

B3n selected ep300 for expanded eval.

**Selected expanded comparison:**

| Metric | B3g tutorial demo BC | B3n invalid shoot | Read |
|---|---:|---:|---|
| Selected expanded eval | `7/30` | `5/30` | worse first-crystal success |
| Selected crystal rate | `23.3%` | `16.7%` | worse |
| Selected depth | `36.2%` | `38.1%` | slightly deeper |
| Near-miss <=3 tiles | `40.0%` | `40.0%` | tied |
| Near-miss <=1.5 tiles | `23.3%` | `20.0%` | slightly worse |
| Mean min target distance | `5.44` tiles | `5.25` tiles | tiny improvement |
| Close-zone jump rate | `2.0%` | `2.3%` | tied |
| Close-zone idle/interact rate | `13.5%` | `15.9%` | worse |
| Stuck-after-close rate | `10.0%` | `10.0%` | tied |
| Loop-after-close rate | `20.0%` | `36.7%` | worse |
| Mean shoot action frac | not recorded in B3g | `3.7%` | new metric |
| Mean invalid shoot count | not applicable | `108.9` | still substantial |
| Mean invalid shoot penalty | not applicable | `-4.35` | flag was active |

**Shoot-spam diagnostic:** using the older B3g top-action rows as a lower bound, shoot
actions appeared `5,856` times inside the top-5 action lists across the 30 selected
rows, including 14 failed rows. B3n reduced that lower-bound count to `3,084` and 11
failed rows, but the policy still found shoot-heavy loops on some levels (`LEFT_SHOOT`
was 46.5% of actions on one failed selected row), and selected trace still averaged
`527.2` invalid-shoot presses over the 4 traced held-out games.

**Finding:** contextual shoot pressure changed behavior but did not improve the route
gate. It slightly improved depth/min-distance while reducing selected first-crystal
success from `7/30` to `5/30` and worsening close-zone loops. This suggests shoot spam
is a symptom of a weak route policy rather than the main cause.

**Decision:** do not promote B3n. Keep the shoot metrics and failure-mode classifier
because they improve visibility, but leave `CRYSTAL_CAVES_INVALID_SHOOT_PENALTY` off by
default and do not combine it with B3g.

**Next recommendation:** stop adding action penalties. The remaining evidence points
back to route-data quality and policy selection: B3g is still the keeper, but it is only
`7/30`. The next higher-value move is either (a) build a stronger scripted/planner route
oracle so demo BC is less biased than the current `35/128` successful heuristic, or
(b) rerun B3g with saved selected checkpoints and a larger diagnostic eval so we can
inspect policy behavior without retraining every time.

### 2026-06-24 B3o. Multi-Variant Tutorial Route Demonstrations

**Status:** completed; conditional data-quality keeper, not a clear route baseline.

**Why this is next:** B3n showed that action penalties can change symptoms without
improving route success. The current best B3g depends on weak on-distribution
demonstrations: only `35/128` tutorial levels produce a successful scripted trajectory.
Before changing the DQN again, improve the data source feeding behavior cloning and demo
replay.

**Implementation:**

- Added per-attempt demo diagnostics: initial/min/final target distance, best approach
  delta, step of best approach, action counts, and failure modes.
- Added `--route-demo-variants`, default `direct` so older runs stay reproducible.
- Added a `recovery` variant that is tried only after the direct controller fails. It
  uses the same target objective but, after stale progress, tries jump/backoff escape
  patterns instead of endlessly pushing into a wall.
- Added a `sweep` variant for close-zone no-jump left/right sweeps. It helped only
  marginally in pre-run checks, so B3o will use `direct,recovery`.

**Demo-only sanity check:**

| Controller variants | Wins | Controller attempts | Kept transitions | Close-zone transitions | Kept by variant |
|---|---:|---:|---:|---:|---|
| `direct` | `35/128` | `128` | `4,149` | `582` | `{'direct': 35}` |
| `direct,recovery` | `66/128` | `221` | `14,644` | `1,632` | `{'direct': 35, 'recovery': 31}` |
| `direct,recovery,sweep` | `67/128` | `283` | `14,774` | `1,647` | `{'direct': 35, 'recovery': 31, 'sweep': 1}` |

**Validation before run:** focused status-session tests passed (`36 passed`), ruff
passed, and the full suite passed (`1022 passed in 22.58s`). A mistaken `black` call on
the Markdown tracker failed because it is not Python; rerunning Black on Python files
passed.

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-bc \
  --episodes 300 \
  --seed 0 \
  --eval-games 16 \
  --selected-eval-games 30 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --cave-pool-size 512 \
  --route-demo-levels 128 \
  --route-demo-max-steps 800 \
  --route-demo-variants direct,recovery \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label tutorial_demo_bc_recovery_pool512_select30_300
```

**Promotion rule:** compare against B3g (`7/30`, `36.2%` depth, `5.44` mean min target
distance). Promote only if the selected expanded eval beats B3g or ties B3g while
improving depth/min-distance and not worsening close-zone loop/stuck rates. If demo
coverage improves but selected route success does not, the current heuristic demos are
still too biased; the next route-data step should be a real planner/oracle rather than
more hand-coded recovery variants.

**Artifact:** `.Codex/artifacts/cc_sessions/20260624_082654_tutorial_demo_bc_recovery_pool512_select30_300`

**Source eval history:**

| Episode | Source wins | Crystals | Depth | Selection score | Mean score | Ends |
|---:|---:|---:|---:|---:|---:|---|
| after BC | 1/16 | 6.3% | 27.7% | 0.076094 | 10.9 | `{'first_crystal_goal': 1, 'stalled': 15}` |
| 50 | 0/16 | 0.0% | 29.5% | 0.001875 | 18.8 | `{'stalled': 5, 'timeout': 11}` |
| 100 | 2/16 | 12.5% | 30.4% | 0.153594 | 35.9 | `{'first_crystal_goal': 2, 'stalled': 3, 'timeout': 11}` |
| 150 | 0/16 | 0.0% | 34.8% | 0.003125 | 31.3 | `{'stalled': 8, 'timeout': 8}` |
| 200 | 1/16 | 6.3% | 35.3% | 0.080625 | 56.3 | `{'first_crystal_goal': 1, 'stalled': 8, 'timeout': 7}` |
| 250 | 1/16 | 6.3% | 35.7% | 0.076406 | 14.1 | `{'first_crystal_goal': 1, 'stalled': 4, 'timeout': 11}` |
| 300 | 2/16 | 12.5% | 46.9% | 0.153906 | 39.1 | `{'first_crystal_goal': 2, 'stalled': 3, 'timeout': 11}` |

**Selected expanded comparison:**

| Metric | B3g tutorial demo BC | B3n invalid shoot | B3o recovery demos |
|---|---:|---:|---:|
| Selected expanded eval | 7/30 | 5/30 | 7/30 |
| Selected crystal rate | 23.3% | 16.7% | 23.3% |
| Selected depth | 36.2% | 38.1% | 44.8% |
| Near-miss <=3 tiles | 40.0% | 40.0% | 43.3% |
| Near-miss <=1.5 tiles | 23.3% | 20.0% | 26.7% |
| Mean min target distance | 5.44 | 5.25 | 5.57 |
| Close-zone jump rate | 2.0% | 2.3% | 3.7% |
| Close-zone idle/interact | 13.5% | 15.9% | 13.2% |
| Stuck-after-close | 10.0% | 10.0% | 3.3% |
| Loop-after-close | 20.0% | 36.7% | 33.3% |

**Finding:** B3o doubled demo coverage before training (`35/128` direct-only to
`66/128` direct+recovery) and tied B3g's first-crystal hits while improving depth,
near-miss bands, close-zone jumping, idle/interact, and stuck-after-close. It did not
clear the promotion rule because mean min distance was slightly worse and
loop-after-close rose from `20.0%` to `33.3%`.

**Decision:** keep B3o's richer demo diagnostics and recovery controller as useful
training data, but do not call it a solved improvement. Use it as a conditional
route-data baseline only when the next test needs the larger demo pool. The next area is
a planner-assisted controller, because adding action penalties and hand-coded recovery
patterns changed symptoms without producing more held-out route wins.

### 2026-06-24 B3p. Planner-Assisted Route Demonstrations

**Status:** completed; not promoted.

**Why this is next:** B3o showed that better demo coverage can improve secondary route
signals but still leaves the learned policy at `7/30`. The remaining data issue is
that the controller is not actually planning; it pushes toward the target and uses
reactive recovery. A bounded lookahead controller can simulate short action macros on a
copied game state, choose the macro that most improves target distance or wins, and
produce less biased demonstrations without changing final/eval levels.

**Implementation:**

- Added opt-in `beam` to `--route-demo-variants`.
- Added `route_beam_plan`, which deep-copies the current headless game, simulates a
  small set of short movement/jump macros, scores win/progress/best-target-distance
  improvement/final-target-distance/health loss, and commits only the first 8 actions.
- Kept the default variant as `direct`; old runs remain reproducible.
- Added focused tests for parser acceptance and side-effect-free planner simulation.
- Added selected-checkpoint path reporting and an opt-in saved selected snapshot when
  `--save-checkpoints` is used, so promising policies can be re-evaluated later without
  retraining.

**Validation:** Black and Ruff passed for `experiments/cc_status_session.py` and
`tests/test_cc_status_session.py`; focused status-session tests passed (`38 passed`).
After the final selected-checkpoint reporting fix, the full suite passed
(`1024 passed in 22.70s`).

**Demo-only sanity check:**

| Controller variants | Wins | Levels | Controller attempts | Kept transitions | Close-zone transitions | Kept by variant |
|---|---:|---:|---:|---:|---:|---|
| `direct,recovery` | 17 | 32 | 54 | 3,150 | 263 | `{'direct': 10, 'recovery': 7}` |
| `direct,recovery,beam` | 20 | 32 | 69 | 3,674 | 362 | `{'direct': 10, 'recovery': 7, 'beam': 3}` |
| `direct,recovery,beam` | 85 | 128 | 283 | 18,238 | 2,136 | `{'direct': 35, 'recovery': 31, 'beam': 19}` |

**128-level gate result:** B3p materially beats B3o's demo source (`66/128`,
`14,644` transitions, `1,632` close-zone transitions) with `85/128`, `18,238`
transitions, and `2,136` close-zone transitions. The planner adds 19 successful
trajectories beyond direct+recovery on the same 128-level pool. It is slower, but the
coverage gain is large enough to justify one full training run.

**B3p training command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-bc \
  --episodes 300 \
  --seed 0 \
  --eval-games 16 \
  --selected-eval-games 30 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --cave-pool-size 512 \
  --route-demo-levels 128 \
  --route-demo-max-steps 800 \
  --route-demo-variants direct,recovery,beam \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --save-checkpoints \
  --label tutorial_demo_bc_beam_pool512_select30_300
```

**Artifact:** `.Codex/artifacts/cc_sessions/20260624_090216_tutorial_demo_bc_beam_pool512_select30_300`

**Source eval history:**

| Episode | Source wins | Crystals | Depth | Selection score | Mean score | Ends |
|---:|---:|---:|---:|---:|---:|---|
| after BC | 1/16 | 6.3% | 27.7% | 0.076094 | 10.9 | `{'first_crystal_goal': 1, 'stalled': 15}` |
| 50 | 1/16 | 6.3% | 22.8% | 0.075625 | 6.3 | `{'first_crystal_goal': 1, 'stalled': 5, 'timeout': 10}` |
| 100 | 0/16 | 0.0% | 36.2% | 0.002656 | 26.6 | `{'stalled': 5, 'timeout': 11}` |
| 150 | 2/16 | 12.5% | 37.5% | 0.153906 | 39.1 | `{'first_crystal_goal': 2, 'stalled': 4, 'timeout': 10}` |
| 200 | 1/16 | 6.3% | 36.2% | 0.078750 | 37.5 | `{'first_crystal_goal': 1, 'stalled': 5, 'timeout': 10}` |
| 250 | 2/16 | 12.5% | 30.8% | 0.154375 | 43.8 | `{'first_crystal_goal': 2, 'stalled': 1, 'timeout': 13}` |
| 300 | 3/16 | 18.8% | 36.2% | 0.228750 | 37.5 | `{'first_crystal_goal': 3, 'stalled': 1, 'timeout': 12}` |

**Selected expanded comparison:**

| Metric | B3g tutorial demo BC | B3o recovery demos | B3p planner demos |
|---|---:|---:|---:|
| Demo wins | 35/128 | 66/128 | 85/128 |
| Demo transitions | 4,149 | 14,644 | 18,238 |
| Close-zone demo transitions | 582 | 1,632 | 2,136 |
| Selected source eval | 2/16 | 2/16 | 3/16 |
| Selected expanded eval | 7/30 | 7/30 | 4/30 |
| Selected crystal rate | 23.3% | 23.3% | 13.3% |
| Selected depth | 36.2% | 44.8% | 36.4% |
| Near-miss <=3 tiles | 40.0% | 43.3% | 33.3% |
| Near-miss <=1.5 tiles | 23.3% | 26.7% | 13.3% |
| Mean min target distance | 5.44 | 5.57 | 5.84 |
| Close-zone jump rate | 2.0% | 3.7% | 0.1% |
| Close-zone idle/interact | 13.5% | 13.2% | 9.2% |
| Stuck-after-close | 10.0% | 3.3% | 3.3% |
| Loop-after-close | 20.0% | 33.3% | 23.3% |

**Finding:** planner demos improved the data source but hurt the learned greedy policy.
The selected source eval looked better on the small 16-game sample (`3/16`), but the
larger selected eval fell to `4/30`, with worse near-miss bands and almost no close-zone
jumping. This is strong evidence that simply adding more heterogeneous scripted
trajectories can dilute the behavior cloning target instead of improving route mastery.

**Checkpoint/tooling note:** running with `--save-checkpoints` made this artifact `2.2G`
because the trainer wrote replay-buffer-heavy final checkpoints. It did not record
`selected_checkpoint_path` for this B3p run because the selected-snapshot patch was
initially added to a different mode; the tutorial-demo path has now been patched for
future runs. For this run, selected source was ep300, so the existing periodic checkpoint
is
`.Codex/artifacts/cc_sessions/20260624_090216_tutorial_demo_bc_beam_pool512_select30_300/tutorial_demo_bc/models/crystal_caves/crystal_caves_ep300.pth`.

**Decision:** do not promote B3p. Keep the `beam` controller as a diagnostic option, not
as default training data. Stop spending more runs on broadening scripted demos. The next
high-signal area should be evaluation/tooling and data selection: save selected
checkpoints without replay buffers, support larger re-evals without retraining, and
filter or weight demonstrations instead of adding every successful trajectory.

### 2026-06-24 B3q. Recommended Next Area: Evaluation + Demo Selection

**Status:** implemented and smoke-tested.

**Why this is next:** B3p showed two things at once: the 16-game source eval can
overrate a policy (`3/16` became only `4/30`), and `--save-checkpoints` is too heavy for
routine status sessions because it writes replay-buffer checkpoints. Before another
training idea, make the measurement loop cheaper and more reliable.

**Implementation:**

- Add a selected-only checkpoint flag that saves just policy/target weights and metadata,
  without replay memory or trainer history.
- Add a small `reeval-selected` or `eval-checkpoint` mode that loads that selected
  snapshot and runs 60-100 held-out games plus near-miss diagnostics without retraining.
- Add demo-source filters for the next training experiment: compare using only shorter
  successful demos, only direct+recovery demos, or down-weight `beam` demos instead of
  feeding every successful trajectory equally.

**Completed fix:**

- Added `save_selected_weight_snapshot` / `load_selected_weight_snapshot`, with a
  `cc_status_selected_weights_v1` payload containing only policy weights, target weights,
  config metadata, state/action sizes, source eval, and selected episode.
- Added `--save-selected-checkpoint` for the tutorial demo modes. This is separate from
  `--save-checkpoints`, so future sessions can keep re-evaluable selected policies
  without writing replay-buffer-heavy trainer snapshots.
- Added `eval-checkpoint` mode with `--checkpoint`, which reconstructs the saved config,
  validates state/action sizes, loads the selected policy, and runs final held-out eval,
  failure tracing, and near-miss eval without retraining.
- Added markdown reporting for checkpoint source metadata, so re-eval reports show which
  training run/episode produced the policy.

**Smoke validation:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py eval-checkpoint \
  --checkpoint .Codex/artifacts/cc_checkpoint_smoke/selected_smoke.pth \
  --seed 123 \
  --eval-games 1 \
  --trace-eval-games 1 \
  --trace-max-steps 50 \
  --trace-sample-every 25 \
  --trace-tail-steps 20 \
  --log-every 1 \
  --report-seconds 1 \
  --label eval_checkpoint_smoke_after_validation_order
```

**Smoke artifacts:**

- Selected-only checkpoint: `.Codex/artifacts/cc_checkpoint_smoke/selected_smoke.pth`
  (`6.1M`).
- Eval-only session: `.Codex/artifacts/cc_sessions/20260624_105917_eval_checkpoint_smoke_after_validation_order`
  (`56K`).
- Report line verified: `Checkpoint eval source: selected_smoke ep 0 (0/1 source wins)`.

**Validation:** Black unchanged, Ruff passed, focused status-session tests passed
(`40 passed in 2.59s`), and the full suite passed (`1026 passed in 21.95s`).

**Promotion rule:** do not run another 300-episode training tweak until the selected
checkpoint can be re-evaluated cheaply on a larger held-out set. This addresses the
user-facing concern that smaller in-between improvements may be missed or over-read.

### 2026-06-24 Research Update: Where To Go Next

**Question:** B3p produced better scripted data but worse greedy policy. Should the next
step be more level generation, more reward shaping, more demos, or a different training
method?

**Research read:**

- DQfD combines Q-learning, prioritized replay, and a supervised large-margin
  demonstration loss to bootstrap DQN from small demo sets:
  <https://arxiv.org/abs/1704.03732>.
- DAgger addresses behavior-cloning distribution shift by repeatedly collecting states
  from the learned policy and labeling those states with the expert/controller:
  <https://proceedings.mlr.press/v15/ross11a.html>.
- CQL targets a common offline-RL failure mode: overestimated Q-values for actions that
  are outside the dataset distribution. Its core idea is a conservative Q regularizer:
  <https://arxiv.org/abs/2006.04779>.
- IQL avoids directly evaluating unseen actions and extracts a policy with
  advantage-weighted behavioral cloning, which is relevant because our demo data is
  mixed-quality and sparse:
  <https://arxiv.org/abs/2110.06169>.
- AWAC is built for offline-data pretraining followed by online fine-tuning, using
  advantage-weighted maximum-likelihood policy updates:
  <https://arxiv.org/abs/2006.09359>.

**Interpretation for this repo:** B3p is exactly the kind of warning these methods are
designed around. Feeding every successful scripted trajectory equally can teach a
multi-modal, inconsistent policy. The agent may imitate a mixed bag of direct, recovery,
and beam behaviors without learning a stable greedy route policy. Reward-only changes
are still low priority because the policy usually fails before reaching terminal reward
events.

**Ranked next options:**

| Rank | Option | Why | Risk | Decision |
|---:|---|---|---|---|
| 1 | **B3r: demo filtering + weighting** | Directly attacks the B3p failure: better source demos hurt because the cloned target got noisier. Keep short/direct/recovery successes, down-weight or exclude beam, and preserve selected-only checkpoints for 60-100 game re-eval. | Low/medium | Tested; not promoted |
| 2 | **B3s: conservative DQN/demo regularizer** | CQL/IQL/DQfD all point at constraining learned actions around supported/demo-good behavior. A discrete conservative-Q penalty or advantage-weighted demo loss may reduce bad greedy actions. | Medium/high | **Do next** |
| 3 | **B3t: DAgger-style relabeling** | The current policy likely visits states the scripted demos do not cover. Roll out the current policy, label those encountered states with the controller, aggregate, retrain. | High | Promising but more invasive |
| 4 | **B3u: skill/hierarchical controller** | The task is a chain of skills. Options or subgoals may be more learnable than one flat DQN policy. | High | Later, after route gate is measured cleanly |
| 5 | **More reward/terminal tuning** | Already weak in prior tests because the agent rarely triggers the reward event. | Low implementation, low expected value | Defer |
| 6 | **More broad level generation** | Training levels are allowed, but current blocker is not final-level authenticity. B3p says source coverage alone is not enough. | Medium | Use only as support for B3r/B3t |

**B3r experiment plan (completed below).**

Build a demo-selection layer before the next 300-episode run:

- Add per-trajectory demo stats: variant, steps, transitions, close-zone transitions,
  close-zone jump rate, idle/interact rate, max tile-loop share, and final success.
- Add a filter/weight option for tutorial demo BC:
  - baseline data source: `direct,recovery,beam` collected on the same 128-level pool;
  - keep all `direct` and `recovery` successes;
  - include `beam` only if trajectory length is not in the worst quartile and close-zone
    jump/alignment behavior is present, or weight beam transitions lower than
    direct/recovery transitions;
  - cap long recovery trajectories so one long, messy solve cannot dominate BC batches.
- Run one candidate with `--save-selected-checkpoint`, then re-evaluate the saved
  selected policy with `eval-checkpoint` on 60-100 held-out games.

**B3r promotion rule:** compare against B3g/B3o's `7/30` selected first-crystal result
and B3o's near-miss profile. Promote only if B3r reaches at least `8/30`, or ties
`7/30` while improving at least two near-miss metrics (`<=3` tiles, `<=1.5` tiles,
close-zone jump rate, loop-after-close, or mean minimum target distance). If the 30-game
selected result is promising, use `eval-checkpoint` for a 60-100 game confirmation
before treating it as real.
