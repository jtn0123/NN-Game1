# Crystal Caves NN Tracker Archive: Findings 2026-06-23 Part 1

Archived from `CC_NN_EXPERIMENT_TRACKER.md` during cleanup on 2026-06-24.

## Findings Log

### 2026-06-23 A1: Invalid `INTERACT` Pressure

**Hypothesis:** penalizing useless `INTERACT` will reduce action spam and free the policy
to spend more decisions on movement toward objectives.

**Command template:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py invalid-interact \
  --episodes 150 \
  --seed 0 \
  --eval-games 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label invalid_interact_150
```

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_072346_invalid_interact_150`

**Result:** not a keeper as a standalone change.

| Metric | Baseline 150 | Anti-loop 150 | Invalid-interact 150 |
|---|---:|---:|---:|
| Held-out wins | 0/8 | 0/8 | 0/8 |
| Held-out crystals | 12.5% | 25.0% | 12.5% |
| Held-out depth | 29.5% | 33.0% | 30.4% |
| Trace any crystal | 0.0% | 0.0% | 0.0% |
| Trace depth | 26.8% | 28.6% | 26.8% |
| Trace tile-loop max visit | 69.9% | 58.2% | 60.6% |
| Trace idle action share | 6.2% | 6.6% | 36.8% |
| Trace interact action share | not recorded | not recorded | 0.02% |
| Trace invalid interact penalty | not applicable | not applicable | -0.03 avg |

**Verdict:** the comparable run did not reproduce `INTERACT` spam; the trace only
averaged 0.5 invalid presses across 3000-step held-out games. The penalty therefore
did not address the stable bottleneck. It slightly improved loop share versus baseline
but regressed idle share and did not match anti-loop's crystal/depth signal. Keep the
telemetry and optional mode; do not enable it by default or combine it yet.

### 2026-06-23 A2: Bridge Demo Replay

**Hypothesis:** replay seeding with successful bridge trajectories will expose rare
successful collect/route/exit transitions early enough that a fresh full-cave policy
learns objective movement sooner than baseline.

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py bridge-demo-replay \
  --bridge-episodes 200 \
  --episodes 150 \
  --seed 0 \
  --eval-games 8 \
  --eval-k 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --bridge-eval-every 50 \
  --vec-envs 8 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --demo-repeat 4 \
  --label bridge_demo_replay_200_150
```

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_073832_bridge_demo_replay_200_150`

**Result:** not a keeper as implemented.

| Metric | Baseline 150 | Anti-loop 150 | Bridge-demo replay |
|---|---:|---:|---:|
| Held-out wins | 0/8 | 0/8 | 0/8 |
| Held-out crystals | 12.5% | 25.0% | 12.5% |
| Held-out depth | 29.5% | 33.0% | 12.5% |
| Held-out mean score | 112.5 | 150.0 | 75.0 |
| Ends | 3 stalled / 5 timeout | 1 stalled / 7 timeout | 2 stalled / 6 timeout |

**Source/demos:** bridge source selected ep 200 with 60% greedy bridge wins, 100%
any-crystal, 60% all-crystals, 3/5 bridge levels solved. Demo collection kept 24
winning trajectories from 40 attempts, totaling 22,672 transitions. Replay seeding
repeated those 4x for 90,688 pushed transitions before full-cave training.

**Verdict:** bridge-demo replay overloaded the fresh full-cave learner with bridge
state/action distribution but did not improve held-out tutorial behavior. It appears to
hurt depth badly versus both baseline and anti-loop. Keep the helper/report plumbing,
but do not promote this replay-seeding recipe. If demo replay is revisited, use
demonstrations from actual full tutorial caves or add a supervised/margin objective;
bridge-only replay is too off-distribution.

### 2026-06-23 A3: Reverse-Start Curriculum

**Hypothesis:** a small number of lanes that start near the current objective or near an
already-unlocked exit will teach the policy the end of the task without replacing normal
full-cave training.

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py reverse-start \
  --episodes 150 \
  --seed 0 \
  --eval-games 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --reverse-start-ratio 0.25 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label reverse_start_150
```

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_080815_reverse_start_150`

**Result:** not a keeper as implemented.

| Metric | Baseline 150 | Anti-loop 150 | Reverse-start 150 |
|---|---:|---:|---:|
| Held-out wins | 0/8 | 0/8 | 0/8 |
| Held-out crystals | 12.5% | 25.0% | 0.0% |
| Held-out depth | 29.5% | 33.0% | 25.9% |
| Held-out mean score | 112.5 | 150.0 | 0.0 |
| Trace any crystal | 0.0% | 0.0% | 0.0% |
| Trace depth | 26.8% | 28.6% | 12.5% |
| Trace tile-loop max visit | 69.9% | 58.2% | 55.3% |

**Source split:** reverse starts applied successfully every time:
`reverse_exit` 46/46, `reverse_objective` 17/17. The reverse lanes learned local
success (`reverse_exit` 73% win, `reverse_objective` 19% win), but normal full-start
lanes stayed weak (`full` 7% training win, 0 held-out wins). Final held-out collected no
crystals.

**Verdict:** reverse-start practice did not transfer back to normal starts. Like bridge
demo replay, it improved the easier source distribution while the actual held-out task
stayed worse than anti-loop. Keep the runner/report plumbing. Do not promote this mix.
If reverse curriculum is revisited, it likely needs a schedule that gradually moves
starts backward along actual successful full-cave routes, not static near-objective
teleports mixed into normal training.

### 2026-06-23 A4: Archive-Start Curriculum

**Hypothesis:** a small Go-Explore-lite archive of real mid-run states can reduce local
loop collapse by letting a few training lanes repeatedly practice from useful states
the normal full-cave lanes already discovered.

**Implementation:** added experiment-only `archive-start` mode to
`experiments/cc_status_session.py`. It archives deep-copied full-lane states keyed by
objective kind, coarse player region, crystals collected, and depth bucket. Archive
lanes reset from archived states with configurable probability, while held-out eval and
trace diagnostics remain normal full tutorial caves.

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py archive-start \
  --episodes 150 \
  --seed 0 \
  --eval-games 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --archive-start-ratio 0.25 \
  --archive-replay-prob 0.7 \
  --archive-max-size 64 \
  --archive-min-steps 30 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label archive_start_150
```

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_095004_archive_start_150`

**Result:** not a keeper as implemented.

| Metric | Baseline 150 | Anti-loop 150 | Archive-start 150 |
|---|---:|---:|---:|
| Held-out wins | 0/8 | 0/8 | 0/8 |
| Held-out crystals | 12.5% | 25.0% | 12.5% |
| Held-out depth | 29.5% | 33.0% | 19.7% |
| Held-out mean score | 112.5 | 150.0 | 75.0 |
| Trace any crystal | 0.0% | 0.0% | 0.0% |
| Trace depth | 26.8% | 28.6% | 23.2% |
| Trace tile-loop max visit | 69.9% | 58.2% | 52.6% |

**Archive mechanics:** the archive did engage, so this was not a no-op. It stored 90
unique milestones, retained 64/64 snapshots, evicted 26, replayed 35/54 archive reset
attempts (65%), and had zero snapshot failures.

**Source split:** archive lanes looked stronger than full-start lanes (`archive` 10%
training win, 0.449 progress; `full` 3% training win, 0.348 progress), but held-out
normal starts did not improve. The final traces still collected 0/4 crystals and all
four traced games were `tile_loop` + `no_crystal`.

**Verdict:** archive starts reduce trace loop concentration somewhat, but not enough to
improve actual objective outcomes. Like A2/A3, this mainly improves the easier source
distribution and does not transfer back to full held-out starts. Do not promote this
mix. If archive exploration is revisited, it needs goal-directed archive selection
toward first-crystal reachability, not coarse replay of any mid-run state.

### 2026-06-23 A5: Novelty Bonus

**Hypothesis:** direct exploration pressure in normal full-start tutorial caves can
reduce local-loop collapse better than alternate-source starts. Anti-loop was the only
weakly positive prior probe, so this tests the positive version: pay once for reaching
new coarse regions instead of only penalizing repeated loops.

**Implementation:** added opt-in `CRYSTAL_CAVES_NOVELTY_BONUS`. The reward is `+0.08`
for first entry into a new global-map cell while an objective is active, capped at `3.0`
total per episode. The spawn cell is marked visited on reset, and diagnostics report
`novelty_bonus_total`.

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py novelty-bonus \
  --episodes 150 \
  --seed 0 \
  --eval-games 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label novelty_bonus_150
```

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_103934_novelty_bonus_150`

**Result:** not a keeper as implemented.

| Metric | Baseline 150 | Anti-loop 150 | Novelty-bonus 150 |
|---|---:|---:|---:|
| Held-out wins | 0/8 | 0/8 | 0/8 |
| Held-out crystals | 12.5% | 25.0% | 0.0% |
| Held-out depth | 29.5% | 33.0% | 23.2% |
| Held-out mean score | 112.5 | 150.0 | 0.0 |
| Trace any crystal | 0.0% | 0.0% | 0.0% |
| Trace depth | 26.8% | 28.6% | 23.2% |
| Trace tile-loop max visit | 69.9% | 58.2% | 48.2% |

**Mechanics:** the bonus did fire in held-out traces (`0.30` mean novelty reward), and
it reduced average max tile-loop concentration versus baseline/anti-loop. But it did
not move the important outcome metrics: the final held-out eval collected no crystals
and split evenly between timeout/stall endings.

**Verdict:** broad novelty pressure causes more movement, but not useful movement
toward the first crystal. Do not promote this version. If revisited, make novelty
goal-conditioned (new regions that reduce target distance or preserve a reachable path
to the current objective), not generic region coverage.

### 2026-06-23 Metrics/Progress Review

**Status:** documented in `CC_NN_METRICS_PROGRESS_REVIEW.md`.

**Reason:** the current top-line metrics are good enough to reject obvious failures, but
too coarse to distinguish small useful behavior changes from noise. In particular,
`6/30` versus `7/30` selected first-crystal eval is not decisive by itself, while trace
diagnostics show possible smaller signals such as better target-distance approach and
lower tile-loop concentration.

**Decision:** before the next NN/model tweak, add a first-objective near-miss eval:
per-level rows, min/final target distance, distance-band rates, step of closest
approach, close-zone actions, and loop-after-close rates. Use these to decide whether
the next intervention should target route planning, close-zone jump/alignment, or
post-crystal objective chaining.

**Implementation:** added to `experiments/cc_status_session.py`. Candidate summaries now
write `near_miss_eval` and, when selected checkpoint evaluation is enabled,
`selected_checkpoint_near_miss_eval`. Each detailed eval writes
`near_miss_eval/<label>/summary.json` plus
`near_miss_eval/<label>/per_level_eval.jsonl`.

**New report metrics:** `near_miss_rate_10`, `near_miss_rate_5`,
`near_miss_rate_3`, `near_miss_rate_1_5`, mean minimum/final target distance,
best/final target-distance delta, step of best approach, close-zone action mix,
stuck-after-close rate, and loop-after-close rate.

**Related fix:** lint caught an existing `run_first_crystal_pretrain` bug where the
selected `best_weights` return value was discarded and then referenced. Fixed the
assignment so first-crystal transfer paths keep the selected source weights correctly.

**Validation:** focused status-session tests passed (`36 passed in 2.70s`), full suite
passed (`1020 passed in 22.61s`), and touched-file ruff passed. A one-episode
`first-crystal-direct` smoke run also wrote final and selected near-miss artifacts under
`.Codex/artifacts/cc_sessions/20260623_205802_near_miss_smoke/`.

### 2026-06-23 Large-File / Rerun Decision

**Question:** could large files be hiding NN bugs, should they be broken up now, and do
we need to rerun old experiments with the new near-miss metrics?

**Large-file scan:**

| File | Lines | Top-level/indented defs/classes | Risk read |
|---|---:|---:|---|
| `experiments/cc_status_session.py` | 5,776 | 136 | high experiment-harness risk |
| `src/game/crystal_caves.py` | 1,171 | 43 | medium core environment risk |
| `src/ai/replay_buffer.py` | 1,050 | 42 | medium/high replay correctness risk |
| `src/ai/agent.py` | 1,025 | 28 | medium/high learning-loop risk |
| `src/game/crystal_caves_gen.py` | 904 | 26 | medium level-distribution risk |
| `src/ai/network.py` | 738 | 47 | medium model-shape risk |

**Finding:** yes, large files are a real bug-risk signal here. The status-session runner
is the biggest concern because it mixes configs, training modes, evals, diagnostics,
reporting, and artifact writing. We already found one bug there during lint
(`run_first_crystal_pretrain` discarded `best_weights` before referencing it), so the
concern is not theoretical.

**Decision on refactoring:** do **not** do a broad split before the next experiment.
Refactoring the harness now would risk breaking comparability right after adding the new
near-miss metrics. Instead:

1. Do a targeted audit of NN-sensitive paths in the large files.
2. Rerun the current baseline/top candidate with the new metrics.
3. After the next evidence batch, split `cc_status_session.py` into smaller modules with
   tests already protecting behavior.

**Recommended future split for `cc_status_session.py`:**

- `experiments/cc_session_configs.py` - config builders and overrides.
- `experiments/cc_session_evals.py` - final eval, source snapshots, near-miss eval,
  trace diagnostics.
- `experiments/cc_session_modes.py` - run modes such as direct, demo BC, transfer.
- `experiments/cc_session_report.py` - markdown/summary writers.
- `experiments/cc_status_session.py` - thin CLI dispatcher.

**Rerun decision:** do **not** rerun every old experiment. Most old runs are still useful
as top-line rejects. They are missing near-miss detail, but that does not make them
invalid. Rerun only experiments whose conclusions still affect the next choice:

1. **Rerun B3d direct first-crystal baseline** with selected `30`-game near-miss eval.
   Purpose: establish the new near-miss baseline for `6/30`.
2. **Rerun B3g tutorial demo BC** with selected `30`-game near-miss eval.
   Purpose: compare current best candidate against B3d on close-zone behavior, not only
   `7/30` vs `6/30`.
3. **Optional rerun B3h DQfD** only if B3g and B3d are close on near-miss metrics.
   Purpose: B3h tied B3g on selected eval (`7/30`) and had slightly better trace
   best-distance approach, but it was not promoted.

**Low-priority reruns:** reward-only, invalid-interact, archive, reverse-start, broad
novelty, bridge-demo replay, and short `8`-game reject runs. These did not move the
top-line outcome enough to justify rerunning before the current first-objective question
is resolved.

**Caution:** old first-crystal-transfer conclusions should be treated as lower
confidence until rerun, because the `best_weights` bug touched the
`run_first_crystal_pretrain` path. Do not spend rerun budget there until the
first-objective route policy is stronger.

### 2026-06-23 B3d/B3g Near-Miss Reruns

**Status:** completed.

**Purpose:** rerun the direct first-crystal baseline and the current best demo-BC
candidate after adding selected-checkpoint near-miss metrics. This answers whether the
old `7/30` vs `6/30` result was just noise or whether the demo-guided policy has a
different failure shape.

**Commands:** same protocols as B3d/B3g, new labels:

- `first-crystal-direct --label tutorial_route_pool512_select30_300_nearmiss`
- `tutorial-demo-bc --label tutorial_demo_bc_pool512_select30_300_nearmiss`

**Artifacts:**

- B3d rerun:
  `.Codex/artifacts/cc_sessions/20260623_210126_tutorial_route_pool512_select30_300_nearmiss`
- B3g rerun:
  `.Codex/artifacts/cc_sessions/20260623_212528_tutorial_demo_bc_pool512_select30_300_nearmiss`

| Metric | B3d direct route | B3g tutorial demo BC | Read |
|---|---:|---:|---|
| Selected checkpoint | ep150 | ep250 | matches old run shape |
| Selected expanded eval | `6/30` | `7/30` | B3g still +1 win |
| Selected depth | `28.6%` | `36.2%` | B3g clearly deeper |
| Near-miss <=10 tiles | `73.3%` | `83.3%` | B3g approaches target more often |
| Near-miss <=5 tiles | `36.7%` | `53.3%` | B3g has a real approach gain |
| Near-miss <=3 tiles | `33.3%` | `40.0%` | B3g modest close-zone gain |
| Near-miss <=1.5 tiles | `20.0%` | `23.3%` | mostly same as wins/crystals |
| Mean min target distance | `6.73` tiles | `5.44` tiles | B3g gets closer |
| Best target-distance delta | `6.42` tiles | `7.72` tiles | B3g improves approach |
| Final target-distance delta | `2.60` tiles | `2.80` tiles | similar final position |
| Close-zone steps | `5.4` | `32.3` | B3g spends much more time near target |
| Close-zone jump rate | `6.2%` | `2.0%` | B3g is not jumping enough near target |
| Close-zone idle/interact rate | `1.4%` | `13.5%` | B3g has more close-zone dithering |
| Stuck-after-close rate | `3.3%` | `10.0%` | B3g gets close then stalls more |
| Loop-after-close rate | `13.3%` | `20.0%` | B3g loops near target more |
| 4-game selected trace crystal rate | `0/4` | `0/4` | sparse trace still misses successes |
| 4-game selected trace depth | `26.8%` | `48.2%` | B3g trace goes deeper |

**Finding:** B3g is not merely a one-game top-line flicker. The near-miss metrics show a
real behavior change: demo BC improves route approach and gets the agent closer to the
current objective. The remaining bottleneck is not "cannot find the crystal at all"; it
is **close-zone execution**. B3g spends about `6x` more steps within 3 tiles of the
target than B3d, but has lower jump rate, higher idle/interact rate, and more
stuck/loop-after-close failures.

**Decision:** promote B3g as the current route baseline, but not as solved. Do not spend
the next run on more generic route guidance. The next improvement should target
close-zone action selection: when the agent is near the objective, teach or reward the
final alignment/jump/collect behavior and suppress dithering loops.

**Recommended next test:** a narrow "close-zone action assist" on top of B3g:

1. Reuse the existing successful tutorial demo transitions.
2. Filter or upweight demo states where target distance is `<=3` tiles.
3. Apply a higher action-supervision weight only for those close-zone states, while
   keeping the general demo action weight low/off.
4. Track whether close-zone jump rate rises, close-zone idle/interact rate falls, and
   `<=1.5` tile / first-crystal success improve.

**Promotion rule:** promote only if B3g's `7/30` selected eval improves by at least
`+2/30`, or if selected eval ties while close-zone jump rate rises and
stuck/loop-after-close rates fall materially.

### 2026-06-23 B3i. Close-Zone Demo Action Assist

**Status:** completed; not promoted over B3g.

**Why this is next:** B3g improved route approach but got stuck near the objective:
`40.0%` of selected held-out levels got within `<=3` tiles, but close-zone jump rate was
only `2.0%`, idle/interact was `13.5%`, and loop-after-close was `20.0%`. The next
smallest change is not more global routing. It is targeted action pressure on the final
few tiles before collection.

**Implementation:** added `tutorial-demo-close-zone` mode. It keeps B3g's demo source,
BC pretrain, replay seeding, eval schedule, and selected-checkpoint protocol, but the
online demo-action margin loss is trained only on successful demo transitions where the
objective was within `<=3` tiles. This is training-only; final levels and rewards are
unchanged.

**Validation before run:** focused status-session tests passed (`36 passed in 2.65s`),
touched-file ruff passed, and the full suite passed (`1020 passed in 22.61s`).

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-close-zone \
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
  --close-zone-demo-distance 3.0 \
  --close-zone-demo-action-weight 0.12 \
  --demo-action-margin 0.8 \
  --demo-action-batch-size 64 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label tutorial_demo_close_zone_w012_d3_pool512_select30_300
```

**Promotion rule:** compare against B3g near-miss rerun:
`7/30` selected eval, `36.2%` depth, `40.0% <=3` tiles, `23.3% <=1.5` tiles,
`2.0%` close-zone jump, `13.5%` close-zone idle/interact, `20.0%` loop-after-close.
Promote only if selected eval improves by at least `+2/30`, or if it ties while
close-zone jump rises and idle/interact plus loop-after-close fall materially.

**Artifact:**
`.Codex/artifacts/cc_sessions/20260623_215617_tutorial_demo_close_zone_w012_d3_pool512_select30_300`

**Run details:** demo collection found `35/128` successful scripted tutorial routes,
seeded `4,149` successful transitions into replay, and exposed `582` close-zone
transitions to online action supervision at weight `0.12`, margin `0.8`, batch `64`.
After pure BC, the source eval was only `1/16`; the useful signal appeared after RL
fine-tuning.

| Source eval episode | Wins | Crystal rate | Depth | Selection score |
|---:|---:|---:|---:|---:|
| 0 | `1/16` | `6.3%` | `32.6%` | `0.076` |
| 50 | `2/16` | `12.5%` | `42.0%` | `0.152` |
| 100 | `2/16` | `12.5%` | `26.3%` | `0.152` |
| 150 | `3/16` | `18.8%` | `28.1%` | `0.227` |
| 200 | `4/16` | `25.0%` | `38.4%` | `0.303` |
| 250 | `3/16` | `18.8%` | `31.3%` | `0.227` |
| 300 | `3/16` | `18.8%` | `7.1%` | `0.227` |

Selected checkpoint was ep200. The final 300-episode model drifted, so final-only
scoring would have understated the run.

| Metric | B3g tutorial demo BC | B3i close-zone assist | Read |
|---|---:|---:|---|
| Selected checkpoint | ep250 | ep200 | B3i peaks earlier |
| Selected expanded eval | `7/30` | `5/30` | B3i loses top-line wins |
| Selected crystal rate | `23.3%` | `16.7%` | worse objective completion |
| Selected depth | `36.2%` | `36.9%` | depth is roughly tied |
| Near-miss <=3 tiles | `40.0%` | `43.3%` | slightly closer more often |
| Near-miss <=1.5 tiles | `23.3%` | `20.0%` | not better at final contact |
| Mean min target distance | `5.44` tiles | `5.60` tiles | no real approach gain |
| Close-zone jump rate | `2.0%` | `0.4%` | worse; assist suppresses jumps |
| Close-zone idle/interact rate | `13.5%` | `5.2%` | idle/interact improves |
| Stuck-after-close rate | `10.0%` | `20.0%` | worse |
| Loop-after-close rate | `20.0%` | `33.3%` | worse |

**Finding:** the close-zone action assist attacked the wrong part of the problem. It
reduced idle/interact dithering and got the policy within `<=3` tiles a little more
often, but it also drove the close-zone jump rate down to almost zero and increased
stuck/loop-after-close failures. That explains the worse selected result despite the
episode-200 source peak.

**Decision:** do not promote `tutorial-demo-close-zone` as the default route baseline.
Keep the mode and metrics because they are useful diagnostics, but treat this specific
configuration (`<=3` tiles, weight `0.12`, successful-demo labels only) as rejected.

**Next recommendation:** stop adding more blind action imitation. The next small test
should separate the final-contact skill from global route learning, either by adding
close-zone counterfactual labels from a jump-aware oracle, or by training a short
late-stage drill/replay slice where the target is visible and one or two jumps away.
The key metric to move is not `<=3` reach rate; it is `<=1.5` contact, close-zone jump
rate, and lower stuck/loop-after-close.

### 2026-06-23 B3j. Bridge Interleaved Final-Contact Practice

**Status:** completed; not promoted. Follow-up required because objective was stricter
than B3g/B3i.

**Why this is next:** B3i showed that close-zone imitation can reduce idle/interact but
also suppresses jumps and worsens stuck/loop-after-close. The next safer approach is to
teach the skill through training-only levels instead of forcing action labels in normal
caves. This uses the existing bridge levels, especially the low exit hop / mini-route
cases, while evaluating only on unchanged procedural tutorial caves.

**Implementation:** upgraded `bridge-interleaved` to use the same source-eval snapshot,
selected-checkpoint restore, selected 30-game eval, failure trace, and near-miss rollup
as B3g/B3i. Also wired `--cave-pool-size` into the mode so its full tutorial lanes use
the same `512` cave pool as the route-demo runs.

**Validation before run:** focused status-session tests passed (`36 passed in 2.51s`),
touched-file ruff passed, and the full suite passed (`1020 passed in 21.95s`).

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py bridge-interleaved \
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
  --interleave-bridge-ratio 0.25 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label bridge_interleaved_25_pool512_select30_300
```

**Promotion rule:** compare against B3g (`7/30`, `36.2%` depth) and B3i's failure shape.
Promote only if selected eval reaches at least `8/30`, or ties B3g while improving
`<=1.5` contact, close-zone jump rate, and stuck/loop-after-close.

**Artifact:**
`.Codex/artifacts/cc_sessions/20260623_222515_bridge_interleaved_25_pool512_select30_300`

**Run details:** mixed training reached `70%` rolling training win rate and `0.791`
rolling progress, but this was dominated by the bridge lanes. The unchanged full
tutorial held-out eval was much weaker: selected checkpoint ep300 scored `1/30` full
wins, `26.7%` crystals, and `30.0%` depth.

| Source eval episode | Wins | Crystal rate | Depth | Selection score | Mean score |
|---:|---:|---:|---:|---:|---:|
| 50 | `0/16` | `12.5%` | `18.8%` | `0.033` | `84.4` |
| 100 | `0/16` | `12.5%` | `12.1%` | `0.033` | `79.7` |
| 150 | `0/16` | `18.8%` | `19.7%` | `0.049` | `117.2` |
| 200 | `0/16` | `12.5%` | `36.6%` | `0.033` | `79.7` |
| 253 | `0/16` | `18.8%` | `33.9%` | `0.051` | `135.9` |
| 300 | `1/16` | `37.5%` | `26.8%` | `0.171` | `339.1` |

| Metric | B3g tutorial demo BC | B3i close-zone assist | B3j bridge interleaved |
|---|---:|---:|---:|
| Selected checkpoint | ep250 | ep200 | ep300 |
| Selected eval | `7/30` first-crystal | `5/30` first-crystal | `1/30` full win |
| Selected crystal rate | `23.3%` | `16.7%` | `26.7%` |
| Selected depth | `36.2%` | `36.9%` | `30.0%` |
| Near-miss <=3 tiles | `40.0%` | `43.3%` | `40.0%` |
| Near-miss <=1.5 tiles | `23.3%` | `20.0%` | `26.7%` |
| Mean min target distance | `5.44` tiles | `5.60` tiles | `6.50` tiles |
| Close-zone jump rate | `2.0%` | `0.4%` | `13.4%` |
| Close-zone idle/interact rate | `13.5%` | `5.2%` | `9.1%` |
| Stuck-after-close rate | `10.0%` | `20.0%` | `3.3%` |
| Loop-after-close rate | `20.0%` | `33.3%` | `40.0%` |

**Finding:** training-only bridge lanes do teach more close-zone jumping, but this run
is not a fair top-line comparison with B3g/B3i because B3j evaluated full-level wins
while B3g/B3i evaluated the first-crystal route objective. The mixed-lane training
metrics were actively misleading: bridge lanes won often, but normal tutorial held-out
games mostly timed out.

**Decision:** do not promote B3j. Keep the selected-checkpoint/near-miss upgrade for
`bridge-interleaved`, but rerun the idea with the full tutorial lanes set to the same
first-crystal objective as B3g/B3i and the bridge lanes kept as full bridge-skill
practice. This corrected run should answer whether bridge skill practice helps the
route objective without changing the comparison target.

### 2026-06-23 B3k. Corrected Bridge-Route Interleave

**Status:** completed; not promoted over B3g.

**Why this is next:** B3j's full-win eval was useful but not directly comparable to the
route-demo baseline. B3k keeps the same `25%` bridge practice mix but sets the normal
tutorial lanes to the same first-crystal objective used by B3g/B3i. Bridge lanes remain
full bridge-skill practice, so they still teach collect/jump/exit behavior rather than
ending after the first bridge crystal.

**Implementation:** added `--interleave-first-crystal-goal` for `bridge-interleaved`.
When enabled, full tutorial lanes use `CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL=True`, while
bridge lanes force `CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL=False`. Artifacts record
`full_lane_goal` and `bridge_lane_goal` under `interleave`.

**Validation before run:** focused status-session tests passed (`36 passed in 2.72s`),
touched-file ruff passed, and the full suite passed (`1020 passed in 22.61s`).

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py bridge-interleaved \
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
  --interleave-bridge-ratio 0.25 \
  --interleave-first-crystal-goal \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label bridge_route_interleaved_25_pool512_select30_300
```

**Promotion rule:** compare directly against B3g's first-crystal route baseline:
`7/30`, `23.3%` crystals, `36.2%` depth, `40.0% <=3`, `23.3% <=1.5`, `2.0%`
close-zone jump, `10.0%` stuck-after-close, `20.0%` loop-after-close. Promote only if
selected eval improves by at least `+2/30`, or if it ties while improving final-contact
and stuck/loop metrics.

**Artifact:**
`.Codex/artifacts/cc_sessions/20260623_225222_bridge_route_interleaved_25_pool512_select30_300`

**Run details:** mixed training finished with `78%` rolling win rate and `0.807` rolling
progress, but selected-checkpoint restore chose ep150. Later training improved mixed
training score while source eval drifted.

| Source eval episode | Wins | Crystal rate | Depth | Selection score | Mean score |
|---:|---:|---:|---:|---:|---:|
| 50 | `0/16` | `0.0%` | `10.7%` | `0.002` | `18.8` |
| 100 | `2/16` | `12.5%` | `22.8%` | `0.151` | `12.5` |
| 150 | `4/16` | `25.0%` | `24.1%` | `0.303` | `25.0` |
| 200 | `2/16` | `12.5%` | `28.6%` | `0.151` | `12.5` |
| 250 | `3/16` | `18.8%` | `20.1%` | `0.227` | `23.4` |
| 300 | `4/16` | `25.0%` | `8.0%` | `0.303` | `29.7` |

| Metric | B3g tutorial demo BC | B3k bridge-route interleave | Read |
|---|---:|---:|---|
| Selected checkpoint | ep250 | ep150 | B3k peaks earlier |
| Selected expanded eval | `7/30` | `5/30` | B3k loses top-line wins |
| Selected crystal rate | `23.3%` | `16.7%` | worse route completion |
| Selected depth | `36.2%` | `21.7%` | much shallower |
| Near-miss <=3 tiles | `40.0%` | `40.0%` | tied |
| Near-miss <=1.5 tiles | `23.3%` | `20.0%` | slightly worse |
| Mean min target distance | `5.44` tiles | `6.14` tiles | worse approach |
| Close-zone jump rate | `2.0%` | `17.5%` | bridge practice teaches jumping |
| Close-zone idle/interact rate | `13.5%` | `2.9%` | less dithering |
| Stuck-after-close rate | `10.0%` | `0.0%` | better |
| Loop-after-close rate | `20.0%` | `20.0%` | tied |

**Finding:** bridge practice fixed part of the local action problem: close-zone jump
rate rose sharply, idle/interact fell, and stuck-after-close went to zero. But the
global route policy got worse: selected wins fell to `5/30`, depth fell to `21.7%`, and
mean min target distance worsened. This suggests `25%` bridge interleaving is too much
or too unaligned; it teaches final-contact movement while diluting normal route
learning.

**Decision:** do not promote B3k. The useful signal is the metric split: bridge-style
skill practice can improve close-zone action selection, but it must be introduced more
lightly or after B3g route learning rather than mixed at `25%` from scratch.

**Next recommendation:** if continuing this line, test a smaller bridge dose (`12.5%`,
one bridge lane out of eight) or a two-stage run: first B3g demo BC to learn route
approach, then a short bridge interleave fine-tune. Do not repeat `25%` bridge from
scratch.

### 2026-06-24 B3l. Corrected Bridge-Route Interleave at 12.5%

**Status:** completed; improved over B3k, not promoted over B3g.

**Why this is next:** B3k proved bridge practice can improve close-zone action
selection (`17.5%` close-zone jump, `0%` stuck-after-close), but it damaged route
learning at a `25%` bridge mix. The smallest follow-up is to keep the corrected
first-crystal route objective and reduce bridge exposure to one lane out of eight
(`12.5%`). This tests whether the issue is dose/dilution rather than the bridge skill
practice itself.

**Command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py bridge-interleaved \
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
  --interleave-bridge-ratio 0.125 \
  --interleave-first-crystal-goal \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label bridge_route_interleaved_125_pool512_select30_300
```

**Promotion rule:** compare directly against B3g and B3k. Promote only if selected
eval reaches at least B3g's `7/30` while keeping B3k's improved close-zone behavior
directionally intact, or if it exceeds B3g by `+2/30`.

**Artifact:**
`.Codex/artifacts/cc_sessions/20260624_051947_bridge_route_interleaved_125_pool512_select30_300`

**Run details:** the source checkpoint selected ep300. The smaller bridge dose improved
source wins late (`5/16`) and selected wins (`6/30`) versus B3k's `5/30`, but still did
not beat B3g's `7/30`.

| Source eval episode | Wins | Crystal rate | Depth | Selection score | Mean score |
|---:|---:|---:|---:|---:|---:|
| 50 | `0/16` | `0.0%` | `12.5%` | `0.000` | `0.0` |
| 100 | `3/16` | `18.8%` | `23.2%` | `0.227` | `23.4` |
| 150 | `3/16` | `18.8%` | `34.8%` | `0.229` | `37.5` |
| 200 | `4/16` | `25.0%` | `21.4%` | `0.305` | `48.4` |
| 250 | `3/16` | `18.8%` | `25.9%` | `0.229` | `37.5` |
| 300 | `5/16` | `31.3%` | `27.7%` | `0.380` | `50.0` |

| Metric | B3g tutorial demo BC | B3k bridge 25% | B3l bridge 12.5% | Read |
|---|---:|---:|---:|---|
| Selected checkpoint | ep250 | ep150 | ep300 | B3l improves late |
| Selected expanded eval | `7/30` | `5/30` | `6/30` | B3l helps but still trails B3g |
| Selected crystal rate | `23.3%` | `16.7%` | `20.0%` | between B3k and B3g |
| Selected depth | `36.2%` | `21.7%` | `26.0%` | still much shallower than B3g |
| Near-miss <=3 tiles | `40.0%` | `40.0%` | `33.3%` | worse close approach |
| Near-miss <=1.5 tiles | `23.3%` | `20.0%` | `20.0%` | no final-contact gain |
| Mean min target distance | `5.44` tiles | `6.14` tiles | `7.51` tiles | worse approach |
| Close-zone jump rate | `2.0%` | `17.5%` | `14.5%` | bridge action effect remains |
| Close-zone idle/interact rate | `13.5%` | `2.9%` | `3.6%` | still less dithering |
| Stuck-after-close rate | `10.0%` | `0.0%` | `3.3%` | better than B3g |
| Loop-after-close rate | `20.0%` | `20.0%` | `20.0%` | tied |

**Finding:** reducing bridge exposure from `25%` to `12.5%` partially fixed the route
dilution: selected wins improved from `5/30` to `6/30`, source wins reached `5/16`, and
the close-zone action benefits mostly remained. However, approach quality got worse
than B3g (`7.51` mean min target distance vs `5.44`, `33.3% <=3` vs `40.0%`), and the
selected eval still missed B3g's `7/30`.

**Decision:** do not promote B3l over B3g. The bridge lane is useful as an action-shape
teacher, but even at one lane it still does not produce a better route policy from
scratch.

**Next recommendation:** stop testing bridge interleave from scratch. The next
high-signal test is the two-stage version: train B3g tutorial demo BC to preserve route
approach, then apply a short/light bridge fine-tune and select by held-out route eval.
This directly tests whether bridge practice is useful only after the route policy
already reaches the close zone.
