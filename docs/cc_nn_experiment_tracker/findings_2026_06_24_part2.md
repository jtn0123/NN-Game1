# Crystal Caves NN Tracker Archive: Findings 2026-06-24 Part 2

Archived from `CC_NN_EXPERIMENT_TRACKER.md` during cleanup on 2026-06-24.

### 2026-06-24 B3r. Filtered/Weighted Route Demonstrations

**Status:** completed; not promoted.

**Implementation:**

- Added per-demo quality metrics to `collect_scripted_route_demonstrations`: close-zone
  action counts, close-zone jump rate, close-zone idle/interact rate, total
  idle/interact rate, max tile-visit fraction, and `kept_rows` metadata.
- Added `--demo-selection-mode filtered-weighted` for tutorial demo modes. The default
  remains `all`, so older B3g/B3o/B3p commands remain reproducible.
- Added `select_route_demo_trajectories`:
  - keeps all successful `direct` and `recovery` demos;
  - weights clean `direct` demos `2x`;
  - weights `recovery` demos `2x` unless they are in the slowest quartile, where they
    are capped to `1x`;
  - keeps `beam` demos only when they are not in the slowest quartile, enter the
    close-zone, contain close-zone jumping, avoid close-zone idle/interact, and avoid
    heavy tile looping.
- Added `demo_selection_summary.json` and markdown report output so future runs show raw
  source data and the actual selected/weighted training data separately.

**Validation:** Black and Ruff passed; focused status-session tests passed
(`41 passed in 2.57s`); full suite passed (`1027 passed in 22.23s`).

**128-level source probe:** `.Codex/artifacts/cc_b3r_selection_probe`

| Dataset | Wins / levels | Attempts | Unique demos | Weighted demos | Transitions | Close-zone transitions | Variant counts | Excluded |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Raw B3p-style source | 85/128 | 283 | 85 | 85 | 18,238 | 2,136 | `{'beam': 19, 'direct': 35, 'recovery': 31}` | n/a |
| B3r filtered-weighted | 76/128 | 283 | 76 | 123 | 22,816 | 2,844 | `{'beam': 10, 'direct': 70, 'recovery': 43}` weighted | `{'beam_quality_filter': 9}` |

**Training command:**

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
  --demo-selection-mode filtered-weighted \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --save-selected-checkpoint \
  --label tutorial_demo_bc_filtered_weighted_pool512_select30_300
```

**Artifact:** `.Codex/artifacts/cc_sessions/20260624_111158_tutorial_demo_bc_filtered_weighted_pool512_select30_300`

**Source eval history:**

| Episode | Source wins | Crystals | Depth | Ends |
|---:|---:|---:|---:|---|
| after BC | 0/16 | 0.0% | 24.1% | `{'stalled': 16}` |
| 50 | 0/16 | 0.0% | 6.2% | `{'stalled': 10, 'timeout': 6}` |
| 100 | 1/16 | 6.2% | 21.0% | `{'first_crystal_goal': 1, 'stalled': 6, 'timeout': 9}` |
| 150 | 2/16 | 12.5% | 32.1% | `{'first_crystal_goal': 2, 'stalled': 4, 'timeout': 10}` |
| 200 | 0/16 | 0.0% | 34.8% | `{'stalled': 9, 'timeout': 7}` |
| 250 | 1/16 | 6.2% | 40.2% | `{'first_crystal_goal': 1, 'stalled': 5, 'timeout': 10}` |
| 300 | 1/16 | 6.2% | 35.7% | `{'first_crystal_goal': 1, 'stalled': 12, 'timeout': 3}` |

Selected source checkpoint was episode `150`. The selected-only checkpoint was saved at
`.Codex/artifacts/cc_sessions/20260624_111158_tutorial_demo_bc_filtered_weighted_pool512_select30_300/tutorial_demo_bc/models/crystal_caves/tutorial_demo_bc_selected_ep150.pth`
and is `6.1M`. The full run artifact is `8.8M`, confirming that selected-only
checkpoints avoid the previous replay-buffer bloat.

**Selected expanded comparison:**

| Metric | B3g tutorial demo BC | B3o recovery demos | B3p planner demos | B3r filtered-weighted |
|---|---:|---:|---:|---:|
| Demo wins | 35/128 | 66/128 | 85/128 | 85/128 raw, 76 kept |
| Demo train transitions | 4,149 | 14,644 | 18,238 | 22,816 weighted |
| Selected source eval | 2/16 | 2/16 | 3/16 | 2/16 |
| Selected expanded eval | 7/30 | 7/30 | 4/30 | 5/30 |
| Selected crystal rate | 23.3% | 23.3% | 13.3% | 16.7% |
| Selected depth | 36.2% | 44.8% | 36.4% | 33.6% |
| Near-miss <=3 tiles | 40.0% | 43.3% | 33.3% | 30.0% |
| Near-miss <=1.5 tiles | 23.3% | 26.7% | 13.3% | 16.7% |
| Mean min target distance | 5.44 | 5.57 | 5.84 | 6.09 |
| Best target-distance delta | 7.72 | n/a | n/a | 7.07 |
| Final target-distance delta | n/a | n/a | n/a | 3.29 |
| Close-zone jump rate | 2.0% | 3.7% | 0.1% | 3.1% |
| Stuck-after-close | 10.0% | 3.3% | 3.3% | 3.3% |
| Loop-after-close | 20.0% | 33.3% | 23.3% | 16.7% |

**Finding:** B3r did what it was supposed to mechanically: it removed 9 weak beam demos,
weighted direct/recovery demos more heavily, trained much better on the source pool
(`56%` training first-crystal rate at the end), and kept artifacts small. But it did not
solve the held-out route gate. It improved over B3p's failed planner-data run (`5/30`
vs `4/30`) and improved some local behavior metrics, but still trails B3g/B3o's `7/30`
selected result and has worse approach distance than both.

**Decision:** do not promote B3r. Keep the demo-quality metrics and
`--demo-selection-mode filtered-weighted` as useful tooling, but do not treat filtered
scripted demos as the next winning lever.

**Next recommendation:** move to B3s, a conservative DQN/demo regularizer. The repeated
pattern is now clear: better scripted data improves training-source behavior, but the
greedy policy still overfits or picks unsupported actions on held-out caves. The next
smallest algorithmic step is to penalize high Q-values for non-demo actions or add an
advantage/large-margin style constraint during online training, then compare against
B3g/B3o on the same selected-checkpoint surface.

### 2026-06-24 B3s. Conservative Demo-Q Regularizer

**Status:** completed; promoted as the current first-crystal route baseline.

**Why this is next:** B3r improved the data source and training-pool behavior but still
missed the held-out gate. The repeated failure pattern is now that better demos improve
training-source behavior while the greedy policy still chooses unsupported or brittle
actions on held-out caves. B3s adds a conservative Q penalty on demo states:

`temperature * logsumexp(Q(s, all_actions) / temperature) - Q(s, demo_action)`

This is a small CQL-style extension of the existing DQfD margin loss. The aim is to
reduce overconfident non-demo actions without changing final/eval levels.

**Implementation:**

- Added `CRYSTAL_CAVES_DEMO_CONSERVATIVE_WEIGHT` and
  `CRYSTAL_CAVES_DEMO_CONSERVATIVE_TEMPERATURE`.
- Extended the existing demo-action supervision path in `src/ai/agent.py` to compute
  both the large-margin demo loss and optional conservative Q loss from the same demo
  state/action dataset.
- Added `avg_demo_conservative_loss_100` to live/session summaries and live heartbeat
  text.
- Added `--demo-conservative-weight`, `--demo-conservative-temperature`, and a distinct
  `tutorial-demo-conservative` status-session mode.
- Preserved old behavior: conservative weight defaults to `0.0`, and existing B3g/B3o
  commands are unchanged.

**Smoke command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-conservative \
  --episodes 1 \
  --seed 0 \
  --eval-games 1 \
  --selected-eval-games 1 \
  --train-eval-games 1 \
  --eval-every 1 \
  --trace-eval-games 1 \
  --trace-max-steps 80 \
  --trace-sample-every 20 \
  --trace-tail-steps 40 \
  --vec-envs 1 \
  --cave-pool-size 64 \
  --route-demo-levels 16 \
  --route-demo-max-steps 800 \
  --route-demo-variants direct,recovery \
  --demo-selection-mode all \
  --bc-epochs 1 \
  --bc-batch-size 64 \
  --demo-repeat 1 \
  --demo-action-weight 0.03 \
  --demo-action-margin 0.8 \
  --demo-conservative-weight 0.02 \
  --demo-conservative-temperature 1.0 \
  --heartbeat-seconds 10 \
  --log-every 1 \
  --report-seconds 1 \
  --save-selected-checkpoint \
  --label b3s_conservative_smoke
```

**Smoke artifact:** `.Codex/artifacts/cc_sessions/20260624_115807_b3s_conservative_smoke`

**Smoke result:** mechanics passed. Live metrics showed the conservative term active:
`demo 0.414/82%/cql 1.414`. The markdown report recorded
`conservative 0.020 @T 1.00`, and the selected-only checkpoint remained small (`6.1M`).

**Validation so far:** Black and Ruff passed for touched files; focused agent/status
tests passed (`101 passed in 4.53s`).

**Planned full run:** use B3o's better `direct,recovery` source data rather than B3p's
beam data or B3r's filtered-weighted data. This tests the algorithmic regularizer
against the current useful data source:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-conservative \
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
  --demo-selection-mode all \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --demo-action-weight 0.03 \
  --demo-action-margin 0.8 \
  --demo-conservative-weight 0.02 \
  --demo-conservative-temperature 1.0 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --save-selected-checkpoint \
  --label tutorial_demo_conservative_recovery_pool512_select30_300
```

**Promotion rule:** compare against B3g/B3o's `7/30` selected first-crystal result.
Promote only if B3s reaches at least `8/30`, or ties `7/30` while improving at least
two secondary metrics without worsening depth: mean min target distance,
`<=1.5` near-miss rate, close-zone jump rate, stuck-after-close, or loop-after-close.

**Full-run artifact:** `.Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300`

**Full-run result:** promoted. B3s is the first B3-line change to clear the route gate
instead of only tying it.

| Metric | B3g tutorial demo BC | B3o recovery demos | B3r filtered-weighted | B3s conservative demo-Q |
|---|---:|---:|---:|---:|
| Selected expanded eval | `7/30` | `7/30` | `5/30` | **`10/30`** |
| Selected crystal rate | `23.3%` | `23.3%` | `16.7%` | **`33.3%`** |
| Selected depth | `36.2%` | `44.8%` | `33.6%` | **`60.5%`** |
| Source eval at selected ep | n/a | `4/16` | `5/16` | **`5/16`** |
| Near-miss <=3 tiles | `40.0%` | `43.3%` | `30.0%` | **`60.0%`** |
| Near-miss <=1.5 tiles | `23.3%` | `26.7%` | `16.7%` | **`40.0%`** |
| Mean min target distance | `5.44` | `5.57` | `6.09` | **`3.51`** |
| Best target-distance delta | `7.72` | n/a | `7.16` | **`9.65`** |
| Close-zone jump rate | `2.0%` | `3.7%` | `3.1%` | `0.0%` |
| Close-zone idle/interact rate | `13.5%` | `6.8%` | `4.6%` | **`3.6%`** |
| Stuck-after-close rate | `10.0%` | **`3.3%`** | **`3.3%`** | `20.0%` |
| Loop-after-close rate | `20.0%` | `33.3%` | **`16.7%`** | `33.3%` |

**Source checkpoint history:**

| Episode | Source wins | Crystals | Depth | End reasons |
|---:|---:|---:|---:|---|
| 0 | `1/16` | `6.2%` | `27.7%` | `{'first_crystal_goal': 1, 'stalled': 15}` |
| 50 | `3/16` | `18.8%` | `33.0%` | `{'first_crystal_goal': 3, 'stalled': 10, 'timeout': 3}` |
| 100 | `3/16` | `18.8%` | `50.0%` | `{'first_crystal_goal': 3, 'stalled': 12, 'timeout': 1}` |
| 150 | `3/16` | `18.8%` | `65.2%` | `{'first_crystal_goal': 3, 'stalled': 12, 'timeout': 1}` |
| 200 | `4/16` | `25.0%` | `62.0%` | `{'first_crystal_goal': 4, 'stalled': 8, 'timeout': 4}` |
| 250 | `4/16` | `25.0%` | `65.2%` | `{'first_crystal_goal': 4, 'stalled': 11, 'timeout': 1}` |
| 300 | **`5/16`** | **`31.2%`** | `59.8%` | `{'first_crystal_goal': 5, 'stalled': 8, 'timeout': 3}` |

**Training details:** source route demos were unchanged from B3o: `66/128` scripted
wins, `14,644` raw transitions, `58,576` replay pushes with `4x` repeat. After-BC source
eval was only `1/16`, so the gain came during online DQN updates with the conservative
term active, not from BC alone. Final live demo telemetry was
`loss100 0.056`, `action accuracy100 97.2%`, `conservative loss100 0.210`.

**Selected checkpoint:** `.Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300/tutorial_demo_conservative/models/crystal_caves/tutorial_demo_conservative_selected_ep300.pth`

**Validation artifact:** `.Codex/artifacts/cc_sessions/20260624_122423_b3s_selected_ep300_eval60`

**Validation result:** the selected B3s policy held up on a larger held-out sample:
`19/60` first-crystal wins (`31.7%`), `31.7%` crystals, `60.0%` depth, with end reasons
`{'first_crystal_goal': 19, 'stalled': 32, 'timeout': 9}`. Near-miss rollup was
`55.0% <=3 tiles`, `36.7% <=1.5 tiles`, mean min target distance `3.99`, best distance
delta `10.36`, close-zone jump rate `1.5%`, stuck-after-close `16.7%`, loop-after-close
`33.3%`.

**Finding:** conservative demo-Q regularization is the first clearly useful algorithmic
addition in this line. It improves the headline selected result (`7/30 -> 10/30`) and
also improves the steadier route-quality metrics: depth, close approach rate, and mean
minimum target distance. The 60-game checkpoint eval staying near the 30-game result
means this is probably not just sample noise.

**Remaining blocker:** B3s still gets close and then often fails execution. Close-zone
jump rate is near zero (`0.0%` on the selected 30-game near-miss eval, `1.5%` on the
60-game validation), while loop-after-close remains high (`33.3%`). The policy has
learned to approach the target much better than B3g/B3o, but it still lacks reliable
last-few-tiles motor execution.

**Decision:** promote B3s as the current route baseline. Future route experiments should
start from `tutorial-demo-conservative` or re-use its saved checkpoint for evaluation.
Do not use B3r/B3p as defaults; their data-quality tooling is useful, but their selected
policies trail B3s.

**Next recommendation:** B3t should be a narrow close-zone execution improvement on top
of B3s, not another broad demo-source change. The top candidate is conservative +
close-zone jump/action assist, but with a lower weight than failed B3i and evaluated
against B3s's stronger baseline. Promotion should require beating B3s's validated route
signal, not merely matching older B3g/B3o.

### 2026-06-24 B3t. Conservative Route + Low-Weight Close-Zone Assist

**Status:** completed; useful diagnostic/tooling, not promoted over B3s.

**Why this is next:** B3s is now the route baseline, but its remaining failure is
last-few-tiles execution: `55-60%` of games get within `<=3` tiles in validation, while
close-zone jump rate stays near zero and loop-after-close remains high. The earlier B3i
close-zone-only action assist was rejected because it replaced the broad route demo
signal and used a strong `0.12` close-zone weight, which suppressed jumps and hurt
selected wins. B3t keeps B3s intact and adds only a second, low-weight close-zone
action-margin loss.

**Permanent code change:** added an opt-in close-zone demo-action dataset and loss to the
agent, plus a named status-session mode:

- `CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_LOSS`
- `CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_WEIGHT` default `0.03`
- `CRYSTAL_CAVES_CLOSE_ZONE_DEMO_ACTION_BATCH_SIZE`
- `Agent.set_close_zone_demo_action_dataset(...)`
- `tutorial-demo-conservative-close-zone`

This is training-only. Faithful final/eval tutorial caves remain unchanged, and the old
`tutorial-demo-close-zone` mode remains available to reproduce B3i.

**Smoke command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-conservative-close-zone \
  --episodes 1 \
  --seed 0 \
  --eval-games 1 \
  --selected-eval-games 1 \
  --train-eval-games 1 \
  --eval-every 1 \
  --trace-eval-games 1 \
  --trace-max-steps 80 \
  --trace-sample-every 20 \
  --trace-tail-steps 40 \
  --vec-envs 1 \
  --cave-pool-size 64 \
  --route-demo-levels 16 \
  --route-demo-max-steps 800 \
  --route-demo-variants direct,recovery \
  --demo-selection-mode all \
  --bc-epochs 1 \
  --bc-batch-size 64 \
  --demo-repeat 1 \
  --demo-action-weight 0.03 \
  --demo-action-margin 0.8 \
  --demo-conservative-weight 0.02 \
  --demo-conservative-temperature 1.0 \
  --close-zone-extra-action-weight 0.03 \
  --heartbeat-seconds 10 \
  --log-every 1 \
  --report-seconds 1 \
  --save-selected-checkpoint \
  --label b3t_conservative_close_zone_smoke
```

**Smoke artifact:** `.Codex/artifacts/cc_sessions/20260624_125058_b3t_conservative_close_zone_smoke`

**Smoke result:** runner path works. Demo collection found `9/16` scripted wins and
`1,625` transitions. The report shows the intended dual supervision:
`all_success` demo action loss at weight `0.030`, conservative weight `0.020`, plus
`150` close-zone extra transitions at weight `0.030`. Live metrics included
`demo 0.371/83%/cql 1.394 | cz 0.026/100%`.

**Validation so far:** formatting passed, py-compile passed, Ruff passed on touched
files, and focused tests passed (`7 passed`).

**Planned full run:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-conservative-close-zone \
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
  --demo-selection-mode all \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --demo-action-weight 0.03 \
  --demo-action-margin 0.8 \
  --demo-action-batch-size 64 \
  --demo-conservative-weight 0.02 \
  --demo-conservative-temperature 1.0 \
  --close-zone-extra-action-weight 0.03 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --save-selected-checkpoint \
  --label tutorial_demo_conservative_close_zone_pool512_select30_300
```

**Promotion rule:** compare directly against B3s, not B3g. Promote only if selected
eval beats B3s's `10/30`, or ties `10/30` while improving at least two close-zone
execution metrics: `<=1.5` tile near-miss rate, close-zone jump rate,
stuck-after-close, or loop-after-close. If selected wins fall below `10/30`, reject even
if one local behavior metric improves.

**Full-run artifact:** `.Codex/artifacts/cc_sessions/20260624_125257_tutorial_demo_conservative_close_zone_pool512_select30_300`

**Full-run result:** mixed. The selected 30-game eval technically beat B3s by one game,
but not enough to trust without a larger validation pass.

| Metric | B3s conservative demo-Q | B3t conservative + close-zone extra | Read |
|---|---:|---:|---|
| Source selected episode | ep300 | ep300 | both select final weights |
| Source eval at selected ep | `5/16` | **`6/16`** | B3t looked better on source |
| Selected expanded eval | `10/30` | **`11/30`** | B3t +1 on the 30-game gate |
| Selected crystal rate | `33.3%` | **`36.7%`** | B3t +3.4 pp |
| Selected depth | **`60.5%`** | `54.0%` | B3t loses depth |
| Near-miss <=3 tiles | `60.0%` | **`63.3%`** | slightly better approach frequency |
| Near-miss <=1.5 tiles | `40.0%` | `40.0%` | tied final-contact reach |
| Mean min target distance | **`3.51`** | `3.49` | effectively tied on 30-game eval |
| Close-zone jump rate | `0.0%` | **`1.5%`** | local execution nudged upward |
| Close-zone idle/interact rate | `3.6%` | `3.8%` | roughly tied |
| Stuck-after-close rate | `20.0%` | **`13.3%`** | B3t improves stuck-after-close |
| Loop-after-close rate | **`33.3%`** | `40.0%` | B3t worsens close loops |

**Source checkpoint history:**

| Episode | Source wins | Crystals | Depth | End reasons |
|---:|---:|---:|---:|---|
| 0 | `1/16` | `6.2%` | `27.7%` | `{'first_crystal_goal': 1, 'stalled': 15}` |
| 50 | `3/16` | `18.8%` | `46.9%` | `{'first_crystal_goal': 3, 'stalled': 8, 'timeout': 5}` |
| 100 | `3/16` | `18.8%` | `52.7%` | `{'first_crystal_goal': 3, 'stalled': 10, 'timeout': 3}` |
| 150 | `4/16` | `25.0%` | `53.6%` | `{'first_crystal_goal': 4, 'stalled': 9, 'timeout': 3}` |
| 200 | `3/16` | `18.8%` | `52.7%` | `{'first_crystal_goal': 3, 'stalled': 9, 'timeout': 4}` |
| 250 | `4/16` | `25.0%` | `55.8%` | `{'first_crystal_goal': 4, 'stalled': 7, 'timeout': 5}` |
| 300 | **`6/16`** | **`37.5%`** | `54.0%` | `{'first_crystal_goal': 6, 'stalled': 4, 'timeout': 6}` |

**Training details:** B3t used the same route demo source as B3s (`66/128`, `14,644`
raw transitions, `58,576` replay pushes) and added `1,632` close-zone extra action
transitions at weight `0.030`. Final live metrics were `demo loss100 0.046`,
`conservative loss100 0.204`, `close-zone loss100 0.051`, and `close-zone accuracy100
97.8%`.

**Selected checkpoint:** `.Codex/artifacts/cc_sessions/20260624_125257_tutorial_demo_conservative_close_zone_pool512_select30_300/tutorial_demo_conservative_close_zone/models/crystal_caves/tutorial_demo_conservative_close_zone_selected_ep300.pth`

**Validation artifact:** `.Codex/artifacts/cc_sessions/20260624_131951_b3t_selected_ep300_eval60`

**Validation result:** the 30-game gain did not hold. B3t validated at `18/60`
first-crystal wins (`30.0%`), while B3s validated at `19/60` (`31.7%`). B3t validation
also had weaker near-miss/contact profile than B3s: `56.7% <=3 tiles` vs B3s `55.0%`
is roughly tied, but `35.0% <=1.5 tiles` trails B3s `36.7%`, mean min target distance
`4.25` trails B3s `3.99`, close-zone jump `0.8%` trails B3s `1.5%`, and
loop-after-close `38.3%` is worse than B3s `33.3%`.

**Finding:** adding a low-weight close-zone action-margin term can improve the 16-game
source checkpoint and may reduce stuck-after-close on the 30-game selected eval, but it
does not reliably improve the held-out route policy. The larger validation says B3t is
noise-level or slightly worse than B3s.

**Decision:** keep the B3t code path and metrics because the permanent infrastructure is
useful, but do not promote the `0.03` close-zone extra configuration over B3s. B3s
remains the current first-crystal route baseline.

**Next recommendation:** stop using demo labels as the close-zone lever. B3i and B3t both
show that action-margin pressure near the target does not reliably fix final execution.
The next area should be either:

- **B3u: close-zone oracle relabeling / counterfactual labels** from a jump-aware local
  planner, so the close-zone target is "what action reaches/contact the crystal" rather
  than "what the scripted route happened to do"; or
- **B3v: split-policy or option head for final contact**, keeping B3s as the route
  policy and training a tiny final-contact option only for `<=3` tile states.

Prefer B3u first because it is narrower: use the existing successful close-zone states,
replace noisy scripted labels with local oracle labels where possible, and evaluate
whether `<=1.5` contact / jump rate improves without hurting B3s route depth.

### 2026-06-24 B3u. Oracle-Relabeled Close-Zone Assist

**Status:** implemented; full comparable run complete; not promoted.

**Why this is next:** B3i and B3t show the same boundary: applying action-margin loss to
the scripted close-zone action does not reliably improve held-out execution. B3t even
validated slightly below B3s (`18/60` vs `19/60`). The likely issue is label quality:
the scripted route action near the target is often whatever the controller happened to
do, not necessarily the best first action for collecting/contacting the crystal. B3u
keeps B3s's route learner intact and changes only the extra close-zone label source.

**Implementation plan:** for every close-zone state in successful route demonstrations,
deep-copy the game and simulate short local action macros (`RIGHT`, `LEFT`, jump/run
variants, backoff-hop variants). Score each candidate by target completion, crystal
collection/progress gain, best/final target-distance delta, health loss, and distance
left. Store the first action from the best-scoring macro as an oracle close-zone label.

**Permanent code change:**

- Added `close_zone_oracle_action(...)` and local candidate macros.
- `collect_scripted_route_demonstrations(...)` now records
  `oracle_close_zone_trajectories` alongside scripted `close_zone_trajectories`.
- `select_route_demo_trajectories(...)` preserves weighted oracle close-zone labels.
- Added `tutorial-demo-oracle-close-zone` mode.
- Reports now show oracle close-zone transition counts, relabel rate, action counts, and
  whether close-zone extra supervision used `scripted` or `oracle` labels.

This remains training-only. Final/eval caves stay normal held-out tutorial caves.

**Initial smoke command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-oracle-close-zone \
  --episodes 1 \
  --seed 0 \
  --eval-games 1 \
  --selected-eval-games 1 \
  --train-eval-games 1 \
  --eval-every 1 \
  --trace-eval-games 1 \
  --trace-max-steps 80 \
  --trace-sample-every 20 \
  --trace-tail-steps 40 \
  --vec-envs 1 \
  --cave-pool-size 64 \
  --route-demo-levels 16 \
  --route-demo-max-steps 800 \
  --route-demo-variants direct,recovery \
  --demo-selection-mode all \
  --bc-epochs 1 \
  --bc-batch-size 64 \
  --demo-repeat 1 \
  --demo-action-weight 0.03 \
  --demo-action-margin 0.8 \
  --demo-conservative-weight 0.02 \
  --demo-conservative-temperature 1.0 \
  --close-zone-extra-action-weight 0.03 \
  --oracle-close-zone-stride 4 \
  --oracle-close-zone-max-per-trajectory 8 \
  --heartbeat-seconds 10 \
  --log-every 1 \
  --report-seconds 1 \
  --save-selected-checkpoint \
  --label b3u_oracle_close_zone_smoke
```

**Smoke artifact:** `.Codex/artifacts/cc_sessions/20260624_143524_b3u_oracle_close_zone_smoke`

**Smoke result:** runner path works. Demo collection found `9/16` scripted wins and
`1,625` transitions. Oracle close-zone relabeling produced `150` close-zone labels, with
an `83.7%` relabel rate and action counts
`{'LEFT_JUMP': 10, 'LEFT': 29, 'JUMP': 15, 'RIGHT_JUMP': 21, 'IDLE': 56, 'RIGHT': 19}`.
The report shows `Close-zone extra action supervision: oracle`, and live metrics showed
the oracle label task active: `cz 0.992/32%`. Low initial accuracy is expected because
the oracle labels intentionally differ from the scripted close-zone labels.

**Cost finding:** the unbounded first implementation was too slow for a full run. A
full-scale attempt was interrupted after roughly 12 minutes while still repeatedly
deep-copying games during route demo collection. The process was using CPU, so this was
a cost problem rather than a hang. B3u was made bounded and opt-in: oracle close-zone
labels are only computed when the oracle label source is selected, default to every
4th close-zone state, and cap at 8 labels per successful trajectory.

**Bounded smoke artifact:** `.Codex/artifacts/cc_sessions/20260624_145330_b3u_oracle_close_zone_bounded_smoke`

**Bounded smoke result:** setup is now fast enough for a full run. The smoke produced
`39` oracle close-zone labels, with a `79.8%` relabel rate and action counts
`{'LEFT_JUMP': 3, 'LEFT': 11, 'IDLE': 9, 'RIGHT': 9, 'RIGHT_JUMP': 4, 'JUMP': 3}`.
The report shows `Close-zone extra action supervision: oracle`, `39 active transitions`,
and close-zone action learning was active (`avg close-zone demo loss 0.899`, accuracy
`43.3%` after the tiny one-episode smoke).

**Validation so far:** formatting passed, py-compile passed, Ruff passed on touched
files, focused status-session tests passed (`5 passed in 4.10s` after the bounded fix),
and the full suite passed (`1032 passed in 24.18s`).

**Full run command:** same comparable B3s/B3t budget, with bounded oracle labeling:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-oracle-close-zone \
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
  --demo-selection-mode all \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --demo-action-weight 0.03 \
  --demo-action-margin 0.8 \
  --demo-action-batch-size 64 \
  --demo-conservative-weight 0.02 \
  --demo-conservative-temperature 1.0 \
  --close-zone-extra-action-weight 0.03 \
  --oracle-close-zone-stride 4 \
  --oracle-close-zone-max-per-trajectory 8 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --save-selected-checkpoint \
  --label tutorial_demo_oracle_close_zone_pool512_select30_300
```

**Promotion rule:** compare against B3s first, B3t second. Promote only if selected eval
beats B3s's `10/30` and the 60-game validation reaches at least B3s's `19/60`, or if
selected eval ties while improving `<=1.5` contact, close-zone jump rate, and
loop/stuck-after-close without losing B3s's `60%` depth profile.

**Full-run artifact:** `.Codex/artifacts/cc_sessions/20260624_145614_tutorial_demo_oracle_close_zone_pool512_select30_300`

**Full-run result:** useful diagnostic, but not a promotion over B3s.

| Metric | B3s conservative demo-Q | B3t scripted close-zone extra | B3u oracle close-zone extra | Read |
|---|---:|---:|---:|---|
| Source selected episode | ep300 | ep300 | ep250 | B3u peaked earlier |
| Source eval at selected ep | `5/16` | **`6/16`** | `5/16` | B3u ties B3s, trails B3t |
| Selected expanded eval | `10/30` | **`11/30`** | `10/30` | B3u ties B3s |
| Selected crystal rate | `33.3%` | **`36.7%`** | `33.3%` | no B3u gain |
| Selected depth | **`60.5%`** | `54.0%` | `55.2%` | B3u gives back route depth |
| Near-miss <=3 tiles | `60.0%` | **`63.3%`** | `60.0%` | B3u ties B3s |
| Near-miss <=1.5 tiles | `40.0%` | `40.0%` | `40.0%` | tied |
| Mean min target distance | `3.51` | **`3.49`** | `3.45` | effectively tied |
| Close-zone jump rate | `0.0%` | `1.5%` | **`2.9%`** | B3u changes local action shape |
| Close-zone idle/interact rate | **`3.6%`** | `3.8%` | `10.1%` | B3u adds more idle/interact |
| Stuck-after-close rate | `20.0%` | **`13.3%`** | `23.3%` | B3u worsens stuck-after-close |
| Loop-after-close rate | **`33.3%`** | `40.0%` | `40.0%` | B3u worsens vs B3s |

**Source checkpoint history:**

| Episode | Source wins | Crystals | Depth | End reasons |
|---:|---:|---:|---:|---|
| 0 | `1/16` | `6.2%` | `27.7%` | `{'stalled': 15, 'first_crystal_goal': 1}` |
| 50 | `0/16` | `0.0%` | `33.0%` | `{'stalled': 15, 'timeout': 1}` |
| 100 | `2/16` | `12.5%` | `52.0%` | `{'stalled': 12, 'first_crystal_goal': 2, 'timeout': 2}` |
| 150 | `3/16` | `18.8%` | `52.0%` | `{'stalled': 9, 'first_crystal_goal': 3, 'timeout': 4}` |
| 200 | `3/16` | `18.8%` | `52.0%` | `{'stalled': 10, 'first_crystal_goal': 3, 'timeout': 3}` |
| 250 | **`5/16`** | **`31.2%`** | `52.0%` | `{'first_crystal_goal': 5, 'stalled': 9, 'timeout': 2}` |
| 300 | `4/16` | `25.0%` | `52.0%` | `{'first_crystal_goal': 4, 'stalled': 8, 'timeout': 4}` |

**Training details:** B3u used the same route demo source as B3s/B3t (`66/128`,
`14,644` raw transitions, `58,576` replay pushes) and added `304` bounded oracle
close-zone labels at weight `0.030`. The oracle labels differed from the scripted
close-zone action `80.8%` of the time, so this was a real label-source change rather
than a no-op. Online learning solved the auxiliary target well (`cz` accuracy reached
about `85%`), while the main route policy did not beat B3s.

**Selected checkpoint:** `.Codex/artifacts/cc_sessions/20260624_145614_tutorial_demo_oracle_close_zone_pool512_select30_300/tutorial_demo_oracle_close_zone/models/crystal_caves/tutorial_demo_oracle_close_zone_selected_ep250.pth`

**Decision:** do not run the 60-game promotion validation and do not promote B3u over
B3s. It failed the tie-breaker: B3u tied B3s on `10/30` and `40.0% <=1.5 tiles`, but
lost route depth (`55.2%` vs `60.5%`) and worsened idle/interact, stuck-after-close,
and loop-after-close. The code path is worth keeping because it gives a controlled
counterfactual-label mechanism and revealed that the network can learn these labels;
the problem is not label learnability.

**Finding:** close-zone action supervision is now unlikely to be the next high-leverage
lever by itself. Scripted close-zone labels (B3t) and oracle close-zone labels (B3u)
both learn internally, but neither survives the larger baseline comparison. The current
route learner still needs better state distribution / curriculum pressure, not just
more loss on successful-demo close-zone states.

**Next recommendation:** test a narrow DAgger-style route correction pass on top of B3s:
roll out the current B3s-like policy on held-out/train-pool caves, collect states where
it stalls/loops or enters close range without collecting, label those actual visited
states with the existing route controller/oracle, then fine-tune or train with a small
aggregated correction set. This is preferable to another close-zone-only loss because
B3u shows labels from successful scripted states do not match the failing state
distribution well enough.
