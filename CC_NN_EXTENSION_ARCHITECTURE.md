# Crystal Caves NN Extension Architecture

Last updated: 2026-06-25

This repo now has enough Crystal Caves NN experiments that adding one more idea should
follow a repeatable path instead of adding another one-off branch in the agent or runner.
The goal is reliability: future neural-network expansion should be easy to test, compare,
and either promote or archive.

## Current Promoted Baseline

The active comparison point is still **B3s conservative demo-Q**:

- Recipe key: `b3s_conservative_demo_q`
- Mode: `tutorial-demo-conservative`
- Code source of truth: `experiments/cc_status/recipes.py`
- Artifact: `.Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300`
- Result: `10/30` selected first-crystal wins, then `19/60` expanded validation

Future route-control experiments should compare against this recipe first. A new method
should not be promoted just because it beats old zero-win baselines.

The active learned contact-adapter comparison point is now **B21 stable-label
offline/head-only contact selector**:

- Selected artifact:
  `.Codex/artifacts/cc_sessions/20260625_181012_b21_stable_contact_head_offline_conf075_eval30`
- Validation artifact:
  `.Codex/artifacts/cc_sessions/20260625_181129_b21_stable_contact_head_offline_conf075_val60`
- Result: `13/30` selected first-crystal wins, then `23/60` expanded validation.
- Use B21, not B15, when comparing B3s plus a learned NN contact head with no
  oracle/planner at eval time.
- Seed-1 robustness evidence is positive but modest: B21 improved matched seed-1
  validation from B3s `16/60` to `18/60`, while lowering route depth.
- B24 per-action `LEFT_JUMP` gating is available as selector infrastructure and slightly
  improved matched seed-1 validation (`19/60`), but it is not promoted over B21 because it
  did not recover route depth.

## Extension Points

### 1. Optional NN Losses

Use `src/ai/extension_contracts.py`:

- `AuxiliaryLossContribution`: one differentiable loss term with a weight.
- `AuxiliaryMetric`: one scalar to append to an agent metric deque.
- `AuxiliaryLossProvider`: protocol for future modules that compute contributions.

Existing route auxiliary, demo action, conservative demo-Q, and close-zone losses now
flow through `_auxiliary_loss_contributions(...)` in `src/ai/agent_experiments.py`.
`Agent.learn(...)` adds the returned contributions generically and records declared
metrics. If a method reports a metric history that does not exist, the agent raises
instead of silently dropping it.

External experiment modules can now register provider objects with
`Agent.register_auxiliary_loss_provider(provider)`. A provider must expose
`auxiliary_loss_contributions(states)` and return `AuxiliaryLossContribution` objects.
Use this for new standalone NN experiments before growing `AgentExperimentMixin` again.

When adding a new NN method:

1. Add its config toggles as experiment-only options unless it is already promoted.
2. Add one contribution in the mixin or a small provider module.
3. Give every emitted metric a named deque on `Agent`.
4. Add tests for contribution names, weights, finite losses, and metric histories.
5. Add the metrics to status-session reports before running long sessions.

### 2. Named Run Recipes

Use `experiments/cc_status/recipes.py`.

Recipes turn important command lines into testable data. The registry currently contains:

- `b3s_conservative_smoke`: one-episode mechanics check.
- `b3s_conservative_demo_q`: promoted full B3s comparison run.
- `b3s_final_contact_option_smoke`: 4-game artifact-validation smoke for the eval-only
  option path; requires `--checkpoint`.
- `b3s_final_contact_option_eval`: 30-game selected comparison for the eval-only option
  path; requires `--checkpoint`.
- `b3s_correction_collect`: policy-visited correction dataset collection; requires
  `--checkpoint`.
- `b4_contact_only_correction_collect`: close-zone-only policy-visited correction dataset
  collection; requires `--checkpoint`.
- `b3s_correction_finetune`: correction action-loss fine-tune; requires `--checkpoint`
  and `--correction-dataset`.
- `b5_anchored_contact_correction`: correction fine-tune with policy-anchor loss;
  requires `--checkpoint` and `--correction-dataset`.
- `b6_contact_interleaved`: archived fixed contact-level interleave; requires
  `--checkpoint`.
- `b7_contact_pool_interleaved`: archived generated contact-pool interleave; requires
  `--checkpoint`.
- `b8_history_state_smoke`: one-episode mechanics smoke for the opt-in history-state
  architecture.
- `b8_history_state_conservative`: archived full comparison for the opt-in history-state
  architecture.
- `b9_c51_distributional_smoke`: one-episode mechanics smoke for the opt-in C51
  distributional DQN head/loss.
- `b9_c51_distributional`: archived full comparison for the opt-in C51 distributional
  DQN head/loss.
- `b10_final_contact_advantage_gate_smoke`: smoke for the eval-only
  policy-advantage-gated final-contact controller; requires `--checkpoint`.
- `b10_final_contact_advantage_gate_eval`: promoted eval-time controller comparison for
  the policy-advantage-gated final-contact controller; requires `--checkpoint`.
- `b11_advantage_gate_correction_collect`: archived dataset collection for B10
  gate-accepted correction labels; requires `--checkpoint`.
- `b11_advantage_gate_correction_finetune`: archived low-weight correction fine-tune from
  B10 gate-accepted labels; requires `--checkpoint` and `--correction-dataset`.
- `b13_route_masked_correction_finetune`: route-masked low-weight correction fine-tune
  from B10 gate-accepted labels; requires `--checkpoint` and `--correction-dataset`.
- `b14_contact_head_finetune`: detached close-zone action-head fine-tune from B10
  gate-accepted labels; requires `--checkpoint` and `--correction-dataset`.
- `b15_contact_head_offline`: promoted offline/head-only close-zone action-head adapter
  from B10 gate-accepted labels; requires `--checkpoint` and `--correction-dataset`.

Recipes are now executable from the status-session wrapper:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python experiments/cc_status_session.py list-recipes
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_conservative_smoke
```

Any extra CLI flags after the recipe key are appended after the recipe defaults, so they
can override normal scalar options:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_conservative_smoke --episodes 2 --label smoke_override
```

When adding a new method, add two recipes:

1. A smoke recipe with tiny budgets that proves the path runs and metrics are live.
2. A comparison recipe that matches B3s's evaluation shape unless there is a clear reason
   to change it.

Keep promotion rules in the recipe. For route methods, the default bar is: beat B3s's
`10/30` selected eval and confirm against B3s's `19/60` expanded validation, or tie wins
while materially improving near-target, close-zone jump/contact, and loop metrics.

### 3. State Architecture Extensions

Use `src/game/crystal_caves.py` plus status-session config helpers for state-shape
experiments. Keep these opt-in because any state-size change makes old checkpoints
incompatible.

Current state extension:

- `CRYSTAL_CAVES_HISTORY_STATE=False` by default.
- `CRYSTAL_CAVES_HISTORY_STEPS=4` when enabled by recipe/CLI.
- Default B3s state remains `295` features.
- B8 history state appends `4 * 7 = 28` metadata scalars for recent action categories
  and normalized target-approach delta, producing a `323`-feature state.
- The status-session CLI exposes this as `--history-state --history-steps 4`.
- `reports.config_snapshot(...)` records `history_state` and `history_steps`.

Use B8 as a fresh architecture run. Do not restore B3s checkpoints into it unless a
checkpoint-conversion path is deliberately built and tested.

Smoke evidence:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260625_085713_b8_history_state_smoke`
- Validation: `ok`.
- Config/state evidence: `history_state=true`, `history_steps=4`, `state_size=323`.
- Demo collection/BC, selected checkpoint save/eval, scorecard, live metrics, and report
  generation all completed.
- Performance evidence: none; the smoke had only `1` eval game.

Full comparison evidence:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260625_085924_b8_history_state_conservative_pool512_select30_300`
- Validation: `ok`.
- Selected checkpoint eval: `10/30` first-crystal wins, `33.3%` crystals, `59.0%`
  depth.
- Route/contact scorecard: `1.757`, below B3s `1.821`.
- `compare-artifact`: `REGRESS`; B8 tied B3s selected wins but improved only mean
  minimum target distance while regressing depth and route/contact score.
- Decision: keep the optional history-state plumbing, but do not promote B8 and do not
  spend the next run sweeping history length.

### 4. Algorithm/Value-Head Extensions

Use `config.NetworkSettings`, `src/ai/network.py`, and `src/ai/agent.py` for value-head
or loss-family experiments. Keep these opt-in unless they are promoted, because they can
change checkpoint semantics and training loss behavior even when the public state/action
shape stays the same.

Current algorithm extension:

- `USE_DISTRIBUTIONAL_DQN=False` by default.
- `C51_NUM_ATOMS=51`, `C51_V_MIN=-20.0`, and `C51_V_MAX=120.0` when enabled.
- `DQN`, `DuelingDQN`, and `SpatialDQN` keep `forward()` returning expected Q-values.
- The networks expose `distributional_logits(...)` and `distributional_probs(...)` for
  the agent loss path.
- `Agent` projects n-step targets onto the C51 support and uses per-sample cross-entropy
  loss when the flag is enabled.
- The status-session CLI exposes this as
  `--distributional-dqn --c51-atoms 51 --c51-v-min -20 --c51-v-max 120`.
- `reports.config_snapshot(...)` records the C51 settings.

Smoke evidence:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260625_113254_b9_c51_distributional_smoke`
- Validation: `ok`.
- Config evidence: `use_distributional_dqn=true`, `c51_num_atoms=51`,
  `c51_v_min=-20.0`, `c51_v_max=120.0`.
- Demo collection/BC, selected checkpoint save/eval, scorecard, live metrics, and report
  generation all completed.
- Performance evidence: none; the smoke had only `1` eval game.

Full comparison evidence:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260625_113409_b9_c51_distributional_pool512_select30_300`
- Validation: `ok`.
- Selected checkpoint eval: `10/30` first-crystal wins, `33.3%` crystals, `59.8%`
  depth.
- Route/contact scorecard: `1.714`, below B3s `1.821`.
- `compare-artifact`: `REGRESS`; B9 tied B3s selected wins/crystals but regressed depth,
  loop-after-close, and route/contact score.
- Decision: keep the optional C51 plumbing, but do not promote B9 and do not spend the
  next run sweeping C51 atoms/support ranges.

### 5. Action-Selection/Controller Extensions

Use `experiments/cc_status/runs_transfer.py` and `experiments/cc_status/demo_planners.py`
for eval-only controllers. Keep these diagnostic unless they pass the same selected and
expanded validation gates as trained policies.

Current controller extension:

- `close_zone_sequence_score(...)` scores one local macro on a copied game state.
- `final_contact_option_action(...)` can enable a policy-advantage gate:
  `gate_policy_advantage=True`, `min_option_advantage=250.0`.
- The gate simulates the local option and the NN policy rollout from the same state, then
  lets the option take over only when the option's score advantage clears the threshold.
- Reports include gate evaluations, rejections, rejection rate, accepted/rejected mean
  advantage, and normal final-contact action counts.

B10 evidence:

- Smoke artifact:
  `.Codex/artifacts/cc_sessions/20260625_121800_b10_final_contact_advantage_gate_smoke/20260625_121346_b10_final_contact_advantage_gate_smoke`
- Selected artifact:
  `.Codex/artifacts/cc_sessions/20260625_122000_b10_final_contact_advantage_gate_eval30/20260625_121437_b10_final_contact_advantage_gate_eval30`
- Validation artifact:
  `.Codex/artifacts/cc_sessions/20260625_122500_b10_final_contact_advantage_gate_val60/20260625_121631_b10_final_contact_advantage_gate_val60`
- Selected result: `15/30` first-crystal wins, `50.0%` crystals, `53.6%` depth,
  route/contact score `2.702`.
- Validation result: `30/60` first-crystal wins, `50.0%` crystals, `54.3%` depth,
  route/contact score `2.685`.
- Gate behavior: option takeover was only about `0.8%` of selected/validation eval steps.
- Metric audit: selected non-success depth `72.4%`; validation non-success depth `69.5%`.
  This showed raw depth was biased downward by earlier successful first-crystal endings.
- Updated `compare-artifact --validation`: `PROMOTE`; raw depth regression stays visible,
  but outcome-conditioned non-success route depth clears the B3s guardrail.
- Decision: promote B10 as the best eval-time controller baseline. Keep B3s as the
  pure-NN training baseline until a learned checkpoint internalizes the same contact
  improvement without the eval-time option.

B11 transfer evidence:

- Dataset artifact:
  `.Codex/artifacts/cc_sessions/20260625_125000_b11_advantage_gate_correction_collect/20260625_123917_b11_advantage_gate_correction_collect`
- Fine-tune artifact:
  `.Codex/artifacts/cc_sessions/20260625_125500_b11_advantage_gate_correction_finetune/20260625_124115_b11_advantage_gate_correction_finetune_w001_a005_150`
- Dataset: `162` kept labels from `625` close-zone candidates, `80.2%` policy/label
  disagreement, `67.7%` gate rejection.
- Fine-tune: correction accuracy reached `78.8%`, but final held-out eval was only
  `4/16` first-crystal wins, `25.0%` crystals, `44.2%` depth.
- Route/contact score: `1.223`, below B3s `1.821` and B10 selected `2.702`.
- Decision: do not continue correction/anchor weight sweeps from this dataset by default.
  The labels can be learned, but low-weight action-margin transfer does not preserve the
  route policy.

B13 route-masked transfer evidence:

- Smoke artifact:
  `.Codex/artifacts/cc_sessions/20260625_134345_b13_route_masked_smoke`
- Fine-tune artifact:
  `.Codex/artifacts/cc_sessions/20260625_134602_b13_route_masked_correction_finetune_w001_a010_150`
- Mechanism: same B11 `162`-state B10-accepted label dataset, correction action weight
  `0.001`, policy anchor weight `0.10`, but the frozen-policy anchor only applies when
  the active target is at least `3.0` tiles away.
- Fine-tune: correction accuracy reached `80.4%`; final held-out eval was `4/16`
  first-crystal wins, `25.0%` crystals, `60.3%` depth.
- Metric audit: B13 non-success depth was `64.9%`, recovering most of B3s's route depth
  and improving over B11's `49.4%` non-success depth.
- Route/contact score: `1.228`, still below B3s `1.821` because stuck-after-close was
  `43.8%` and loop-after-close was `56.2%`.
- Decision: keep the route-mask anchor support, but do not promote B13. The next
  higher-value training architecture is a separate learned close-zone action head or
  gated adapter instead of more scalar correction/anchor sweeps.

B14 detached contact-head evidence:

- Smoke artifact:
  `.Codex/artifacts/cc_sessions/20260625_143831_b14_contact_head_smoke`
- Fine-tune artifact:
  `.Codex/artifacts/cc_sessions/20260625_144008_b14_contact_head_finetune_w002_150`
- Mechanism: add opt-in `CRYSTAL_CAVES_CONTACT_ACTION_HEAD`, train a detached
  `SpatialDQN.contact_action_logits(...)` head from the same B11/B10-accepted labels, and
  use it only inside the `3.0` tile target-distance mask during selector eval.
- Fine-tune: head accuracy reached `80.2%`, but selector eval regressed to `1/16`
  first-crystal wins, `6.2%` crystals, and `37.5%` depth.
- Selector stats: `13255` head actions, `32.1%` head action rate, mean confidence `0.703`;
  head action mix was dominated by `JUMP 11139`.
- Route/contact score: `0.344`, below B13 `1.228` and B3s `1.821`; compare vs B13
  returned `REGRESS`.
- Decision: keep the detached-head infrastructure, but reject this online RL fine-tune
  recipe. The next version should freeze B3s and train the head offline with balanced
  batches plus a confidence gate before overriding the base policy.

B15 offline/head-only contact-head evidence:

- Smoke artifact:
  `.Codex/artifacts/cc_sessions/20260625_150206_b15_contact_head_offline_smoke`
- Selected artifact:
  `.Codex/artifacts/cc_sessions/20260625_150413_b15_contact_head_offline_balanced_conf075_500_eval30`
- Validation artifact:
  `.Codex/artifacts/cc_sessions/20260625_150521_b15_contact_head_offline_balanced_conf075_500_val60`
- Mechanism: restore B3s, enable the detached contact head, train only
  `contact_action_head.*` offline for `500` balanced supervised steps, and use the head
  inside the `3.0` tile target-distance mask only when confidence is at least `0.75`.
- Route weights stayed fixed (`route delta 0.00e+00`); head delta was `3.36e-01`.
- Selected result: `13/30` first-crystal wins, `43.3%` crystals, `56.9%` depth,
  route/contact score `2.302`.
- Validation result: `22/60` first-crystal wins, `36.7%` crystals, `57.0%` depth.
- Metric audit: validation non-success depth was `69.4%`, clearing the route guardrail
  and nearly matching B3s validation non-success depth `69.9%`.
- Selector stayed narrow: selected head action rate `0.5%`, validation head action rate
  `0.6%`, with thousands of low-confidence fallbacks to B3s.
- `compare-artifact --validation`: `PROMOTE`.
- Decision: promote B15 as the best learned contact-head adapter baseline. Keep B10 as
  the best overall eval-time controller outcome because B10 validation wins remain higher
  (`30/60` vs B15 `22/60`).
- Second-seed check: same-seed B3s control was `8/30`, `26.7%` crystals, `54.3%` depth;
  B15 seed 1 was also `8/30`, `26.7%` crystals, `52.9%` depth. This is neutral, not a
  robustness promotion. Do not run seed-1 expanded validation until data/gating improves
  the selected result.

B16 class-aware jump confidence evidence:

- Artifacts:
  `.Codex/artifacts/cc_sessions/20260625_164745_b16_contact_head_jump_conf085_seed1_eval30`
  and
  `.Codex/artifacts/cc_sessions/20260625_164903_b16_contact_head_jump_conf085_seed0_eval30`.
- Mechanism: keep B15's frozen B3s/offline balanced head recipe, but require `0.85`
  confidence before jump-variant head actions can override B3s; non-jump actions keep
  the base `0.75` confidence threshold.
- Seed 1 improved from B3s/B15 `8/30` to `9/30` and score `1.526`, with only `6`
  accepted head actions and `1737` confidence rejects.
- Seed 0 regressed versus B15 from `13/30` to `12/30` and score `2.302` to `2.193`,
  though it still beat B3s.
- Decision: keep the recipe and selector support, but do not promote B16 over B15 or run
  expanded validation yet. The next architecture/data step should collect a larger,
  stratified contact-label set and calibrate class thresholds on held-out labels. B17
  completed the hard-seed collection half of that plan.

B17 hard-seed contact-label evidence:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260625_165300_b17_advantage_gate_correction_collect_seed1`.
- Mechanism: dataset-only seed-1 collection from the B3s checkpoint using the B10
  policy-advantage-gated final-contact labeler.
- Kept labels: `222` from `60` games, versus B11's `162` from `60` seed-0 games.
- Label action mix: `JUMP 56`, `LEFT 33`, `LEFT_JUMP 51`, `RIGHT 22`,
  `RIGHT_JUMP 60`; B11 had only `LEFT_JUMP 18` and `RIGHT_JUMP 21`.
- Disagreement/gate: `83.8%` policy/label disagreement and `82.4%` advantage-gate
  rejection.
- Decision: keep the dataset. Next step is B18 combine/calibration: combine B11+B17,
  split calibration labels, report per-class accuracy/confidence/coverage, and then
  decide whether a new offline contact-head run is justified.

B18 combined contact-head calibration evidence:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260625_173639_b18_contact_head_combined_calibration_b11_b17`.
- Mechanism: combine B11+B17 into `384` labels, split `288` train / `96` held-out
  calibration labels, fit only the detached contact head, and evaluate held-out label
  accuracy before any game eval.
- Route weights stayed fixed (`route delta 0.00e+00`).
- Train-label accuracy was `75.7%`, but held-out calibration accuracy was only `58.3%`
  against the `70%` gate.
- Weak classes: `LEFT 23.5%`, `LEFT_JUMP 52.9%`; stronger classes: `JUMP 65.6%`,
  `RIGHT 60.0%`, `RIGHT_JUMP 80.0%`.
- Decision: hold. Do not launch a selected contact-head eval from this combined head yet.
  B19 completed that audit, so do not launch another selected contact-head eval from raw
  B11+B17 labels yet.

B19 contact-label quality evidence:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260625_180033_b19_contact_label_quality_audit_b11_b17`.
- Dataset: the same combined B11+B17 `384` labels.
- No rounded duplicate-state conflicts at `3` decimals, so the issue is not obvious exact
  duplicate states with different labels.
- Semantic ambiguity: `36` local-geometry groups covering `295/384` labels had multiple
  one-step labels.
- Adjacent-frame label flips: `76/313` checked adjacent pairs (`24.3%`) changed label while
  targeting the same tile.
- Direction mismatch heuristic: `87/242` horizontal labels (`36.0%`) moved opposite the
  target tile x-direction. This is only a heuristic because platformer jumps sometimes need
  setup movement, but the rate is high enough to explain weak one-step supervised
  calibration.
- Decision: raw B11+B17 labels are too phase-dependent for another direct contact-head
  selected eval. Next is B20 stable-label filtering: drop high-ambiguity semantic buckets
  and adjacent-flip rows, then rerun calibration before any game eval.

B20 stable-label filter and calibration evidence:

- Filter artifact:
  `.Codex/artifacts/cc_sessions/20260625_180829_b20_stable_contact_label_filter_b11_b17`.
- Filter rule: keep semantic groups with at least `0.67` majority label share, keep only
  majority-label rows, and drop adjacent label-flip rows.
- Retained `117/384` labels with all five classes represented; smallest class was
  `RIGHT` with `13`.
- Calibration artifact:
  `.Codex/artifacts/cc_sessions/20260625_180944_b20_stable_contact_head_calibration_b11_b17`.
- Held-out label accuracy improved from B18 `58.3%` to `82.8%`, clearing the `70%` gate.
- Route weights stayed fixed (`route delta 0.00e+00`).

B21 stable-label adapter evidence:

- Selected artifact:
  `.Codex/artifacts/cc_sessions/20260625_181012_b21_stable_contact_head_offline_conf075_eval30`.
- Validation artifact:
  `.Codex/artifacts/cc_sessions/20260625_181129_b21_stable_contact_head_offline_conf075_val60`.
- Selected result: `13/30` first-crystal wins, `43.3%` crystals, `56.7%` depth,
  route/contact score `2.333`.
- Validation result: `23/60` first-crystal wins, `38.3%` crystals, `56.9%` raw depth,
  route/contact score `2.036`.
- Metric audit: raw depth missed the guardrail by about `0.001`, but non-success route
  depth was `70.8%`, clearing B3s's non-success route guardrail.
- Promotion fix: `compare-artifact --validation` now reads artifact-level per-game rows
  for success/non-success depth. Corrected B21 decision is `PROMOTE`.
- Seed-1 selected artifact:
  `.Codex/artifacts/cc_sessions/20260625_182515_b21_stable_contact_head_offline_conf075_seed1_eval30`.
- Seed-1 validation artifact:
  `.Codex/artifacts/cc_sessions/20260625_182705_b21_stable_contact_head_offline_conf075_seed1_val60`.
- Matched B3s seed-1 validation control:
  `.Codex/artifacts/cc_sessions/20260625_182902_b3s_selected_seed1_val60_control`.
- Seed-1 validation result: B21 `18/60`, `30.0%` crystals, `53.0%` depth versus B3s
  `16/60`, `26.7%` crystals, `55.5%` depth.
- Seed-1 risk: non-success route depth dropped from B3s `64.4%` to B21 `62.1%`, and
  accepted contact-head actions were dominated by `LEFT_JUMP 2265`.
- Broad jump-confidence follow-ups were rejected:
  `.Codex/artifacts/cc_sessions/20260625_183234_b22_stable_contact_head_jump_conf090_seed1_eval30`
  and
  `.Codex/artifacts/cc_sessions/20260625_183355_b23_stable_contact_head_jump_conf085_seed1_eval30`
  both fell to `9/30` versus B21 seed-1 selected `10/30`.
- B24 per-action `LEFT_JUMP:0.90` gating:
  `.Codex/artifacts/cc_sessions/20260625_184641_b24_stable_contact_head_left_jump_conf090_seed1_eval30`,
  `.Codex/artifacts/cc_sessions/20260625_184808_b24_stable_contact_head_left_jump_conf090_seed1_val60`,
  and
  `.Codex/artifacts/cc_sessions/20260625_185016_b24_stable_contact_head_left_jump_conf090_seed0_eval30`.
- B24 preserved B21 seed-1 selected wins (`10/30`) and improved matched seed-1 validation
  from `18/60` to `19/60`, while reducing accepted `LEFT_JUMP` actions from `2265` to
  `312`. It did not recover route depth (`61.7%` non-success depth vs B21 `62.1%` and
  B3s `64.4%`), so it is useful selector infrastructure but not a promoted baseline.
- B25-B30 tested policy-visited label aggregation. B25 collected `225` B24-policy-visited
  labels, B26 filtered B20+B25 to `184` stable labels, and B27 improved held-out label
  calibration to `85.1%`. Direct game eval still regressed: B28/B29/B30 all landed at
  `9/30` on seed 1 versus B24's `10/30`. Treat B25/B26 as diagnostic data, not as a flat
  positive imitation dataset.
- Decision: promote B21 as the current learned contact-head adapter baseline. Keep B24's
  per-action gates available, but do not promote B24 or B26-based heads. The next
  improvement lane should be source-aware/phase-aware use of policy-visited labels, not
  more confidence-threshold sweeps or direct aggregate-label head training.

### 6. Documentation

Update these files while working:

- `CC_NN_EXPERIMENT_TRACKER.md` for active run decisions and outcomes.
- `CC_NN_CLEANUP_AUDIT.md` when deciding what to keep, archive, or delete.
- This file when the extension workflow itself changes.

Rejected ideas should stay documented but should not stay as default config paths.

### 7. Artifact Validation

Status-session runs validate artifacts by default after writing `summary.json` and
`report.md`. The validator writes `artifact_validation.json` and fails the process if
required evidence is missing.

Validation currently checks:

- root `summary.json` and `report.md`;
- at least one run object with label, timing, config, and eval history;
- per-run artifact folder;
- `live_metrics.json` and `live_metrics.jsonl` when heartbeat reporting is enabled;
- `final_eval` for full tutorial-style runs or `drill_eval` for drill runs;
- selected-checkpoint eval, failure diagnostics, near-miss eval, and checkpoint existence
  when selected checkpoint evidence is declared.

Use `--no-artifact-validation` only for local debugging, not for promotable runs.

### 6. Promotion Gate

Use `experiments/cc_status/promotion.py` before replacing the active route-control
baseline. The public command is:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python experiments/cc_status_session.py compare-artifact <candidate-artifact> [--validation <expanded-validation-artifact>] [--json]
```

The default baseline is the frozen B3s reference (`10/30` selected eval, `19/60`
expanded validation). Decisions mean:

- `PROMOTE`: selected eval and expanded validation clear B3s.
- `HOLD`: the run is promising or useful, but is under-sampled or missing expanded
  validation.
- `REGRESS`: the run trails B3s or ties wins without enough support-metric improvement.

The gate compares win rates, not raw wins, so larger validation samples do not look
better merely because they ran more games. It also keeps support improvements and
regressions visible for crystals, depth, near-miss distance, close-zone jumps, stuck
rate, and loop rate.

Use `metric-audit` when a first-crystal proxy run improves wins but appears to lose raw
mean depth:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python experiments/cc_status_session.py metric-audit <artifact-or-summary>
```

The audit splits depth by successful versus non-successful end states. Promotion
snapshots now include non-success depth, and the validation gate can use it as the
route-preservation check when raw depth is pulled down by more episodes ending
successfully early. Raw depth regression is still reported as a support regression.

### 7. Policy-Visited Correction Data

Use `experiments/cc_status/corrections.py` to collect DAgger-style correction labels
from states the learned policy actually visits. The public command is:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_correction_collect \
  --checkpoint <selected-checkpoint.pth>
```

The recipe requires `--checkpoint` explicitly, so it cannot be launched accidentally
without a selected B3s-style weight snapshot.

This writes:

- `correction_examples.npz`: `states`, correction `actions`, policy actions, and trigger
  masks.
- `correction_examples.jsonl`: readable per-state metadata with objective, Q summary,
  trigger reasons, policy action, and correction label.
- correction summary JSON wired into `summary.json`, `report.md`, and artifact
  validation.

Defaults keep only policy/label disagreements so the dataset is focused on actual
corrections. Use `--correction-keep-agreements` only for smoke/debug runs where a
non-empty artifact is useful even if the policy already matches the label.

The collected dataset is now trainable through the same auxiliary-loss extension path:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_correction_finetune \
  --checkpoint <selected-checkpoint.pth> \
  --correction-dataset <correction_examples.npz>
```

`correction-finetune` restores the selected checkpoint, installs the correction states
with `Agent.set_correction_action_dataset(...)`, and enables an opt-in low-weight
action-margin loss:

- `--correction-action-weight` default `0.02`
- `--correction-action-margin` default `0.6`
- `--correction-action-batch-size` default `64`

The run summary and markdown report include correction transition count, dataset path,
weight, margin, batch size, `correction_action_samples_100`, and
`avg_correction_action_loss_100` / `avg_correction_action_accuracy_100`.
Selected-checkpoint restore also preserves these fields so a correction-fine-tuned
checkpoint can be re-evaluated honestly later.

`correction-finetune` now fails fast on empty correction datasets. Artifact validation
also checks that correction fine-tune runs have a real dataset file, positive transition
count, positive sample count, positive weight/batch, valid metric fields, and normal
final eval evidence.

Recipe hardening:

- `b3s_correction_collect` is a registered dataset recipe and requires
  `--checkpoint`.
- `b3s_correction_finetune` is a registered comparison recipe and requires both
  `--checkpoint` and `--correction-dataset`.
- `list-recipes` now shows those requirements so future sessions do not have to
  rediscover the correction command shape from this document.

Smoke evidence:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260624_193530_codex_correction_arch_harden_smoke`
- Input dataset: `2` states, shape `(2, 295)`.
- Result: artifact validation `ok: true`; correction loss path executed with
  `avg_correction_action_loss_100 = 0.0106`,
  `avg_correction_action_accuracy_100 = 100%`, and
  `correction_action_samples_100 = 100`.
- Caveat: this was a tiny mechanics smoke, not an NN improvement run.

## Latest Tested Method And Next Direction

The latest tested infrastructure change is the B12 metric audit / promotion-gate fix.
It added `metric-audit` and outcome-conditioned non-success depth to promotion snapshots.
B10 now promotes as the best eval-time controller baseline because its failed validation
episodes still preserve route depth (`69.5%`), even though raw mean depth is lower
(`54.3%`) due to more successful episodes ending early.

The latest tested NN-training probe remains B16 class-aware jump confidence gating on top
of B15. It improved the harder seed-1 selected check from `8/30` to `9/30`, but regressed
seed 0 versus B15 from `13/30` to `12/30`. B15 remains the best learned contact-head
adapter baseline because it promotes on seed 0 selected+validation (`13/30`, `22/60`) and
B16 has not beaten it across seeds. B17+B18 improved dataset coverage but failed held-out
calibration (`58.3%`), so the next work is label-quality audit rather than another
selector run.

The older eval-only final-contact option on top of B3s should not replace B3s under the
current gate, but it remains an important positive signal: better local contact logic
increased first-crystal wins on both selected and expanded samples.

- Selected artifact:
  `.codex/artifacts/cc_sessions/20260624_222911_b3s_final_contact_option_eval30`
- Expanded validation artifact:
  `.codex/artifacts/cc_sessions/20260624_223318_b3s_final_contact_option_val60`
- Mechanism: restore the B3s selected checkpoint, use the NN for route actions, and
  switch to a local oracle-planned close-zone option when the current target is within
  `3.0` tiles. The option commits `8` planned actions before replanning.
- Selected result: `13/30` first-crystal wins, `43.3%` crystals, `50.5%` depth.
- Validation result: `24/60` first-crystal wins, `40.0%` crystals, `53.6%` depth.
- Gate result: `REGRESS` because expanded validation depth was `0.536`, below the
  required `0.570` guardrail, even though validation wins improved over B3s `19/60`.
- Interpretation: the close-zone option is useful but overactive. It controlled `35.9%`
  of selected-eval steps and `29.8%` of validation steps, improving contact success
  while damaging the broader depth profile.

Keep `eval-final-contact-option` as a diagnostic mode. Do not promote it as the active
baseline without either clearing the depth guardrail or explicitly changing the
promotion rule.

Two follow-ups changed one variable at a time and should not be promoted:

- B3w narrowed the trigger to `1.5` tiles with commit `8`:
  `.Codex/artifacts/cc_sessions/20260624_230035_b3s_final_contact_option_d15_commit8_eval30`.
  It restored depth (`61.9%`) but under-fired, dropping to `8/30` selected wins and
  `26.7%` crystals. Gate result: `REGRESS`.
- B3x kept the `3.0` tile trigger and shortened commit to `4`:
  `.Codex/artifacts/cc_sessions/20260624_230324_b3s_final_contact_option_d30_commit4_eval30`.
  It reached `11/30` selected wins and `36.7%` crystals, but depth stayed low (`50.9%`)
  and it trailed B3v's `13/30` selected wins. Gate result: `HOLD`, but not worth
  expanded validation yet.
- B3y kept the `3.0` tile trigger and `8`-action commit, but cancelled queued macro
  actions once the target left the close zone:
  `.Codex/artifacts/cc_sessions/20260624_231322_b3s_final_contact_option_d30_commit8_cancel_eval30`.
  It tied B3v's selected result (`13/30`, `43.3%` crystals, `50.5%` depth), but
  cancelled only `61` queued actions across `55` plans and did not fix the depth
  profile. Gate result: `HOLD`.

Do not continue with more small final-contact threshold/commit variants by default. B10
changed the mechanism from broad takeover to advantage-gated takeover, reduced option
control to about `0.8%` of eval steps, and is now the promoted eval-time controller
baseline after the metric audit showed non-success route depth held. Future work should
try to internalize this behavior into the network instead of tuning more option
thresholds.

The previous full DAgger-style correction pass on top of B3s was also tested and should
not be promoted as-is.

- Correction collection artifact:
  `.Codex/artifacts/cc_sessions/20260624_215607_b3s_correction_collect_b3s_ep300_30g`
- Fine-tune artifact:
  `.Codex/artifacts/cc_sessions/20260624_215710_b3s_correction_finetune_b3s_ep300_1024_300`
- Dataset: `1024` policy-visited states, `82.3%` policy/label disagreement, mostly
  loop/stale triggers.
- Fine-tune result: final held-out `3/16` first-crystal wins, `18.8%` crystals, and
  `29.5%` depth versus B3s's `10/30` selected wins and `60.5%` depth.
- Gate result: `HOLD` only because the candidate final eval sample was smaller than
  B3s's selected sample; behaviorally it regressed wins, crystals, depth, near-miss
  rates, and mean target distance.

Do not repeat the same correction recipe unchanged (`1024` disagreement states, weight
`0.020`, margin `0.60`) as the next default path. If revisiting correction training,
change one variable at a time:

1. Lower the correction weight, for example `0.005` or `0.010`.
2. Stratify/downsample repeated loop states so stale loops do not dominate route
   behavior.
3. Add selected-checkpoint save/eval support for `correction-finetune` before another
   full comparison, so it can clear or fail the normal `30`-game selected gate.

If not revisiting correction training, move to a new NN or curriculum method rather than
more eval-only final-contact tuning. B3s and the option probes show that the policy can
approach the first objective; the remaining problem is teaching reliable contact while
preserving route depth under the learned policy itself.

Latest experiment: B4 contact-only correction fine-tune. The B4 collection gate
passed at
`.Codex/artifacts/cc_sessions/20260625_030239_b4_contact_only_correction_collect`:
`275` close-zone-only disagreement examples, `81.6%` policy-label disagreement, and a
usable action mix across idle, jump, left/right, and left/right-jump labels. This avoids
repeating the loop-heavy `1024`-state correction dataset that regressed.

The first fine-tune from the B3s selected checkpoint at
`--correction-action-weight 0.010` did not pass the early-stop gate:
`.Codex/artifacts/cc_sessions/20260625_030453_b4_contact_only_correction_finetune_w010_300`
was stopped at episode `155/300` after the episode-150 eval fell to `1/8` wins,
`12.5%` crystals, and `20.5%` depth. The correction objective was active
(`275` transitions, `100` recent sampled updates, `82.1%` correction-label accuracy),
so the failure mode is not missing plumbing; it is auxiliary contact pressure damaging
the route policy.

The lower-weight `0.005` retry also failed:
`.Codex/artifacts/cc_sessions/20260625_031606_b4_contact_only_correction_finetune_w005_300`
was stopped at episode `102/300` after episode-100 eval reached only `0/8` wins,
`0.0%` crystals, and `16.1%` depth, while still sampling the correction objective
(`275` transitions, `100` recent sampled updates, `81.3%` correction-label accuracy).
Treat this correction-loss lane as closed unless there is a concrete new mechanism, not
just another weight sweep. Move to staged curriculum or network/curriculum architecture
changes that preserve route behavior while teaching contact.

Current next mechanism: B5 anchored contact correction. This keeps the B4 contact-only
dataset but adds an external frozen-teacher policy-anchor provider through
`src/ai/extension_contracts.py`: the current policy receives the low-weight contact
action-margin loss, while a KL distillation loss anchors replay-batch action preferences
to the restored B3s checkpoint. This tests whether preserving route behavior during
fine-tune fixes the B4 failure mode.

- Recipe: `b5_anchored_contact_correction`.
- Correction weight: `0.005`.
- Policy anchor weight/temperature: `0.020` / `1.0`.
- Smoke artifact:
  `.Codex/artifacts/cc_sessions/20260625_070236_b5_anchored_contact_smoke_20`.
- Smoke result: artifact validation `ok`; both correction and anchor metrics were
  active (`100` recent samples each). Do not use the 20-episode smoke as a promotion
  sample.

First comparison at anchor weight `0.020` failed early:
`.Codex/artifacts/cc_sessions/20260625_070641_b5_anchored_contact_correction_w005_a002_300`
was stopped at episode `101/300`. Held-out eval was `1/8` wins, `12.5%` crystals,
`25.0%` depth at episode 100. The correction objective was active (`81.6%` correction
accuracy), and the anchor objective was active, but teacher-action match was only
`33.4%`. That points to insufficient route-preservation pressure rather than missing
plumbing. One stronger-anchor calibration is acceptable; if it still fails, move off
correction/anchor methods to staged curriculum.

The stronger anchor calibration also failed:
`.Codex/artifacts/cc_sessions/20260625_071613_b5_anchored_contact_correction_w005_a010_300`
used anchor weight `0.100` and was stopped at episode `108/300`. It improved
teacher-action match to `54.3%`, but episode-100 held-out eval was still only `2/8`
wins, `25.0%` crystals, and `22.3%` depth. Do not keep sweeping anchor weights. The
next method should be staged curriculum / training-level scheduling that teaches contact
in isolated levels without adding contact-label pressure to every replay update.

Concrete next build status: B6 staged contact curriculum plumbing is now implemented
and smoke-tested. It reuses the existing status-session, selected-checkpoint,
interleaved-lane, and artifact-validation infrastructure. The missing layer that was
added:

- training-only fixed contact levels (`contact_floor`, `contact_jump_up`,
  `contact_drop_return`, `contact_step_pair`, `contact_exit_after_crystal`);
- a `contact-interleaved` mode that restores B3s and trains `6` normal tutorial lanes
  plus `2` contact lanes at `vec-envs=8`;
- a `b6_contact_interleaved` recipe requiring `--checkpoint`;
- contact-lane metrics in live/report output;
- a route-preserving selection gate that rejects low-depth contact gains.

Do not add correction, close-zone action, or policy-anchor losses to B6. The point is to
teach contact through normal rewards and training-only level distribution.

Smoke artifact:
`.Codex/artifacts/cc_sessions/20260625_074529_b6_contact_interleaved_smoke_20`.
Artifact validation passed. The smoke confirmed checkpoint restore, lane mixing,
selected checkpoint save/eval, live metric aliases, and report output. It was only
`20` episodes with tiny eval counts, so it is not performance evidence.

Real fixed-level B6 run:
`.Codex/artifacts/cc_sessions/20260625_075416_b6_contact_interleaved_25pct_300`.
It was stopped after the ep150 source eval because it failed the continuation rule.
The source evals were ep50 `0/16`, `12.5%` crystals, `36.2%` depth; ep100 `0/16`,
`12.5%` crystals, `39.3%` depth; ep150 `0/16`, `6.2%` crystals, `35.7%` depth. Contact
lanes reached `100%` win/crystal/exit in live metrics, but held-out full tutorial caves
did not improve. Do not rerun fixed contact interleave unchanged; if revisiting this
lane, make the training contact distribution more game-faithful or move to a stronger
route/contact diagnostic first.

The unified route/contact scorecard is now built into status-session summaries, reports,
partial interrupted summaries, selected-source checkpoint selection, and `compare-artifact`.
It consolidates first-crystal rate, depth, close-zone reach, close-zone action mix,
stuck/loop-after-close, target-distance bands, target-distance best/final deltas, stall
rate, and end reasons. The selected-policy score preserves route depth instead of
selecting on wins alone:
`3.0 * first_crystal_rate + 1.5 * crystal_frac + 1.0 * depth_frac + 0.5 * close_zone_rate - 1.0 * loop_after_close_rate - 0.5 * stall_rate`.

Backfilled scorecard results:

- B3s conservative demo-Q: `1.821`, `contact regression`.
- B6 fixed contact interleave: `0.799`, `route regression`.
- B7 generated contact-pool interleave: `1.308`, `depth regression`.
- B8 history state: `1.757`, `contact regression`.
- B9 C51 distributional DQN: `1.714`, `contact regression`.
- B10 advantage-gated final-contact selected: `2.702`, `depth regression`.
- B10 advantage-gated final-contact validation: `2.685`, `depth regression`.

Interpretation: B6 and B7 should not be rerun unchanged. Both mastered contact lanes
while losing the route profile needed to reach those contacts on held-out tutorial
caves. B7 was less bad than B6, but still below the B3s route/contact score and depth
guardrail.

B8 history-state route/policy control is built and tested. It tied B3s on selected wins
(`10/30`) but trailed the B3s route/contact score (`1.757` vs `1.821`) and did not change
the close-zone jump failure (`0%` close-zone jump rate). Keep the opt-in state extension
for future experiments, but do not make it the default baseline.

B9 C51 distributional DQN is also built and tested. It tied B3s on selected wins/crystals
(`10/30`, `33.3%`) but trailed B3s on route/contact score (`1.714` vs `1.821`) and
worsened loop-after-close (`43.3%`). Keep the opt-in C51 extension for future controlled
tests, but do not make it the default baseline.

B10 advantage-gated final-contact control is built and tested. It improved selected wins
to `15/30` and validation wins to `30/60`, with option takeover only about `0.8%` of eval
steps. After the metric audit, it now promotes as the best eval-time controller baseline:
raw validation depth is `0.543`, but validation non-success depth is `0.695`, clearing the
outcome-conditioned route guardrail. Keep B3s as the pure-NN training baseline until a
learned checkpoint matches this without the option.

B11 gate-accepted label transfer is also built and tested. It produced a focused
`162`-state correction dataset and the fine-tune learned the labels, but final held-out
performance fell to `4/16` wins, `25.0%` crystals, and `44.2%` depth. Keep the collection
mode for analysis, but do not use this correction-fine-tune recipe as the next default
baseline path.

B15 offline/head-only contact transfer is built and seed-0 validation-tested. It should be
the default shape for future learned contact additions: freeze the base route policy
first, fit the adapter offline, gate its use by confidence, then improve data/gating before
claiming multi-seed robustness. B16 class-aware jump confidence gating is also built and
tested; it is useful infrastructure, but not a promotion because it helps seed 1 slightly
while regressing seed 0 versus B15. B17 hard-seed collection and B18 combine/calibration
are complete; do a label-quality audit next instead of online fine-tuning, blind
confidence sweeps, or another selected contact-head eval from the current combined data.

## Auxiliary-Loss Organization

The agent now has two layers for experiment-only losses:

- `src/ai/extension_contracts.py` defines the provider/result contract used by the core
  optimize loop.
- `src/ai/action_margin_loss.py` owns reusable DQfD-style supervised action-margin loss
  math for demo, close-zone, correction, and future provider losses.

Future methods should reuse `sample_action_margin_loss(...)` instead of copying the Q
margin/conservative-Q calculation into a runner or mixin. This keeps new NN expansion
code easier to test independently from a full `Agent` instance.

## Reliability Checklist

Before a new NN expansion gets a full run:

1. Unit tests cover loss shape, finite values, metric names, and opt-in behavior.
2. The status-session recipe can build and execute the intended command.
3. A one-episode smoke run passes artifact validation and proves metrics appear in
   `summary.json`, `report.md`, `live_metrics.json`, and `artifact_validation.json`.
4. Interrupted status-session runs must still write a partial `summary.json`/`report.md`
   from `live_metrics.json` and, when available, `source_eval_history.jsonl`.
5. Correction-style methods first produce a validated correction dataset artifact before
   adding or enabling a training loss.
6. Correction-fine-tune runs must show positive correction sample counts in live/final
   metrics before they receive a full comparison budget. The loss itself may be zero if
   the margin is already satisfied.
7. No new method is enabled by default in top-level `Config` before promotion.
8. Files stay under the current 1000-line gate, and new modules should aim for 500 lines
   unless they are stable infrastructure.
9. `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passes before PR
   or merge decisions.

## What Not To Cement

Do not promote these as default next paths without new evidence:

- More terminal reward tuning.
- More broad successful scripted-demo coverage.
- Beam demo source as default training data.
- Standalone close-zone action loss from successful scripted states.
- Route-direction auxiliary loss as the main next lever.

Those paths either failed to improve B3s or improved internal metrics without improving
held-out outcomes.
