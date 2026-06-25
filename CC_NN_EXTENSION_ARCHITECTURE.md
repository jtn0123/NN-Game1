# Crystal Caves NN Extension Architecture

Last updated: 2026-06-24

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

### 3. Documentation

Update these files while working:

- `CC_NN_EXPERIMENT_TRACKER.md` for active run decisions and outcomes.
- `CC_NN_CLEANUP_AUDIT.md` when deciding what to keep, archive, or delete.
- This file when the extension workflow itself changes.

Rejected ideas should stay documented but should not stay as default config paths.

### 4. Artifact Validation

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

### 5. Promotion Gate

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

### 6. Policy-Visited Correction Data

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

- `b3s_correction_collect` is a recommended dataset recipe and requires
  `--checkpoint`.
- `b3s_correction_finetune` is a recommended comparison recipe and requires both
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

## Recommended Next Method

The next method should be a narrow DAgger-style correction training pass on top of B3s:

1. Start from the B3s recipe/checkpoint.
2. Run `run-recipe b3s_correction_collect` on train-pool or held-out diagnostic caves.
3. Inspect disagreement rate, trigger mix, label actions, and row samples.
4. Fine-tune with `run-recipe b3s_correction_finetune` and a low-weight supervised
   correction action loss.
5. Evaluate with the same selected-checkpoint plus expanded held-out validation process.

Why this is preferred: B3t/B3u showed that successful scripted close-zone labels can be
learned, but they did not match the states where the policy actually fails. The correction
pass attacks that distribution shift directly.

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
4. Correction-style methods first produce a validated correction dataset artifact before
   adding or enabling a training loss.
5. Correction-fine-tune runs must show positive correction sample counts in live/final
   metrics before they receive a full comparison budget. The loss itself may be zero if
   the margin is already satisfied.
6. No new method is enabled by default in top-level `Config` before promotion.
7. Files stay under the current 1000-line gate, and new modules should aim for 500 lines
   unless they are stable infrastructure.
8. `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passes before PR
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
