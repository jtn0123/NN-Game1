# Crystal Caves NN Cleanup Audit

**Date:** 2026-06-24
**Scope:** current dirty Crystal Caves NN experiment tree on `claude/cc-drill-levels`.
**Goal:** separate durable improvements from negative-value experiment paths so future
sessions do not keep retesting the same failed ideas.

## Executive Verdict

Keep and cement the measurement/runner infrastructure and B3s conservative-demo route
baseline. Do not promote most reward/penalty/scaffold/close-zone tweaks into defaults.
The repeated pattern is clear: many changes improve an internal metric, source-pool
training score, or a tiny eval sample, but do not survive the held-out selected-checkpoint
gate.

Current best route baseline remains:

- **B3s conservative demo-Q:** `10/30` selected first-crystal wins, `33.3%` crystals,
  `60.5%` depth; validated at `19/60`, `31.7%`, `60.0%` depth.
- Artifact:
  `.Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300`
- Validation artifact:
  `.Codex/artifacts/cc_sessions/20260624_122423_b3s_selected_ep300_eval60`

## Cement / Keep

| Area | Decision | Why |
|---|---|---|
| `experiments/cc_status_session.py` runner | Keep, but split later | It gave comparable runs, live metrics, selected checkpoints, near-miss eval, and eval-only checkpoint validation. This directly fixed the earlier "we might miss smaller improvements" concern. |
| `CC_NN_EXPERIMENT_TRACKER.md` | Keep | It is the durable no-go / findings ledger. Future sessions should read it before trying new NN ideas. |
| Selected-only checkpoint + `eval-checkpoint` | Cement | It replaced replay-heavy snapshots and allowed 60-game validation without retraining. This prevented B3t from being falsely promoted. |
| Near-miss and close-zone diagnostics | Cement | These explain why a top-line result moved: depth, target distance, close-zone jump, idle/interact, stuck-after-close, loop-after-close. |
| `PrioritizedNStepReplayBuffer` | Keep | PER+n-step is a real correctness/infrastructure fix. It may not solve held-out completion alone, but it removes a misleading config downgrade. |
| Demo action loss + conservative CQL-style term | Keep | B3s is the first algorithmic improvement that held up. Keep `tutorial-demo-conservative` as the baseline recipe. |
| Direct+recovery route demos | Keep | They doubled successful demo coverage over direct-only and became the input data for B3s. |
| Drill levels and skill diagnostics | Keep | They showed the agent can learn isolated skills and helped identify transfer/generalization as the actual blocker. |

## Keep Only As Diagnostic / Experiment-Only

| Area | Decision | Why |
|---|---|---|
| Route scaffolds: `route_floor`, `route_catch`, `route_offset` | Keep as diagnostic scaffolds, not defaults | They were controllable and useful for diagnosis, but transfer to normal tutorial did not beat direct route training. |
| Bridge levels and bridge interleave modes | Keep only if used for future option/skill tests | B3k/B3l/B3m improved some local action-shape metrics but consistently hurt route approach/depth. Do not use bridge interleave as default route training. |
| Route auxiliary direction head | Quarantine as optional | The network learned the auxiliary task (`~97%`), but route outcomes worsened. Keep metric plumbing if cheap; do not enable by default. |
| Beam planner demo source | Keep parser/controller only as diagnostic | It improved demo coverage but hurt learned policy badly. Do not use `beam` as default training data. |
| Filtered/weighted demo selection | Keep tooling, not baseline | It made artifacts smaller and data quality visible, but selected eval still trailed B3s. |
| Close-zone scripted/oracle action losses | Keep only as diagnostic | B3t and B3u learned internal labels but did not beat B3s validation. Do not keep adding close-zone action loss runs without a state-distribution change. |
| Invalid-action and novelty reward flags | Keep disabled or remove from core later | They helped expose symptoms, but standalone results were not promoted. |

## Remove / Do Not Repeat

| Area | Cleanup Decision | Evidence |
|---|---|---|
| Replay-heavy full checkpoints from rejected B3p | Delete local `.pth` bloat after preserving reports | B3p was not promoted and wrote `2.2G` mostly in `crystal_caves_final.pth` and `crystal_caves_best.pth`; selected-only checkpoint support replaced this workflow. |
| More terminal reward tuning | Do not repeat soon | Prior reward magnitude changes did not help because the agent usually fails before terminal rewards fire. |
| More broad scripted-demo coverage as the next lever | Do not repeat soon | B3p: demo coverage rose to `85/128`, but selected eval fell to `4/30`. More data was not better data. |
| More bridge interleave from scratch | Do not repeat soon | B3k/B3l reduced some local failures but hurt route depth and first-crystal results. |
| Close-zone action loss as standalone next lever | Do not repeat soon | B3t failed 60-game validation, B3u tied B3s but worsened depth/stuck/loop profile. |
| Route direction auxiliary as next lever | Do not repeat soon | It solved the auxiliary metric but not the route-control problem. |

## Cleanup Performed

- Added `.Codex/artifacts/` to `.gitignore` so local session outputs do not keep
  appearing as untracked source changes.
- Moved niche Crystal Caves experiment defaults out of top-level `Config` and into
  `src/game/crystal_caves_experiments.py`; the status-session runner installs them
  only when it needs old experiment reproducibility.
- Split oversized implementation/test/docs files under the 1000-line budget:
  - `experiments/cc_status_session.py` is now a compatibility wrapper over
    `experiments/cc_status/` modules.
  - AI experiment losses, prioritized n-step replay, Crystal Caves geometry,
    headless helpers, metrics publisher NN helpers, dashboard CSS, and matching
    tests were moved into focused files.
  - `CC_NN_EXPERIMENT_TRACKER.md` now stays compact and links to archived history
    under `docs/cc_nn_experiment_tracker/`.
- Deleted the two replay-heavy ignored checkpoints from rejected B3p:
  - `.Codex/artifacts/cc_sessions/20260624_090216_tutorial_demo_bc_beam_pool512_select30_300/tutorial_demo_bc/models/crystal_caves/crystal_caves_final.pth`
  - `.Codex/artifacts/cc_sessions/20260624_090216_tutorial_demo_bc_beam_pool512_select30_300/tutorial_demo_bc/models/crystal_caves/crystal_caves_best.pth`
- Result: B3p artifact size fell from `2.2G` to `52M`; total `.Codex/artifacts` size
  fell to `170M`.
- Validation after cleanup:
  - `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed.
  - Size gate passed: all checked files are `<= 1000` lines.
  - Full coverage gate passed: `1032 passed`, total coverage `76.81%`.

## Code Cleanup Recommendations

### Immediate Safe Cleanup

1. Delete ignored replay-heavy `.pth` files from rejected runs when they are not selected
   checkpoints and the tracker already stores the result.
2. Keep `report.md`, `summary.json`, near-miss summaries, and selected-only checkpoints
   for B3s and validation runs.
3. Add `.Codex/artifacts/` to `.gitignore` if these local artifacts should never appear
   as untracked noise. The repo already ignores `*.pth`, but not the artifact reports.

### Source Cleanup Before PR

1. Keep `PrioritizedNStepReplayBuffer`, demo-action dataset APIs, and conservative demo-Q
   support in core AI code.
2. Keep close-zone dataset APIs only if the runner still needs B3t/B3u reproducibility;
   otherwise move them behind experiment-only helpers to reduce core agent surface.
3. Keep negative or niche experiment flags in `src/game/crystal_caves_experiments.py`,
   not top-level `Config`: anti-loop, novelty, invalid-interact, invalid-shoot,
   route-aux, close-zone labels, demo-action labels, and bridge mode.
4. Keep new status-session modules small. Do not add new run modes directly to the
   wrapper; place config/vec/eval/demo/report/run-mode code in the matching
   `experiments/cc_status/` module.

## Next Strategic Direction

Do not keep pushing successful-demo close-zone labels. The best next experiment is a
narrow DAgger-style correction pass:

1. Start from B3s recipe/checkpoint.
2. Roll out the learned policy on train-pool caves.
3. Collect actual failure states: stalls, loops, close-zone-without-collection.
4. Label those visited states with the route controller or local oracle.
5. Fine-tune with a small correction set and evaluate against B3s using the same
   selected-checkpoint + 60-game validation process.

This directly attacks distribution shift: B3u showed that labels from successful scripted
close-zone states can be learned, but they do not match the states where the policy
actually fails.

## Extension Reliability Update

Added `CC_NN_EXTENSION_ARCHITECTURE.md` as the permanent workflow for future NN expansion.
The code now has two guardrails:

- `src/ai/extension_contracts.py` defines typed auxiliary loss/metric contributions, and
  the existing route/demo/conservative/close-zone losses flow through that path.
- `experiments/cc_status/recipes.py` makes the promoted B3s full run and B3s smoke run
  machine-readable and test-covered.

Use this before adding the next NN method: implement the contribution, add metrics, add
a smoke recipe and a comparison recipe, then promote only against the B3s validation bar.

## Recipe / Artifact Reliability Update

Added the first plug-and-play experiment controls:

- `python experiments/cc_status_session.py list-recipes` lists available B3s recipes
  without loading the full game stack.
- `python -u experiments/cc_status_session.py run-recipe <recipe-key> [overrides]`
  expands a named recipe into the normal status-session CLI, keeping existing run code
  as the execution path.
- `experiments/cc_status/artifacts.py` validates `summary.json`, `report.md`, live
  metrics, selected-checkpoint diagnostics, near-miss eval, and selected checkpoint files
  after each run.
- Validation writes `artifact_validation.json` and fails the process on missing required
  evidence. Use `--no-artifact-validation` only for local debugging.

This is the new minimum bar for future NN expansion: if a smoke recipe cannot execute and
produce valid artifacts, it should not receive a full comparison run.

Validation smoke:

- Command:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_conservative_smoke --label codex_recipe_validator_smoke2`
- Artifact:
  `.Codex/artifacts/cc_sessions/20260624_185336_codex_recipe_validator_smoke2`
- Result: passed end-to-end and wrote `artifact_validation.json` with `ok: true`.
- Regression fixed during smoke: `summarize_trainer()` depended on trainer source-stat
  helpers that had been left in `training.py` after the file split. Those pure helpers
  now live in `experiments/cc_status/stats.py`, where both live metrics and reports can
  import them without a circular dependency.

## Promotion Gate Reliability Update

Added `experiments/cc_status/promotion.py` and the wrapper command:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python experiments/cc_status_session.py compare-artifact <candidate-artifact> [--validation <expanded-validation-artifact>]
```

This makes the B3s promotion bar executable instead of subjective:

- `PROMOTE` only when selected eval and expanded validation clear the frozen B3s
  reference.
- `HOLD` for under-sampled, smoke-only, or promising-but-unvalidated runs.
- `REGRESS` for candidates that trail B3s or tie selected wins without support metrics.

The gate compares win rates when sample sizes differ and records support metric changes
for crystals, depth, near misses, close-zone jumps, stuck rate, and loop rate. Future
experiments should keep non-promoted findings in the tracker, but only replace the active
baseline after this gate passes.

## Correction Dataset Architecture Update

Added `experiments/cc_status/corrections.py` and the wrapper mode:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py collect-corrections --checkpoint <selected-checkpoint.pth>
```

This cements the next DAgger-style path without yet changing training:

- rolls out a selected checkpoint greedily on eval caves;
- captures policy-visited close-zone, stale, and loop states;
- labels close-zone states with the local oracle and stale/loop states with the recovery
  route controller;
- writes `correction_examples.npz`, per-state JSONL, summary JSON, report lines, and
  artifact validation evidence.

Default behavior keeps only policy/label disagreements. `--correction-keep-agreements`
exists for smoke/debug checks and should not be treated as the default training dataset.

Validated smoke:

- Command used existing smoke selected checkpoint with `--correction-stale-steps 1`,
  `--correction-sample-every 1`, and `--correction-keep-agreements`.
- Artifact:
  `.Codex/artifacts/cc_sessions/20260624_191511_codex_correction_collector_keep_smoke`
- Result: artifact validation `ok: true`; dataset arrays shape `(2, 295)`, actions
  `(2,)`, trigger masks `(2,)`.

Next cleanup/architecture step: wire this dataset into a low-weight correction action
loss through the existing auxiliary-loss extension path, then require promotion through
`compare-artifact`.

## Correction Fine-Tune Architecture Update

The correction dataset is now wired into training rather than only collection:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py correction-finetune \
  --checkpoint <selected-checkpoint.pth> \
  --correction-dataset <correction_examples.npz>
```

What was cemented:

- `Agent.set_correction_action_dataset(...)` installs correction states/actions.
- `CRYSTAL_CAVES_CORRECTION_ACTION_LOSS` enables a low-weight action-margin loss through
  the generic auxiliary-loss path.
- Status-session live metrics, summaries, reports, and selected-checkpoint config
  snapshots now include correction action loss, accuracy, weight, margin, and batch size.
- `correction-finetune` artifact validation passes.

Validated smoke:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260624_192405_codex_correction_finetune_smoke`
- Dataset: `2` states, shape `(2, 295)`.
- Result: artifact validation `ok: true`; correction action loss sampled
  (`0.0106` avg100) with `100%` accuracy.
- Interpretation: architecture proof only. It is not a meaningful performance run.

Validation after this update:

- Focused tests: `62 passed`.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1065 passed`, total coverage `77.02%`, file-size gate clean, dependency audit clean.

Next experiment step: collect a larger disagreement-only correction dataset from the B3s
selected checkpoint, run `correction-finetune` with B3s-comparable eval settings, then
use `compare-artifact` before treating it as better than B3s.

## Architecture Hardening Update

Follow-up from the architecture review:

- Added `Agent.register_auxiliary_loss_provider(...)` so future standalone NN losses can
  be added as provider objects returning `AuxiliaryLossContribution` values.
- Kept the current built-in route/demo/close-zone/correction losses in place to avoid a
  broad behavior refactor before the next run.
- Made `correction-finetune` reject empty correction datasets before training starts.
- Added artifact validation for `correction_training`: dataset file, positive transition
  count, positive sample count, positive weight/batch, non-negative margin/loss, and
  accuracy in `[0, 1]`.
- Added `correction_action_samples_100` to live metrics, summaries, and reports so a
  zero hinge loss is still visibly different from an unsampled correction loss.

Validated smoke:

- Artifact:
  `.Codex/artifacts/cc_sessions/20260624_193530_codex_correction_arch_harden_smoke`
- Result: artifact validation `ok: true`; `2` correction transitions, `100` correction
  samples, `0.0106` avg correction action loss, `100%` correction action accuracy.

## Non-Run Architecture Follow-Up

Follow-up focused on organization and repeatability, with no new NN training run:

- Added guarded status-session recipes for the next correction workflow:
  `b3s_correction_collect` and `b3s_correction_finetune`.
- Added `required_overrides` to recipes so checkpoint/dataset-dependent recipes fail
  before expansion unless the caller supplies explicit paths.
- Marked the correction collect/fine-tune recipes as recommended next workflow steps
  while keeping B3s as the promoted baseline.
- Updated tests so `list-recipes`, `run-recipe`, and required input validation are
  covered.

Result: the next architecture-supported experiment can be run by recipe name, but cannot
be accidentally launched without the selected checkpoint and correction dataset inputs.

Additional organization cleanup:

- Added `src/ai/action_margin_loss.py` as the shared implementation of supervised
  DQfD-style action-margin loss.
- Rewired the demo/close-zone/correction mixin path to call that helper instead of
  owning the Q-margin math inline.
- Added a direct helper test so future auxiliary-loss providers can reuse the function
  without depending on full-agent tests.

Validation after this non-run architecture pass:

- Focused tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_cc_status_recipes.py tests/test_agent_experiments.py -q`
  passed with `22 passed`.
- Focused static checks: Ruff, mypy, and `git diff --check` passed for touched
  recipe/auxiliary-loss/doc files.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1074 passed`, coverage `77.04%`, JS tests green, mypy green, file-size gate clean,
  dependency audit clean, and package build successful.

## Simplify + Debug Architecture Pass

Follow-up command pass on architecture and NN code, with no NN training run:

Simplification checks:

- Searched for a literal repo `simplify` command; none exists.
- File-size gate passed:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python .github/scripts/check_file_size.py --max-lines 1000`
- Internal import graph across `src/ai` and `experiments/cc_status` found no cycles.
- AST size scan found the largest simplification targets:
  - `experiments/cc_status/cli.py::main` at about `799` LOC.
  - `experiments/cc_status/cli_args.py::add_status_session_arguments` at about `452` LOC.
  - `experiments/cc_status/reports.py::write_markdown_report` at about `415` LOC.
  - `src/ai/agent.py::__init__` at about `191` LOC.
  - `src/ai/agent.py::_learn_step_internal` at about `137` LOC.

Debugging checks:

- Focused NN tests passed:
  `tests/test_agent_experiments.py tests/test_prioritized_n_step.py tests/test_replay_buffer.py tests/test_network.py tests/test_agent.py`
  with `175 passed`.
- Focused status-session tests passed:
  `tests/test_cc_status_recipes.py tests/test_cc_status_artifacts.py tests/test_cc_status_promotion.py tests/test_cc_status_reports.py tests/test_cc_status_session.py tests/test_cc_status_corrections.py`
  with `79 passed`.
- Ruff passed across `src/ai`, `experiments/cc_status`, and focused tests.
- Direct mypy over `src/ai experiments/cc_status` initially exposed:
  - real call-site drift: `invalid-interact` and `novelty-bonus` did not pass
    `cave_pool_size` after `run_diagnostic_baseline(...)` gained that argument.
  - a nullable artifact-validation typing issue around `run["config"]`.
  - remaining `attr-defined` noise from intentionally dynamic Crystal Caves experiment
    flags installed by `install_crystal_caves_experiment_defaults(...)`.

Fixes made:

- Added missing `cave_pool_size=opts.cave_pool_size` to the `invalid-interact` and
  `novelty-bonus` diagnostic modes.
- Typed the artifact validator's local `config` dict after checking `run.get("config")`.

Validation after fixes:

- Focused tests passed with `63 passed`.
- Ruff passed.
- Mypy with only the intentional dynamic-config attr errors disabled passed:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy --disable-error-code attr-defined src/ai experiments/cc_status`
  reported `Success: no issues found in 35 source files`.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1074 passed`, coverage `77.04%`, dependency audit clean, file-size gate clean, and
  package build successful.

## Direct Status-Session Mypy Fix

Follow-up fix for the remaining direct mypy errors:

- Added `CrystalCavesExperimentConfig` in `experiments/cc_status/common.py`, a typed
  protocol for the experiment-only flags installed by
  `install_crystal_caves_experiment_defaults(...)`.
- Added `cc_experiment_config(config)` as the explicit adapter from base `Config` to that
  experiment-only view.
- Routed status-session reads/writes of dynamic Crystal Caves flags through the adapter
  in config helpers, reports, vector env setup, baseline/demo runs, and selected
  checkpoint restore.
- Kept rejected/niche experiment flags out of global `Config`, preserving the cleaner
  production config surface while making the experiment boundary typed.

Validation:

- Direct mypy now passes without suppressions:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy src/ai experiments/cc_status`
  reported `Success: no issues found in 35 source files`.
- Focused tests passed:
  `tests/test_cc_status_artifacts.py tests/test_cc_status_session.py tests/test_cc_status_recipes.py tests/test_cc_status_reports.py tests/test_cc_status_corrections.py tests/test_agent_experiments.py tests/test_prioritized_n_step.py`
  with `87 passed`.
- Ruff and `git diff --check` passed on the touched architecture/NN files.
- Full verification passed:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` with
  `1074 passed`, coverage `77.04%`, dependency audit clean, file-size gate clean, and
  package build successful.

## Parser Simplification Debug Pass

Follow-up simplify/debug pass, with no NN training run:

- Re-scanned `src/ai` and `experiments/cc_status` for oversized functions/classes.
- Confirmed all files remain below the 1000-line file-size gate.
- Refactored `experiments/cc_status/cli_args.py` so the public
  `add_status_session_arguments(...)` function is now an 11-line dispatcher instead of a
  452-line flag block.
- Split the CLI surface into focused helper groups: mode, training schedule, route
  demos, demo supervision, eval/logging/trace, interleave/bridge, reverse/archive,
  runtime, checkpoint/correction, and artifact persistence.
- Added a parser regression test covering representative flags from each major group so
  future CLI additions fail loudly if the parser wiring drifts.

Resulting `cli_args.py` function-size scan:

- `add_status_session_arguments`: `11` LOC.
- Largest helper: `_add_checkpoint_correction_arguments`, `82` LOC.
- Other helpers are `4` to `76` LOC.

Remaining simplification targets after this pass:

- `experiments/cc_status/cli.py::main` is still the biggest status-session function
  at about `801` LOC. It should be split by command/mode dispatch later, but that is a
  higher-risk behavior refactor than the parser cleanup.
- `experiments/cc_status/reports.py::write_markdown_report` is still about `415` LOC and
  is a good next low-to-medium-risk simplification target.
- Large classes remain in core NN code (`Agent`, replay buffers, network classes), but
  those should be split only around tested contracts because they are behavior-heavy.

Validation:

- Focused status-session tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_cc_status_session.py tests/test_cc_status_recipes.py tests/test_cc_status_corrections.py -q`
  passed with `45 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy src/ai experiments/cc_status`
  reported `Success: no issues found in 35 source files`.
- Ruff:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m ruff check experiments/cc_status/cli_args.py tests/test_cc_status_session.py`
  passed.
- CLI smoke:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python experiments/cc_status_session.py list-recipes`
  listed all available recipes successfully.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1075 passed`, coverage `77.04%`, dependency audit clean, file-size gate clean, and
  package build successful.

## CLI Dispatch Guard Debug Pass

Follow-up simplify/debug pass, with no NN training run:

- Added `STATUS_SESSION_MODES` in `experiments/cc_status/cli_args.py` so the supported
  mode surface is inspectable and not hidden inside one argparse call.
- Added a focused test that parses `experiments/cc_status/cli.py` and verifies every
  parser mode has a matching explicit `opts.mode == ...` dispatch branch.
- Made `baseline-and-transfer` an explicit branch instead of relying on the final `else`
  fallback.
- Added an unreachable-mode assertion after the explicit dispatch list so future parser
  or dispatch drift fails directly instead of silently running baseline+transfer.
- Extracted non-branching `cli.py` mechanics into helpers: line buffering, recipe command
  expansion, arg parsing, session payload creation, live-metrics validation requirement,
  and final artifact writing/validation.

Result:

- `experiments/cc_status/cli.py::main` dropped from about `801` LOC to about `757` LOC.
- This is intentionally a small reduction; the larger value is the new dispatch guard.
  The remaining `main` body is still the largest status-session hotspot because it owns
  the actual run-mode calls.

Remaining simplification target:

- Split `cli.py::main` by run-mode family next: checkpoint/correction modes, diagnostic
  baseline modes, mixed/reverse/archive modes, drill/bridge/transfer modes, route-demo
  modes, tutorial-demo variants, and baseline+transfer. That should reduce the largest
  function substantially, but it is higher behavior risk and should keep the new
  parser-vs-dispatch test in place.

Validation:

- Focused status tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_cc_status_session.py tests/test_cc_status_recipes.py tests/test_cc_status_artifacts.py -q`
  passed with `49 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy src/ai experiments/cc_status`
  reported `Success: no issues found in 35 source files`.
- Ruff and Black passed for touched CLI/test files.
- CLI smoke:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python experiments/cc_status_session.py list-recipes`
  listed all available recipes successfully.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1077 passed`, coverage `77.04%`, dependency audit clean, file-size gate clean, and
  package build successful.

## Next Simplification / Debug Candidate Scan

Follow-up scan after the CLI run-family split, with no NN training run and no behavior
changes:

- Repository file-size gate is now clean, so the next work should not chase a blanket
  "everything under 500 lines" rule. The useful target is bug-hiding complexity around
  training, evaluation, checkpointing, and experiment orchestration.
- Largest current functions from an AST scan:
  - `src/app/headless.py::train_vectorized`, `428` LOC.
  - `experiments/cc_status/runs_demo.py::run_tutorial_demo_bridge_finetune`, `360` LOC.
  - `experiments/cc_status/runs_demo.py::run_tutorial_demo_bc`, `320` LOC.
  - `experiments/cc_status/demo_collect.py::collect_scripted_route_demonstrations`,
    `299` LOC.
  - `experiments/cc_status/evals.py::trace_heldout_failures`, `212` LOC.
  - `experiments/cc_status/evals.py::first_objective_near_miss_eval`, `204` LOC.
- Largest NN-relevant classes remain:
  - `src/ai/agent.py::Agent`, `777` LOC.
  - `src/ai/agent_persistence.py::AgentPersistenceMixin`, `473` LOC.
  - `src/ai/replay_buffer.py`, `938` total file LOC.

Recommended order:

1. Refactor `src/app/headless.py::train_vectorized` first. It mixes vector stepping,
   episode accounting, dashboard emission, checkpointing, deterministic eval,
   early-stop rollback, exploration boost, scheduler/epsilon stepping, and terminal
   reporting in one loop. This is the highest debug value because mistakes here can
   make runs look worse or better without changing the NN itself.
2. Refactor `experiments/cc_status/runs_demo.py` second. `run_tutorial_demo_bc` and
   `run_tutorial_demo_bridge_finetune` repeat the same pattern: configure, collect
   demos, seed/BC, train, select best snapshot, run evals/diagnostics, and assemble
   summary payloads. Extracting shared "demo run result assembly" should make new
   training methods easier to add and compare.
3. Refactor `experiments/cc_status/evals.py` third. `trace_heldout_failures` and
   `first_objective_near_miss_eval` duplicate greedy eval setup, policy eval-mode
   handling, per-game loops, action counters, target-distance metrics, row writing,
   and summary writing. A shared greedy-eval context/helper would reduce diagnostic
   drift.
4. Audit `src/ai/agent_persistence.py` fourth. It has broad compatibility catches for
   old checkpoint metadata/history and replay-buffer restore. These are not obviously
   wrong, but they are a good debug target because checkpoint/resume failures can
   silently erase history or replay context.
5. Defer older game rendering/gameplay files unless a specific bug points there.
   Files like `asteroids.py`, `space_invaders.py`, and rendering mixins are large, but
   they are low leverage for Crystal Caves NN improvement.

Suggested guardrails before touching behavior-heavy files:

- Add focused tests around the exact contract before each split.
- For `train_vectorized`, test episode accounting, eval-best save/rollback,
  exploration-boost transitions, dashboard eval recording, and final-save behavior
  with small fakes instead of a real long NN run.
- For demo/eval helpers, snapshot the summary keys and artifact paths so refactors do
  not quietly break run comparison reports.
- Keep running focused tests plus full `make verify` after each meaningful batch.

## Report Writer Simplification Debug Pass

Follow-up simplify/debug pass, with no NN training run:

- Refactored `experiments/cc_status/reports.py::write_markdown_report(...)` from a
  415-line ordered report assembly block into a 6-line dispatcher.
- Split report rendering into section helpers for run summary, final eval, selected
  policy/checkpoint lines, route/checkpoint source lines, correction lines,
  transfer/demo lines, mixed training lines, failure diagnostics, source/bridge/level
  tables, and comparison output.
- Preserved the report line order so existing status artifacts and tests continue to
  represent the same evidence.
- Used the existing report-focused test suite as the behavior guard instead of changing
  report text or adding new report semantics.

Resulting `reports.py` function-size scan:

- `write_markdown_report`: `6` LOC.
- Largest new report helper: `_append_route_demo_lines`, `43` LOC.
- Existing non-report-assembly helpers remain larger:
  `summarize_trainer` at `77` LOC and `config_snapshot` at `66` LOC.

Remaining simplification targets after this pass:

- `experiments/cc_status/cli.py::main` remains the largest status-session function at
  about `801` LOC. It is the next architecture cleanup candidate, but it should be split
  carefully by mode/command groups because it directly controls execution flow.
- The largest remaining experiment functions are run-mode implementations such as
  `run_tutorial_demo_bridge_finetune`, `run_tutorial_demo_bc`,
  `collect_scripted_route_demonstrations`, and correction collection. These are behavior
  heavy and should only be split around well-tested boundaries.
- Core NN classes (`Agent`, replay buffers, network classes) remain large by class size.
  Refactor those only after identifying a contract boundary with focused tests.

Validation:

- Focused report/status tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_cc_status_reports.py tests/test_cc_status_artifacts.py tests/test_cc_status_session.py -q`
  passed with `57 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy src/ai experiments/cc_status`
  reported `Success: no issues found in 35 source files`.
- Ruff and Black check passed for `experiments/cc_status/reports.py`.
- `git diff --check` passed.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1075 passed`, coverage `77.04%`, dependency audit clean, file-size gate clean, and
  package build successful.

## CLI Run-Family Split Pass

Follow-up architecture/simplification pass, with no NN training run:

- Split `experiments/cc_status/cli.py::main` by run family instead of leaving all mode
  implementations in one dispatcher.
- Added explicit helpers for checkpoint/correction modes, baseline/diagnostic modes,
  mixed/reset modes, drill/bridge/transfer modes, first-crystal/route modes,
  tutorial-demo modes, and baseline+transfer.
- Extracted shared tutorial-demo argument assembly into `_tutorial_demo_bc_kwargs(...)`
  so future tutorial demonstration variants can be added without copying the same
  tracing, eval, checkpoint, route-demo, and BC parameters.
- Kept all mode checks as explicit `opts.mode == "..."` branches so the parser-vs-
  dispatch guard can still audit the complete mode surface.
- The guard test caught one regression during the refactor: `baseline-and-transfer`
  was briefly implemented as a negative check and disappeared from the AST-based mode
  audit. It was changed back to an explicit branch before final validation.

Result:

- `experiments/cc_status/cli.py::main` dropped from about `757` LOC after the previous
  dispatch-guard pass to `27` LOC.
- `experiments/cc_status/cli.py` is now `860` total lines, under the repository
  `1000`-line source-file gate.
- Largest remaining functions in `cli.py` are family adapters, not the dispatcher:
  `_run_tutorial_demo_mode` at `136` LOC, `_run_baseline_diagnostic_mode` at `117` LOC,
  `_run_route_mode` at `110` LOC, `_run_skill_transfer_mode` at `104` LOC, and
  `_run_mixed_reset_mode` at `100` LOC.

Remaining simplification targets after this pass:

- Do not split `cli.py` further just to reduce line counts. The dispatcher is now small,
  the file passes the size gate, and the remaining helpers are coherent run-family
  adapters.
- Next higher-value cleanup should move to behavior-heavy implementation files with
  contract tests first, especially run implementations in `experiments/cc_status` and
  core NN modules where future plug-in additions need safer boundaries.

Validation:

- Focused status-session tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_cc_status_session.py tests/test_cc_status_recipes.py tests/test_cc_status_artifacts.py -q`
  passed with `49 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy src/ai experiments/cc_status`
  reported `Success: no issues found in 35 source files`.
- Ruff:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m ruff check experiments/cc_status/cli.py tests/test_cc_status_session.py`
  passed.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1077 passed`, coverage `77.04%`, dependency audit clean, file-size gate clean, and
  package build successful.

## Headless Vectorized Training Split Pass

First cleanup item from the ordered simplification/debug list, with no NN training run:

- Refactored `src/app/headless.py::train_vectorized` from a single 428-line loop into
  a small orchestration loop plus focused helpers.
- Extracted vectorized episode completion into `_complete_vectorized_episode(...)`,
  including score/win/progress tracking, Q/loss persistence metrics, best-score save,
  target-update accounting, dashboard episode metrics, periodic checkpoint save, and
  eval trigger.
- Extracted periodic held-out eval behavior into helpers for eval-best checkpoint save,
  dashboard eval recording, early-stop rollback, exploration-boost activation, and eval
  dashboard logging.
- Extracted post-episode boost/epsilon/scheduler handling and progress logging.
- Removed the unused `episodes_completed` local from `train_vectorized`; it was updated
  but not read.

Result:

- `src/app/headless.py::train_vectorized` dropped from `428` LOC to `89` LOC.
- `src/app/headless.py` remains under the source-file-size gate at `948` total lines.
- Largest new helpers are directly testable:
  `_complete_vectorized_episode` at `65` LOC,
  `_maybe_log_vectorized_progress` at `56` LOC,
  `_handle_vectorized_eval_control_flow` at `43` LOC, and
  `_emit_vectorized_episode_metrics` at `43` LOC.

Focused tests added:

- Vectorized episode completion records score/win/progress/Q/loss/reward state, saves
  best and periodic checkpoints, updates target-update bookkeeping, resets per-env
  counters, emits dashboard metrics, and sends NN visualization.
- Periodic vectorized eval records dashboard held-out metrics and early-stops/restores
  when the eval plateau patience trips.
- Post-episode vectorized update ends exploration boost, resets epsilon, clears the
  evaluator plateau count, resumes epsilon decay, and steps the scheduler.

Validation:

- Focused lifecycle/runtime tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_main_lifecycle.py tests/test_training_runtime.py -q`
  passed with `36 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy src/app/headless.py src/app/headless_helpers.py src/app/training_runtime.py`
  reported `Success: no issues found in 3 source files`.
- Ruff:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m ruff check src/app/headless.py tests/test_main_lifecycle.py`
  passed.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1080 passed`, coverage `77.04%`, dependency audit clean, file-size gate clean, and
  package build successful.

Next target in order:

- `experiments/cc_status/runs_demo.py`, especially `run_tutorial_demo_bc` and
  `run_tutorial_demo_bridge_finetune`, remains the next highest-value architecture
  cleanup area for future plug-in training methods.

## Post-Headless Split Candidate Refresh

Follow-up simplification/debug scan after `train_vectorized` was reduced from a
428-line loop to an 89-line orchestration method.

Current largest behavior-heavy functions:

- `experiments/cc_status/runs_demo.py::run_tutorial_demo_bridge_finetune` is now the
  largest single function at `360` LOC.
- `experiments/cc_status/runs_demo.py::run_tutorial_demo_bc` is next at `320` LOC.
- `experiments/cc_status/demo_collect.py::collect_scripted_route_demonstrations` is
  `299` LOC.
- `src/app/interactive.py::run_training` is `294` LOC, but it is less central to the
  current Crystal Caves status-session path.
- `experiments/cc_status/corrections.py::collect_policy_correction_dataset` is `251`
  LOC.
- `experiments/cc_status/evals.py::trace_heldout_failures` is `212` LOC and
  `first_objective_near_miss_eval` is `204` LOC.
- `src/ai/agent_persistence.py::load` is `154` LOC with several intentionally broad
  compatibility catches.
- `src/ai/agent.py::_learn_step_internal` is `137` LOC and remains a core-learning
  risk area.

Recommended next simplification target:

- Start with `experiments/cc_status/runs_demo.py`.
- It owns the two largest remaining functions and directly affects how we add new
  training methods.
- The useful extractions are repeatable contracts, not cosmetic splits:
  demo collection/selection artifact writing, online demo-action dataset setup,
  source-training best-snapshot selection, diagnostics plus near-miss eval, selected
  checkpoint eval/diagnostics/near-miss/save, and route-demo extra payload assembly.
- Add helper-level tests with fake trainers/agents and monkeypatched eval functions
  before changing behavior. The goal is to make future demo/bridge/transfer methods
  plug in without duplicating artifact and eval plumbing.

Recommended next debug target after that:

- `experiments/cc_status/evals.py`, especially the held-out failure trace and
  near-miss eval paths.
- These diagnostics decide whether a run is actually improving. If they drift or
  silently count the wrong event, we can reject a useful method or chase a false one.
- Look for shared greedy-eval setup, action counters, target-distance metrics, and
  artifact-row construction that can be extracted and tested.

Debug targets to defer until there is a narrower failing symptom:

- `src/ai/agent_persistence.py` broad checkpoint/resume catches. These should get
  compatibility tests before being tightened because older checkpoints may rely on
  forgiving load behavior.
- `src/ai/agent.py::_learn_step_internal`. This is high leverage but high risk; any
  cleanup needs numeric regression tests around targets, n-step/PER behavior, loss
  scaling, and priority updates.

Lower-value simplification targets for now:

- Do not split `src/game/asteroids.py`, `src/game/crystal_caves.py`, or large CSS/JS
  files purely for line-count reasons. They are near the size gate, but they are not
  the current bottleneck for safer NN experimentation.
- Do not keep shrinking `experiments/cc_status/cli.py` right now. The dispatcher has
  already been split and the remaining helpers are coherent run-family adapters.

Conclusion:

- Yes, there are still worthwhile simplification/debug areas.
- The next best one is `experiments/cc_status/runs_demo.py` because it reduces the
  cost and risk of adding new training approaches.
- The next best diagnostic reliability pass is `experiments/cc_status/evals.py`
  because better training methods are only useful if the status metrics can detect
  smaller improvements early.

## Demo Run And Eval Helper Pass

Implemented the top two next areas from the post-headless refresh.

`experiments/cc_status/runs_demo.py` changes:

- Extracted `_run_source_snapshot_training(...)` so demo BC and bridge fine-tune runs
  share the same source-eval training, source-history selection, and best-weight
  selection contract.
- Extracted `_run_final_diagnostics(...)` so held-out failure traces and near-miss evals
  are launched consistently from both demo run paths.
- Extracted `_evaluate_selected_checkpoint(...)` so selected-checkpoint eval,
  diagnostics, near-miss eval, optional selected checkpoint save, and final-weight
  restoration live in one tested place.
- Added a `finally` restoration path for selected checkpoint evals. If final eval or
  diagnostics fail, the trainer is restored to the final weights instead of staying on
  the selected checkpoint policy.

Resulting size:

- `run_tutorial_demo_bridge_finetune`: `360` LOC -> `313` LOC.
- `run_tutorial_demo_bc`: `320` LOC -> `276` LOC.
- New helper contracts:
  `_evaluate_selected_checkpoint` is `69` LOC,
  `_run_source_snapshot_training` is `45` LOC,
  `_run_final_diagnostics` is `33` LOC.

`experiments/cc_status/evals.py` changes:

- Added shared `_action_labels(...)` and `_action_label(...)` helpers so diagnostic
  action labels are generated consistently.
- Added `_resolved_end_reason(...)` so near-miss eval, held-out failure tracing, and
  level-set eval use the same `won` / `timeout` / `ended` fallback logic.
- Added `_enter_greedy_agent_eval(...)` and `_restore_greedy_agent_eval(...)` so
  diagnostic evals consistently force greedy evaluation and restore epsilon/policy
  training mode afterward.
- The held-out trace path no longer assumes every agent-like object has a direct
  `policy_net.training` attribute; it uses the same tolerant policy-mode handling as
  the near-miss eval path.

Resulting size:

- `trace_heldout_failures`: `212` LOC -> `198` LOC.
- `first_objective_near_miss_eval`: `204` LOC -> `188` LOC.

Test organization:

- Added `tests/test_cc_status_helpers.py` for focused helper-contract tests instead of
  growing the broad status-session/report files.
- This kept `tests/test_cc_status_session.py` at `966` LOC and
  `tests/test_cc_status_reports.py` at `995` LOC, both under the 1000-line target.

Validation:

- Focused status-session/report/helper tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_cc_status_helpers.py tests/test_cc_status_session.py tests/test_cc_status_reports.py -q`
  passed with `57 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy experiments/cc_status/runs_demo.py experiments/cc_status/evals.py`
  reported `Success: no issues found in 2 source files`.
- Ruff:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m ruff check experiments/cc_status/runs_demo.py experiments/cc_status/evals.py tests/test_cc_status_helpers.py tests/test_cc_status_session.py tests/test_cc_status_reports.py`
  passed.
- Broad status-session tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_cc_status_*.py -q`
  passed with `87 passed`.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1085 passed`, coverage `77.04%`, dependency audit clean, file-size gate clean, and
  package build successful.

Remaining next target:

- `experiments/cc_status/demo_collect.py::collect_scripted_route_demonstrations`
  remains a good simplification target at `299` LOC. It is lower risk than core
  learning code and directly affects demo quality, close-zone labels, and future
  training data generation.
- `src/ai/agent_persistence.py::load` remains the next debug-hardening target after
  demo collection, but should be tightened only with compatibility tests for older
  checkpoints.

## Demo Collection Split Pass

Implemented the next remaining simplification target:
`experiments/cc_status/demo_collect.py::collect_scripted_route_demonstrations`.

Changes:

- Extracted `_validate_route_demo_collection_args(...)` for demo collection argument
  validation.
- Extracted `_collect_route_demo_attempt(...)` so one scripted attempt owns reset,
  action selection, target-distance tracking, close-zone transition capture, oracle
  close-zone relabeling, and transition recording.
- Extracted `_route_demo_attempt_row(...)` so per-attempt diagnostic rows are built in
  one place.
- Extracted `_route_demo_summary(...)` so collection orchestration does not also own
  aggregate metrics.
- Extracted `_route_demo_action(...)`, `_route_demo_end_reason(...)`, and
  `_maybe_oracle_close_zone_label(...)` as small helper contracts.

Reliability fix:

- `mean_failed_min_target_distance_tiles` now returns `0.0` when failed attempts have
  no target-distance samples. Previously this path could produce `nan` because failed
  rows existed but all filtered distance values were absent.

Resulting size:

- `collect_scripted_route_demonstrations`: `299` LOC -> `75` LOC.
- New helper contracts:
  `_collect_route_demo_attempt` is `117` LOC,
  `_route_demo_summary` is `86` LOC,
  `_route_demo_attempt_row` is `82` LOC,
  and the remaining new helpers are `24` LOC or smaller.

Tests added:

- Bad route-demo controller variants are rejected by the validation helper.
- Summary aggregation returns `0.0` rather than `nan` when failed attempts have no
  target-distance samples.
- Oracle close-zone relabel helper records relabel counts, action counts, scores, and
  synthetic label transitions.

Validation:

- Focused helper/session tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_cc_status_helpers.py tests/test_cc_status_session.py -q`
  passed with `41 passed`.
- Broad status-session tests after formatting:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_cc_status_*.py -q`
  passed with `90 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy experiments/cc_status/demo_collect.py`
  reported `Success: no issues found in 1 source file`.
- Ruff:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m ruff check experiments/cc_status/demo_collect.py tests/test_cc_status_helpers.py`
  passed.
- Black formatting was applied to `experiments/cc_status/demo_collect.py`.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1088 passed`, coverage `77.04%`, dependency audit clean, file-size gate clean, and
  package build successful.

Remaining next target:

- `src/ai/agent_persistence.py::load` is now the next debug-hardening target, with
  compatibility tests first. It has broad checkpoint/resume catches that are probably
  intentional for old checkpoints, so the next pass should characterize expected
  fallback behavior before tightening anything.

## Agent Persistence Load Split Pass

Implemented the next debug-hardening target:
`src/ai/agent_persistence.py::load`.

Compatibility tests added first:

- Legacy checkpoints without `_learn_step` and `_next_target_update` still load, with
  `_learn_step` defaulting to `0` and `_next_target_update` defaulting to
  `steps + TARGET_UPDATE`.
- Bad optional checkpoint sections do not block core model restoration:
  malformed `metadata`, malformed `training_history`, and an incompatible
  `replay_buffer` are ignored while weights, epsilon, step counters, and target-update
  counters are restored.

Refactor:

- Extracted `_load_checkpoint_payload(...)` for the restricted-loader plus trusted
  compatibility fallback path.
- Extracted `_checkpoint_architecture_sizes(...)`,
  `_checkpoint_architecture_matches(...)`, and `_print_architecture_mismatch(...)`.
- Extracted `_restore_core_checkpoint_state(...)` for policy/target/optimizer/counter
  restoration.
- Extracted `_load_checkpoint_metadata(...)`, `_load_training_history(...)`, and
  `_load_replay_buffer(...)` for optional sections.
- Extracted `_format_saved_time(...)` and `_print_resume_summary(...)` so resume
  reporting no longer dominates `load`.

Resulting size:

- `load`: `154` LOC -> `40` LOC.
- New helper contracts:
  `_print_resume_summary` is `54` LOC,
  `_restore_core_checkpoint_state` is `17` LOC,
  `_load_checkpoint_payload` is `11` LOC,
  and the rest are `10` LOC or smaller.

Validation:

- Focused persistence/agent tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_agent_persistence_contracts.py tests/test_agent.py -q`
  passed with `59 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy src/ai`
  reported `Success: no issues found in 12 source files`.
- Ruff:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m ruff check src/ai tests/test_agent_persistence_contracts.py tests/test_agent.py`
  passed.
- Black formatting was applied to `src/ai/agent_persistence.py`.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python`
  exited `0`; the gate reached dependency audit and package build successfully.

Remaining next target:

- `src/ai/agent_persistence.py::save` is now the largest remaining persistence method
  at `142` LOC. It should get the same treatment: characterize save metadata/replay
  behavior first, then split checkpoint assembly, state-dict portability handling, file
  verification, and user-facing save reporting.

## Agent Persistence Save Split Pass

Implemented the next debug-hardening target:
`src/ai/agent_persistence.py::save`.

Compatibility tests added first:

- Default saves remain lightweight: they include policy/target/optimizer state,
  epsilon, step counters, architecture sizes, and metadata, but omit
  `training_history` and `replay_buffer` unless explicitly requested.
- Explicit full saves still include dashboard `training_history` and serialized replay
  state when `save_replay_buffer=True` and the replay buffer has experiences.

Refactor:

- Extracted `_ensure_checkpoint_dir(...)` and `_training_time_seconds(...)` for setup.
- Extracted `_build_save_metadata(...)` for the metadata snapshot.
- Extracted `_portable_state_dict(...)` so compiled-model portability is isolated.
- Extracted `_build_checkpoint_payload(...)` for the stable checkpoint schema.
- Extracted `_attach_replay_buffer(...)` and `_replay_buffer_size_mb(...)`.
- Extracted `_write_checkpoint_file(...)`, `_verify_saved_checkpoint(...)`,
  `_save_reason_emoji(...)`, and `_print_save_summary(...)` for write/verify/reporting.

Resulting size:

- `save`: `142` LOC -> `48` LOC.
- New helper contracts:
  `_write_checkpoint_file` is `39` LOC,
  `_build_save_metadata` is `31` LOC,
  `_print_save_summary` is `26` LOC,
  and the remaining save helpers are `20` LOC or smaller.

Validation:

- Focused persistence/agent tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_agent_persistence_contracts.py tests/test_agent.py -q`
  passed with `61 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy src/ai`
  reported `Success: no issues found in 12 source files`.
- Ruff:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m ruff check src/ai tests/test_agent_persistence_contracts.py tests/test_agent.py`
  passed.
- Black formatting was checked for `src/ai/agent_persistence.py` and
  `tests/test_agent_persistence_contracts.py`.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1092 passed`, coverage `77.16%`, dashboard smoke clean, dependency audit clean,
  file-size gate clean, and package build successful.

Remaining next target:

- `src/ai/agent_persistence.py::load_weights_only` is now the next persistence method
  worth simplifying. It is only `47` LOC, so this should be a targeted cleanup:
  reuse the same checkpoint payload loading and architecture-check helpers where doing
  so keeps behavior clear, then characterize missing-file, bad-architecture, and bad
  state-dict paths.

## Agent Persistence Weight-Only Load Pass

Implemented the next targeted persistence cleanup:
`src/ai/agent_persistence.py::load_weights_only`.

Compatibility tests added first:

- Weight-only loads reject architecture-mismatched checkpoints and preserve live
  epsilon/step counters.
- Weight-only loads reject malformed policy/target state dictionaries and preserve live
  epsilon/step counters.

Refactor:

- Extended `_load_checkpoint_payload(...)` with optional quiet/error-prefix handling so
  full load and weight-only load share the same restricted-loader/trusted-fallback path.
- Extracted `_restore_checkpoint_network_weights(...)` for policy/target state-dict
  adaptation and restoration.
- Updated `_restore_core_checkpoint_state(...)` to call the shared network restore
  helper before restoring optimizer and training counters.
- Updated `load_weights_only(...)` to reuse `_load_checkpoint_payload(...)`,
  `_checkpoint_architecture_matches(...)`, and `_restore_checkpoint_network_weights(...)`.

Resulting size:

- `load_weights_only`: `47` LOC -> `34` LOC.
- Shared helper contracts:
  `_load_checkpoint_payload` is `18` LOC,
  `_restore_checkpoint_network_weights` is `10` LOC,
  and `_restore_core_checkpoint_state` is `9` LOC.

Validation:

- Focused persistence/agent tests:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m pytest tests/test_agent_persistence_contracts.py tests/test_agent.py -q`
  passed with `63 passed`.
- Direct mypy:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m mypy src/ai`
  reported `Success: no issues found in 12 source files`.
- Ruff:
  `/Users/justin/.pyenv/versions/3.12.11/bin/python -m ruff check src/ai tests/test_agent_persistence_contracts.py tests/test_agent.py`
  passed.
- Black formatting was applied to `src/ai/agent_persistence.py`.
- Full verification:
  `make verify PYTHON=/Users/justin/.pyenv/versions/3.12.11/bin/python` passed with
  `1094 passed`, coverage `77.21%`, dashboard smoke clean, dependency audit clean,
  file-size gate clean, and package build successful.

Remaining next target:

- `src/ai/agent_persistence.py::inspect_model` is the next persistence method worth a
  focused cleanup. It is `47` LOC and reads checkpoint metadata through a separate
  static path, so the next pass should first characterize missing-file, unreadable-file,
  and metadata-present cases before splitting file-stat collection from info assembly.
