# Crystal Caves NN Experiment Tracker

**Last updated:** 2026-06-24
**Purpose:** tracked handoff for the next Crystal Caves DQN improvement sessions.
Use this alongside `CC_NN_HANDOFF.md` and the per-run artifacts under
`.Codex/artifacts/cc_sessions/`.

## Decision Rule

Judge changes on held-out greedy evaluation and trace diagnostics, not training-only
score. For 150-episode probes, a candidate is worth promoting only if it improves at
least one stable held-out signal without breaking the others:

- crystal collection rate
- depth reached
- trace target-distance improvement
- tile-loop / idle / invalid-action rates
- end reasons shifting away from stalls/timeouts

Held-out wins remain the real goal, but short single-seed probes are too noisy to use
as the only gate.

## Current Baselines

| Run | Artifact | Wins | Crystals | Depth | Notes |
|---|---|---:|---:|---:|---|
| diagnostic baseline 150 | `.Codex/artifacts/cc_sessions/20260623_050020_diagnostic_baseline_150` | 0/8 | 12.5% | 29.5% | trace: 0/4 any crystal, heavy tile loops |
| anti-loop 150 | `.Codex/artifacts/cc_sessions/20260623_052806_anti_loop_150` | 0/8 | 25.0% | 33.0% | weak positive; trace loop share improved, still no traced crystals |
| first-crystal transfer 150+150 | `.Codex/artifacts/cc_sessions/20260623_063645_first_crystal_best_transfer_150_150` | 0/8 | 12.5% | 14.3% | not a keeper; transfer underperformed anti-loop |
| B3g tutorial demo BC | `.Codex/artifacts/cc_sessions/20260623_212528_tutorial_demo_bc_pool512_select30_300_nearmiss` | 7/30 | 23.3% | 36.2% | previous first-crystal route baseline |
| B3l bridge interleave 12.5% | `.Codex/artifacts/cc_sessions/20260624_051947_bridge_route_interleaved_125_pool512_select30_300` | 6/30 | 20.0% | 26.0% | better close-zone action shape, worse route approach |
| B3m route then bridge fine-tune | `.Codex/artifacts/cc_sessions/20260624_060412_tutorial_demo_bridge_ft100_b125_pool512_select30` | 7/30 | 23.3% | 16.4% | tied B3g wins but lost too much depth; not promoted |
| B3n demo BC + invalid shoot | `.Codex/artifacts/cc_sessions/20260624_064246_tutorial_demo_bc_invalid_shoot_pool512_select30_300` | 5/30 | 16.7% | 38.1% | reduced some shoot spam but hurt route success; not promoted |
| B3o recovery route demos | `.Codex/artifacts/cc_sessions/20260624_082654_tutorial_demo_bc_recovery_pool512_select30_300` | 7/30 | 23.3% | 44.8% | tied B3g wins with better depth/near-miss/stuck, but worse close-zone loops; conditional data-quality keeper only |
| B3p planner route demos | `.Codex/artifacts/cc_sessions/20260624_090216_tutorial_demo_bc_beam_pool512_select30_300` | 4/30 | 13.3% | 36.4% | much better demo coverage, worse selected policy; not promoted |
| B3r filtered/weighted demos | `.Codex/artifacts/cc_sessions/20260624_111158_tutorial_demo_bc_filtered_weighted_pool512_select30_300` | 5/30 | 16.7% | 33.6% | filters weak beam demos and trains better on-source, but still trails B3g/B3o; not promoted |
| B3s conservative demo-Q | `.Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300` | 10/30 | 33.3% | 60.5% | **new route baseline**; validated separately at 19/60 |
| B3t conservative + close-zone extra | `.Codex/artifacts/cc_sessions/20260624_125257_tutorial_demo_conservative_close_zone_pool512_select30_300` | 11/30 | 36.7% | 54.0% | selected eval beat B3s by +1, but 60-game validation fell to 18/60; not promoted over B3s |
| B3u oracle close-zone labels | `.Codex/artifacts/cc_sessions/20260624_145614_tutorial_demo_oracle_close_zone_pool512_select30_300` | 10/30 | 33.3% | 55.2% | oracle labels learned cleanly but only tied B3s and lost depth/loop profile; not promoted |

## Ranked Backlog

### A1. Contextual Invalid-Action Pressure

**Status:** tested; not promoted.

**Why this is next:** recent held-out traces show wasted `INTERACT`/idle loops. RL
literature on action elimination and invalid-action masking supports pruning or
penalizing actions that are impossible in the current context. The narrowest test is
environment-side pressure on useless `INTERACT`, because it is easy to isolate and does
not require changing the DQN target calculation.

**Implementation plan:**

- Add opt-in `CRYSTAL_CAVES_INVALID_INTERACT_PENALTY`.
- Penalize `INTERACT` only when no unused adjacent switch can be thrown.
- Record `invalid_interact_count`, `invalid_interact_penalty_total`, and
  `interact_action_frac` in trace diagnostics.
- Run a comparable 150-episode probe with the status-session runner.

**Promotion rule:** promote if held-out crystals/depth match or beat anti-loop and
trace `INTERACT`/tile-loop rates drop. If it only reduces `INTERACT` but harms
navigation, keep the telemetry and do not combine it by default.

**References:** Action Elimination with Deep Reinforcement Learning
(`https://arxiv.org/abs/1809.02121`), Invalid Action Masking in Policy Gradient
Algorithms (`https://arxiv.org/abs/2006.14171`).

### A2. Demo Replay / DQfD-Lite From Oracle Routes

**Status:** tested as bridge-demo replay; not promoted.

**Why:** the agent can learn isolated drills but still fails transfer. A small stream of
oracle transitions could give the replay buffer rare successful prefixes without
changing the game objective.

**Lowest-risk test:** seed replay before full-cave training. A direct full-cave scripted
first-crystal demonstrator was tried first, but it collected only 2/16 crystals in a
throwaway check, so it was not reliable enough. The safer A2 implementation is
`bridge-demo-replay`: train/select a bridge source policy, roll out successful bridge
trajectories, seed those into a fresh full-cave trainer's replay buffer, and do not copy
bridge weights.

**Metric to watch:** first-crystal held-out rate, target-distance best delta, and whether
the greedy trace starts reaching the first objective earlier than baseline.

**Reference:** Deep Q-learning from Demonstrations (`https://arxiv.org/abs/1704.03732`).

### A3. Reverse-Curriculum Starts Near the Objective

**Status:** tested as `reverse-start`; not promoted.

**Why:** the full task has sparse successful endings. Starting episodes near the final
crystal/exit can teach the backward chain from completion to collection.

**Lowest-risk test:** add an experiment-only reset mode that starts the player near the
current objective for a minority of vector lanes, while held-out eval remains full
tutorial caves. The implemented probe uses 6 normal tutorial lanes and 2 reverse lanes
at `vec-envs=8`: one `reverse_objective` lane and one `reverse_exit` lane.

**Metric to watch:** exit-unlocked traces and completion after all crystals are collected.

**Reference:** Reverse Curriculum Generation for Reinforcement Learning
(`https://arxiv.org/abs/1707.05300`).

### A4. Archive / Go-Explore-Lite

**Status:** tested as `archive-start`; not promoted.

**Why:** traces repeatedly collapse into local loops. A simple archive over tile/objective
milestones could force coverage of useful states before policy learning takes over.

**Lowest-risk test:** keep a small archive of reached `(objective kind, coarse region,
crystal count, depth bucket)` states from normal full-cave lanes and restart a minority
of training lanes from deep-copied snapshots. Held-out eval remains normal full tutorial
caves. The initial implementation is experiment-only in `experiments/cc_status_session.py`
as `archive-start`: 6 full lanes, 2 archive lanes, 70% archive replay probability, max
64 states, and minimum 30 steps before a full-lane state can be archived.

**Metric to watch:** whether the archive fills/replays during the run, then held-out
crystals/depth and trace loop/target-distance signals versus anti-loop 150. If the
archive never fills, the implementation needs a looser milestone rule before judging
the idea.

**Reference:** Go-Explore (`https://arxiv.org/abs/1901.10995`).

### A5. Novelty Bonus

**Status:** tested as `novelty-bonus`; not promoted.

**Why:** anti-loop was weakly positive, suggesting exploration pressure helps. A tile or
coarse-region novelty bonus is simpler than RND and easier to inspect.

**Lowest-risk test:** one-time per-region bonus for first visit while an objective is
active, capped below crystal reward. The implemented probe adds opt-in
`CRYSTAL_CAVES_NOVELTY_BONUS`: `+0.08` per new global-map cell, total cap `3.0`, with
the spawn cell pre-marked so the agent only gets paid for moving into new regions.
Compare against anti-loop, not against baseline only.

**References:** Random Network Distillation (`https://arxiv.org/abs/1810.12894`),
Curiosity-driven Exploration (`https://arxiv.org/abs/1705.05363`).

### A6. HER-Lite Objective Relabeling

**Status:** queued.

**Why:** failed episodes still contain useful partial successes like reaching a region,
crystal, or switch. Hindsight relabeling can convert those into training signal.

**Risk:** more invasive for this DQN because the current state does not cleanly encode a
goal vector independent of environment state.

**Reference:** Hindsight Experience Replay (`https://arxiv.org/abs/1707.01495`).

### A7. Hierarchical Options

**Status:** queued.

**Why:** "go to crystal", "throw switch", and "go to exit" are natural sub-policies.

**Risk:** high implementation surface and evaluation complexity. Do this only after
A1-A4 fail to produce a stronger held-out route signal.

**References:** h-DQN (`https://arxiv.org/abs/1604.06057`), Option-Critic
(`https://arxiv.org/abs/1609.05140`).

### A8. Distributional/Rainbow DQN Upgrade

**Status:** queued.

**Why:** the task has rare high-value outcomes and unstable Q estimates.

**Risk:** unlikely to fix the navigation curriculum wall by itself. Treat as a later
agent-quality upgrade after the environment/curriculum probes produce a route signal.

**References:** Rainbow (`https://cdn.aaai.org/ojs/11796/11796-13-15324-1-2-20201228.pdf`),
C51 (`https://arxiv.org/pdf/1707.06887`).


## Detailed History Archives

The long B-series run notes and findings log were split into smaller archive files so
future sessions can read the relevant section without reopening a 3,000+ line tracker.
No findings were deleted.

- `docs/cc_nn_experiment_tracker/b_series_route_mastery.md` - B1 through B3h route-mastery plan and results.
- `docs/cc_nn_experiment_tracker/findings_2026_06_23_part1.md` - A-series findings, metrics review, and rerun decision.
- `docs/cc_nn_experiment_tracker/findings_2026_06_24_part1.md` - B3l through B3q follow-up runs and evaluation/demo-selection work.
- `docs/cc_nn_experiment_tracker/findings_2026_06_24_part2.md` - B3r through B3u filtered demos, conservative demo-Q, and close-zone label findings.

## Cleanup Note

The active baseline remains B3s conservative demo-Q unless a future run beats it on the
same selected-checkpoint and expanded held-out validation protocol. B3t and B3u are
archived as non-promoted close-zone variants because they did not improve the expanded
validation outcome enough to replace B3s.

## Extension Workflow

Future NN additions should use `CC_NN_EXTENSION_ARCHITECTURE.md`:

- Add optional losses through `src/ai/extension_contracts.py` contributions instead of
  growing bespoke `Agent.learn` branches.
- Add status-session smoke/comparison recipes in `experiments/cc_status/recipes.py`.
- Run them through `python experiments/cc_status_session.py run-recipe <key>` so command
  shape stays comparable and overrideable.
- Trust only runs whose `artifact_validation.json` passes.
- Compare against the B3s recipe before promoting a new route-control method.
- Run `python experiments/cc_status_session.py compare-artifact <candidate> [--validation <artifact>]`
  and promote only on a `PROMOTE` decision. Treat `HOLD` as useful evidence that still
  needs more validation; treat `REGRESS` as archived unless a specific bug invalidated
  the run.

2026-06-24 reliability smoke: `run-recipe b3s_conservative_smoke --label
codex_recipe_validator_smoke2` passed end-to-end and produced
`.Codex/artifacts/cc_sessions/20260624_185336_codex_recipe_validator_smoke2/artifact_validation.json`
with `ok: true`.

2026-06-24 architecture update: `compare-artifact` now uses the frozen B3s selected and
expanded-validation bar, compares win rates when sample sizes differ, and reports support
metric improvements/regressions for close-zone and near-miss behavior.

2026-06-24 correction architecture update: added `collect-corrections` status-session
mode for DAgger-style policy-visited state labels. It writes `correction_examples.npz`,
per-state JSONL, summary/report lines, and passes artifact validation. Smoke artifact:
`.Codex/artifacts/cc_sessions/20260624_191511_codex_correction_collector_keep_smoke`
(`2` kept loop-trigger states, shape `(2, 295)`, validation `ok: true`). Default future
datasets should keep only disagreements; `--correction-keep-agreements` is for smoke and
debug only.

2026-06-24 correction fine-tune architecture update: added `correction-finetune`
status-session mode and `Agent.set_correction_action_dataset(...)`. Correction datasets
now train through the generic auxiliary-loss contribution path with report/live metrics:
`avg_correction_action_loss_100` and `avg_correction_action_accuracy_100`. Smoke
artifact:
`.Codex/artifacts/cc_sessions/20260624_192405_codex_correction_finetune_smoke`
validated `ok: true` and proved the loss path executed on the tiny two-state correction
dataset (`loss 0.0106`, `100%` action accuracy). This smoke is not improvement evidence;
the next real test needs a larger disagreement-only correction dataset from the B3s
selected checkpoint, then a comparable `correction-finetune` run and `compare-artifact`
decision against B3s.

2026-06-24 correction architecture hardening update: added an external auxiliary-loss
provider registry via `Agent.register_auxiliary_loss_provider(...)`, made
`correction-finetune` fail on empty correction datasets, added correction-training
artifact validation, and added `correction_action_samples_100` so live/final metrics can
distinguish active zero-loss supervision from missing supervision. Hardening smoke
artifact:
`.Codex/artifacts/cc_sessions/20260624_193530_codex_correction_arch_harden_smoke`
validated `ok: true` with `2` correction states, `100` sampled correction loss updates,
`loss 0.0106`, and `100%` correction action accuracy.

2026-06-24 non-run architecture update: added guarded status-session recipes for the
next correction workflow:

- `b3s_correction_collect` expands to `collect-corrections` and requires an explicit
  `--checkpoint` override.
- `b3s_correction_finetune` expands to `correction-finetune` and requires explicit
  `--checkpoint` plus `--correction-dataset` overrides.

This makes the next correction experiment repeatable by name without launching any new
training during the architecture pass. `list-recipes` now exposes the required inputs.

Same pass also extracted reusable supervised action-margin loss math into
`src/ai/action_margin_loss.py` and rewired the existing demo/close-zone/correction mixin
path through it. This is an organization change only; focused tests covered the helper
and the existing agent experiment hooks.
