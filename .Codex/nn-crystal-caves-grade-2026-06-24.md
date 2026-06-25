# Crystal Caves NN Focused Grade - 2026-06-24

## Scope

This is a focused grade of the current Crystal Caves neural-network path and the
Crystal Caves game/environment. It does not replace `.Codex/grade-report.md`; it
narrows the review to the parts that matter for improving Crystal Caves learning.

Evidence used:

- Current code inspection of `src/ai`, `src/game/crystal_caves*`, and
  `experiments/cc_status`.
- Current promotion gate baseline in `experiments/cc_status/promotion.py`.
- Current experiment tracker and cleanup docs.
- Latest recorded full verification from this work stream: `1074 passed`,
  coverage `77.04%`, mypy/ruff/black/file-size/dependency/package checks clean.
- Latest correction architecture smoke:
  `.Codex/artifacts/cc_sessions/20260624_193530_codex_correction_arch_harden_smoke`
  validated `ok: true` with `2` correction states, `100` sampled correction-loss
  updates, `0.0106` avg correction loss, and `100%` correction action accuracy.

No new long training session was run for this grade. The current learning grade is
based on saved comparable artifacts because another single short run would add more
noise than evidence.

## Executive Grade

| Area | Grade | Meaning |
|---|---:|---|
| NN platform / architecture | **B+** | The code can support new methods, replay variants, auxiliary losses, recipes, metrics, and artifact validation. |
| Current learned NN gameplay outcome | **C-** | The current best promoted route policy is meaningfully above old zero-signal baselines, but it is still not solving final Crystal Caves. |
| Crystal Caves game/environment | **B+** | The environment is feature-rich, tested, and much more learnable than before, but reward/state/generator complexity still makes NN progress hard to interpret. |
| Overall Crystal Caves NN readiness | **B** | Good infrastructure; incomplete agent competence. The next gains are more likely from targeted correction/curriculum loops than from broad reward tweaking. |

## NN Grade

### Strengths

- The agent has modern DQN pieces: spatial state support, dueling heads, NoisyLinear
  exploration, n-step returns, and a combined prioritized n-step replay buffer.
- The previous PER+n-step silent downgrade has been addressed by
  `PrioritizedNStepReplayBuffer`, so config intent and runtime behavior now align.
- New extension architecture is moving in the right direction:
  `AuxiliaryLossProvider` plus `Agent.register_auxiliary_loss_provider(...)` lets new
  experimental losses plug in without editing the core optimize loop.
- `experiments/cc_status` is now a real comparable-run harness: named recipes, B3s
  frozen baseline, promotion decisions, report generation, artifact validation, live
  metrics, selected checkpoints, traces, and correction datasets.
- The current baseline is no longer a vague "maybe": B3s is frozen as
  `10/30` selected first-crystal wins and `19/60` expanded validation wins, with
  supporting depth/near-miss/stuck/loop metrics.
- Tests cover the newer reliability gates: recipes, promotion, reports, artifacts,
  correction collection/fine-tune validation, replay, and agent behavior.

### Weaknesses

- The best promoted policy is still a route/first-crystal baseline, not a full-level
  completion policy. It improves signal, but it does not prove the NN can clear final
  Crystal Caves levels.
- The method surface is easier to extend than before, but not fully plug-and-play yet:
  built-in route/demo/close-zone/correction losses still live partly in agent mixins and
  experiment runners instead of first-class method modules.
- Recipes now cement B3s and expose guarded correction collect/fine-tune recipes, but
  there is still no promoted correction-finetune result. The workflow is repeatable;
  the method still needs evidence.
- Metrics are much better, but short-run win rates remain noisy. Promotion protects
  against obvious false positives, yet we still need better multi-seed/statistical
  confidence before calling small differences real.
- Several files are still dense: `src/ai/replay_buffer.py` ~938 LOC,
  `src/ai/agent.py` ~865 LOC, `src/ai/network.py` ~736 LOC,
  and multiple status runner modules are large. This is acceptable under the current
  1000 LOC cap but still slows review.

### NN Scorecard

| Dimension | Grade | Notes |
|---|---:|---|
| Core DQN implementation | **B+** | Strong feature set; still DQN-limited for sparse long-horizon platforming. |
| Replay correctness | **A-** | PER+n-step is now explicit and tested; keep watching truncated-tail assumptions. |
| Extension architecture | **B** | Provider hook is good; next step is method modules/registries and recipe templates. |
| Experiment reproducibility | **A-** | Recipes, artifacts, promotion gates, and reports are strong. |
| Measurement detail | **B+** | Near-miss and correction metrics help; multi-seed confidence and failure taxonomy can still improve. |
| Current model performance | **C-** | Better first-crystal routing, still not enough completion behavior. |

## Crystal Caves Grade

### Strengths

- The game is a substantial clean-room Crystal Caves-like platformer with crystals,
  exit unlock, switches/doors, hazards, enemies, ammo, elevators, jumping, falling, and
  staged difficulty.
- The environment has been split into focused modules:
  `crystal_caves.py`, `crystal_caves_logic.py`, `crystal_caves_rendering.py`,
  `crystal_caves_dressing.py`, `crystal_caves_geometry.py`,
  `crystal_caves_gen.py`, and drill/bridge level modules.
- State is richer than the old local-only window: 19x11 local tile view,
  11x6 global objective map, and metadata for player/objective/physics context.
- The generator has jump-aware solvability, keyed-door reachability, walk-coverage
  floors for training tiers, and a normal-level quality rubric.
- Training levels are allowed and are now treated as scaffolding rather than forced
  final-game simplification. This matches the current strategy: train on helpful levels,
  keep final levels close to Crystal Caves 1991.
- The environment exposes progress components and richer info, which gives the runner
  more ways to diagnose "almost progress" before full wins appear.

### Weaknesses

- `src/game/crystal_caves.py` is still near the file-size ceiling at ~979 LOC, and
  `crystal_caves_rendering.py`, `crystal_caves_art.py`, and `crystal_caves_gen.py` are
  also large. This is manageable but still makes behavior audits harder.
- Reward shaping is dense and hand-tuned. That is understandable for sparse platformer
  RL, but it also makes it hard to know whether a new improvement is policy learning or
  reward exploitation.
- Training scaffolds, bridge levels, drills, and final CC-like levels all share much of
  the same runtime path. That is useful, but the distinction should remain explicit in
  configs, docs, and reports so training-only aids do not accidentally become final-level
  assumptions.
- Level-gen quality is much better than before, but "solvable by oracle" is not the same
  as "learnable by the current NN." The generator needs an explicit learnability/fidelity
  evaluation suite, not just reachability and style scoring.
- The game is good enough for NN work, but it is still a hard long-horizon task: collect
  every crystal, infer switch/door causality, route vertically, avoid hazards, and reach
  exit after unlock. The environment is not the blocker alone; the training method has
  to teach these stages separately.

### Crystal Caves Scorecard

| Dimension | Grade | Notes |
|---|---:|---|
| Mechanics completeness | **B+** | Strong CC-like mechanics; visual/player fidelity not re-reviewed in this pass. |
| RL state design | **B+** | Local + global + metadata is much better; still may not encode enough path/action affordance. |
| Reward design | **B-** | Necessary but complex; many knobs have already failed to move outcomes. |
| Level generation | **B+** | Solvability and walk-coverage gates are strong; learnability grading is next. |
| Training/final separation | **B** | Good intent and docs; should be made even more explicit in runner reports. |
| Testability | **A-** | Broad tests and gates; add generator learnability fixtures next. |

## What This Means

We are progressing architecturally, but not yet enough on the actual learned policy.
That explains why the work feels slow: many changes improved observability,
correctness, or method hygiene, while only B3s clearly improved the policy metric we
currently trust. The right conclusion is not "nothing works"; it is "broad reward and
level tweaks are low-value now, and the next work needs to target policy-visited failure
states directly."

## Recommended Next Work

1. **Make correction-finetune the next real promoted method.**
   - Collect a larger disagreement-only correction dataset from the B3s selected
     checkpoint.
   - Inspect label mix, trigger mix, disagreement rate, and sampled states.
   - Run a B3s-comparable correction-finetune session.
   - Use `compare-artifact` against B3s before promotion.
   - Why first: it trains on the states where the current policy actually fails, unlike
     close-zone/oracle labels that can improve internal metrics without improving route
     behavior.

2. **Add a correction recipe pair.**
   - Status: addressed for the B3s correction workflow with guarded
     `b3s_correction_collect` and `b3s_correction_finetune` recipes.
   - Remaining work: run them only when ready to collect real evidence.

3. **Add a level-gen learnability/fidelity report.**
   - For each training and final tier, record solvability, walk coverage, crystal
     approach distance, vertical transitions, hazard proximity, switch necessity, and
     current-policy success/near-miss.
   - Why: this separates "good Crystal Caves level" from "good training level."

4. **Modularize NN method plugins.**
   - Move correction/demo/route auxiliary methods behind first-class provider modules
     and keep `Agent` focused on DQN optimization.
   - Why: this makes new NN ideas safer to add and easier to delete if they fail.

5. **Continue file shrink only where it reduces review risk.**
   - Priority files: `crystal_caves.py`, `crystal_caves_gen.py`, `agent.py`,
     `replay_buffer.py`, and oversized status-session modules.
   - Do not refactor purely to hit 500 LOC if it delays method validation.

## Bottom Line

The Crystal Caves environment grades well enough to keep using. The NN architecture now
grades well enough to add methods safely. The weak grade is the learned behavior itself:
the agent has better first-crystal routing but not robust level completion. The highest
value next move is a real B3s-based correction-finetune comparison, backed by named
recipes and artifact promotion, not another broad reward or level-generation tweak.
