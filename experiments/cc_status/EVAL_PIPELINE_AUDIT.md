# Eval/scoring pipeline audit — findings and fixes (2026-07)

Two independent audit passes (metric math; eval hygiene) over `diagnose_gap.py`,
`aggregate_diag.py`, `lever_ab.py`, `paired_ab.py` and the engine's eval hooks,
plus a hand check of imported-mode support. Fixes shipped in the same commit as
this document. **Protocol note: runs started after this commit are not
schedule-identical to RUN-01..22** (in-training eval decoupled, same-frame
precedence change, unrounded fractions) — treat cross-era comparisons of small
deltas with care.

## Fixed

| # | Finding | Fix |
|---|---|---|
| 1 | `use_eval_levels` was a silent NO-OP for authored/imported sets: "eval" sampled levels randomly WITH replacement, `reset_eval_cursor` did nothing, and training-only curriculum starts stayed active during eval | Authored/imported sets now enter true eval mode and cycle their fixed CAVES deterministically; eval-mode gate then protects curriculum starts. Stale-cache guard added (`_eval_source`) so train↔eval switches can't keep the wrong caves |
| 2 | No versioned pipeline supported `CRYSTAL_CAVES_IMPORTED` — official RUN-20/21/22 numbers came from an ad-hoc runner living only on the M4 | `diagnose_gap --imported` + `make_config(imported=True)`: the canonical pipeline now trains/evals the fixed 16-level set; report notes the train-vs-test gap does not apply |
| 3 | In-training Evaluator ran on held-out seeds (superset of the test split) and steered training (plateau epsilon boost, early-stop, keep-best) — the holdout doubled as a control signal | Diagnosis configs set `EVAL_EVERY=0`, `DISABLE_EXPLORATION_BOOST=True` |
| 4 | Milestone evals consumed global `np.random` draws, so `--checkpoint-every` changed the trained policy itself | Eval blocks snapshot/restore the RNG state |
| 5 | FAR leg-2 probe placement drew from wherever training left the global RNG — irreproducible from a checkpoint | Probe placement seeded deterministically per game index |
| 6 | Missing metric keys silently scored 0.0 (pipeline break reads as a bad agent) | `_aggregate` excludes missing rows from means and reports `missing_<metric>` loudly |
| 7 | `aggregate_diag` FINAL bucket unguarded against ragged seeds; `or curve_avg` fallback silently re-admitted single-seed buckets; run-param mismatches across per-seed files unchecked | Final bucket now full-seed-guarded with a warning; fallback selects max-n_seeds buckets loudly; homogeneity warnings |
| 8 | Best-checkpoint win rates printed with no N or interval — 0.021 is ONE episode of 48; argmax selection adds winner's curse | Wilson 95% CI + N printed at the best-checkpoint line, plus an explicit winner's-curse note |
| 9 | `lever_ab` swallowed crashed (arm, seed) runs (`except: return []`) — silently shrank paired N | Full traceback printed; failures recorded in `summary.json["failed_runs"]` and warned |
| 10 | Same-frame exit+fatal-hit scored as a death, while first-crystal-goal+death scored as a win — inconsistent precedence biased the taxonomy | `_check_exit` now runs before `_check_player_danger`: a same-frame win is a win, consistently |
| 11 | `crystal_frac`/`depth_frac` rounded to 3 decimals per episode BEFORE downstream averaging; the rounded value was the best-checkpoint tiebreaker | Unrounded values emitted; rounding stays display-only |
| 12 | Zero-crystal cave pinned `crystal_frac` to 0 forever (max(1,..) denominator) | Vacuously 1.0 (exit is open from the start); mirrored in the trace helpers |
| 13 | Death-trace dropped unknown end reasons silently | All observed reasons emitted alongside the standard taxonomy |
| 14 | Results labelled "IQM" were plain means (leftover naming from the fixed flooring bug) | Keys/headers renamed `mean`; tests updated |

## Verified correct (no action)

Greedy eval is genuinely deterministic (epsilon=0, NoisyNet mean weights, no
per-step noise); eval episodes never enter the replay buffer or epsilon
schedules; procedural train/held-out/regenerate seed ranges are disjoint;
`--record-play` GIFs use the identical scored-eval code path; the historical
IQM flooring bug is fixed (means everywhere, regression-tested); `won` cannot
be set by timeouts/stalls; lever_ab CIs are seed-clustered bootstrap.

## Known limitations (documented, not code-fixed)

- Best-checkpoint TRAIN numbers remain selection-biased (argmax over the same
  episodes); the printed CI bounds chance, not selection. Treat 1-2 win blips
  as noise until repeated.
- Per-level rates at a checkpoint have N=3 (one greedy episode per seed) — the
  per-level tables are qualitative, not statistical.
- For procedural runs, "level_index" across seeds refers to DIFFERENT caves
  (per-seed eval sets); never aggregate per-level across seeds there. Imported
  runs are exempt (fixed set).
- `CrystalCaves.__init__` consumes one RNG draw via its initial `reset()`;
  eval-game construction inside milestone evals is now RNG-isolated, but other
  ad-hoc constructions should copy that pattern.
