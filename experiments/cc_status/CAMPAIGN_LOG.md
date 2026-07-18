# Crystal Caves DQN — Completion-Wall Campaign Log (RUN-26 → RUN-61)

Canonical record of the July 2026 campaign that took the agent from "exit never
unlocks" (grade F on full-level completion) to a verified from-spawn champion.
Run on the PR #39 branch (`claude/nn-performance-grading-2840b4`). The blow-by-blow
narrative lives in the PR #39 comment stream (~85 milestone comments); this file
preserves the verified results, the method, and the settled lever ledger so the
knowledge survives independently of the PR.

## Headline results

- **First verified from-spawn win** (2026-07-13, RUN-39e ep~8k): The Switchback
  Spire (level 14) — all 32 crystals + exit, under the 4,500-step fidelity clock.
- **Verified champion** —
  `experiments/cc_status/data/champions/switchback_champion_run39e_ep10000.pth`:
  **42/100** official from-spawn wins (4,500-step clock) and **11/100 under the
  original 3,000-step rules** (fastest win: 2,091 steps), measured by
  100-episode harness-faithful exams (full 16-level eval context, NoisyNet
  sampling as trained).
- **Conquest board at campaign end**: Switchback Spire ✅ conquered ·
  Stalactite Chasm 92% ladder / 0.923 exam crystal fraction · Scaffold Reactor
  89–93% ladder · Ore Shaft 64% (walled). Remaining 12 levels not yet campaigned
  (5 lack machine demos entirely).
- **Scale**: 51 run directories (RUN-26…RUN-61), ~1.06M episodes,
  **1.445 billion environment steps**, ≈30+ recorded wall-hours — about nine
  sim-months of continuous play.
- **Demo library**: machine (robot) wins banked for **11 of 16 levels** plus two
  alternate routes (`experiments/cc_status/data/demos_goexplore/`,
  per-level focus copies in `data/demos_focus_*`). Levels 5, 6, 8, 9, 10 remain
  unsolved even by the Go-Explore harvester.

## The recipe that produced wins

`experiments/cc_status/diagnose_gap.py` with:

```
--geo-compass --geo-compass-hazard-aware --ngu-bonus --enemy-motion \
--reward-clip 35 --stall-window 1440 --max-steps 4500 \
--demo-dir experiments/cc_status/data/demos_focus_<level> --demo-reset-p 0.9 \
--demo-backward --demo-backward-retreat 60 --demo-backward-wins 2 \
--demo-backward-window 240 --demo-backward-deep 600 --demo-heal \
--demo-level-bias 0.7 --vec-envs 8
```

i.e. combo perception + a Salimans & Chen (1812.03381) backward curriculum over
robot demos, with demo gradients OFF (`--demo-td-weight 0 --demo-margin-weight 0`
territory — all demo-gradient forms measured harmful or neutral).

Network: 625,313 parameters — 314 inputs, 512→512→256 trunk, dueling 128/128
heads, NoisyNets.

## The five backward-ladder mechanism bugs (all telemetry-caught)

1. Win banking read `self.won` *after* reset cleared it → snapshot
   `_prev_episode_won` as reset's first act.
2. Credit compared the previous episode's level to the newly-sampled level
   (~1/16 banking rate) → credit `prev_level` directly.
3. Per-env ladders cost 8× wins per rung → class-level shared
   `_BC_SHARED_OFFSET` / `_BC_SHARED_WINS` pooled across vec envs.
4. Uniform window sampling made exact-frontier attempts 1-in-241 → 50/50
   frontier split.
5. Deep rungs starved (long episodes, few attempts) → `--demo-backward-deep`
   easing: 1 win/rung and half retreat past the threshold.

Guard tests: `tests/test_win_at_k.py`, `tests/test_run26_prep.py` (source guards
for every lever).

## Settled lever ledger

**Dead (measured, do not retry without new evidence):**
- Demo gradients — DQfD TD and margin losses, all weights (RUN-26 family, 28).
- Win-at-K curricula, static and ramped (RUN-33/34/35).
- Spatial CNN — representation is not the bottleneck (RUN-37).
- Warm-start/resume — resumed brains (fresh Adam + empty PER buffer) never
  recover win capability (RUN-41/47 + every restart). Champions are mid-run
  dynamical artifacts: **bank checkpoints, not runs**.
- Consolidation passes — improved collection stats while *degrading* win rate
  (RUN-40…44 audit: all "training wins" were rehearsal cuts, zero frontier
  advances).

**Live (confirmed positive):**
- Fidelity clock `--max-steps 4500` (7 of 11 demo routes exceed the old 3,000
  cap; the 1991 original has no timer).
- Combo perception (NGU + enemy-motion + hazard-aware compass): best plain-RL
  config, ~0.407–0.426 crystal fraction ceiling without curriculum.
- Backward ladder (fixed as above): produced every from-spawn win.
- **Heal-on-handoff (`--demo-heal`)** — the master key. Robot demos tank hits
  early and finish at 1 HP; training from those suffixes was near-impossible.
  Full HP at prefix handoff (training only): Stalactite 54→89%, Scaffold 27→93%
  in one run each.
- Per-level fresh campaigns ("champion factory") + 100-episode archive exams.

## Campaign laws (hard-won, all violated at least once before being named)

1. **Route-length law** — summit difficulty tracks demo route length, not
   crystal count. Conquered Switchback = shortest route (2,552 steps); walled
   Ore Shaft = longest in-cap route (2,998).
2. **Resume law** — never trust a resumed run; fresh campaign per target.
3. **Exam protocol / winner's curse** — promote by WIN metric from ≥100-episode
   exams only. The harness's 9-episode checkpoint evals cannot see a 10–40% win
   rate; collection stats anti-correlate with win capability late in runs.
4. **Exams need the full 16-level eval context** — state includes level index;
   pinning a single level silently breaks the policy.
5. **Opening starvation** (the unsolved one) — the final 3–8% of every ladder
   (the route's first steps from spawn) starves of win signal: rungs there need
   full-route wins to bank, and those are rare from deep starts. Ten campaigns
   post-championship produced zero new conquests on this blocker.

## Where the next gains are (future directions)

- **Opening-focused imitation**: behavior-clone only the first ~300 steps of
  demo routes (the segment the ladder can't reach), then hand off to the ladder.
- **Massively parallel workers**: Salimans & Chen used ~1,000 workers on
  Montezuma; our 8 vec-envs bank ~1 rung/2 wins. A PPO/IMPALA-style rewrite
  with 100+ workers is a new-project-sized lift but the literature-backed path.
- **Demo diversity**: alternate routes measurably widened rehearsal coverage;
  harvest more alts, especially short ones (route-length law).
- **Harvest the 5 undemoed levels** (5, 6, 8, 9, 10): without a demo there is
  no ladder; these need a stronger planner or human demos.
- **Ops**: `caffeinate` long runs — two machine sleeps and one battery death
  each cost hours (processes survive sleep but stall; scratch dirs are wiped).

## Reproduction & artifacts

- Runs: `runs/RUN-*/seed_*/` (live_metrics.json, checkpoints, `ladder_seed*.json`).
- Champion exams: 100-episode script pattern documented in PR #39 (2026-07-14
  comment); exam must load the checkpoint into the prepare_trainer agent and
  cycle `game._eval_cursor` through all 16 levels.
- Demos: `data/demos_goexplore/` (canonical), `data/demos_focus_*/` (per-level
  focus sets used by winning runs).
