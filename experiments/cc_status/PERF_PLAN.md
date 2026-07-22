# Training-Throughput Program — M4 MacBook vs Ryzen 9800X3D rig

Goal: maximize **campaign throughput** (env steps/sec per run × concurrent
runs) for the Crystal Caves factory, with zero change to learning semantics
unless explicitly flagged and re-validated.

## The standard benchmark (all numbers come from this, nothing else)

The first 6,000 episodes of the RUN-66 recipe (short-route Stalactite, combo
perception + backward ladder + opening-BC decay), seed 0, 8 vec-envs, CPU:

```
python experiments/cc_status/diagnose_gap.py --difficulty tutorial --imported \
  --episodes 6000 --seeds 0 --games 48 --vec-envs 8 --checkpoint-every 6000 --cpu \
  --geo-compass --geo-compass-hazard-aware --ngu-bonus --enemy-motion \
  --reward-clip 35 --stall-window 1440 --max-steps 4500 \
  --demo-dir experiments/cc_status/data/demos_focus_stalactite_short \
  --demo-reset-p 0.9 --demo-backward --demo-backward-retreat 60 \
  --demo-backward-wins 2 --demo-backward-window 240 --demo-backward-deep 600 \
  --demo-heal --demo-level-bias 0.7 --demo-td-weight 0 --demo-margin-weight 0.3 \
  --demo-opening-steps 300 --demo-margin-decay 18000 --out runs/bench6k
```

Report three numbers per row: **wall-clock to 6k episodes**, **mean steps/s**
(mean of the `⚡ N steps/s` samples across the whole log), and **eps/min**.
Windows launches use `python -u` and `PYTHONUTF8=1` (block buffering hides
progress; cp1252 crashes on the harness's emoji).

## Baselines (2026-07-21)

| Machine | Config | Wall-clock 6k | Mean steps/s | Notes |
|---|---|---|---|---|
| M4 MacBook (14C) | stock | canceled at 4,869 eps (81%) | 1,523 (partial avg) | reference only — user decision 2026-07-21: all testing moves to the rig; treat M4 as "slightly faster" and free for daily use |
| 9800X3D rig (8C/16T) | stock, native Windows | 92.9 min | 1,389 | torch 2.13 Win, defaults |
| **9800X3D rig, WSL2 Ubuntu** | stock, Linux runtime | **79.7 min** | **1,562** | **WINNER: +12.5% over native Windows, ahead of the M4 reference. Adopted as the rig's standard runtime (Python 3.12.13 via uv, repo in ext4 at ~/nn).** |

## Lever board (research-ranked 2026-07-21; one at a time, every row re-runs the standard bench)

| # | Lever | Machine | Expected | Effort | Status |
|---|---|---|---|---|---|
| P1 | Thread hygiene: OMP thread sweep + PASSIVE wait | both | — | trivial | **DEAD (2026-07-22): defaults 1,411 beat OMP=1 (1,276), OMP=2 (925), OMP=4 (766), OMP=1+default-wait (1,265) on matched 1,500-ep probes. Torch's heuristics win; do not override.** |
| P2 | Profile split (py-spy): env-step vs act() vs learn() vs PER — everything below re-ranks on this | both | information | 30 min | queued |
| P3 | Freebies: `torch.inference_mode()` in act, `set_flush_denormal(True)` | both | +0–10% | small | queued |
| P4 | **Learner in a background thread** (torch C++ ops release the GIL → learn() overlaps env stepping) | both | removes learn() from critical path | small-med | queued |
| P5 | Numba `@njit` game-step kernels (flatten physics state to numpy first) | both | 10–100× on kernels | 1–3 days | queued |
| P6 | **Parallel campaigns** (2–3 independent runs per machine) | rig esp. | ~linear ×N | none | ready now |
| P7 | Vectorize PER sampling (`np.searchsorted` stratified + batched priority updates) | both | kills Python-loop tail | half day | queued |
| P8 | PufferLib-style multiprocess envs (shared-mem obs, several envs/worker, batch-as-ready) — NOT naive AsyncVectorEnv (regresses for cheap envs per Gymnasium's own docs) | both | 3–6× env throughput | large | after P5 |
| P9 | Shared batched-inference across campaigns (SEED-RL pattern) | rig | amortizes per-forward overhead | large | with P6 at scale |
| — | Windows power plan / priority | rig | single digits | — | do once, don't bench |
| — | torch.compile, PyPy, int8 quant, Python 3.12/13 upgrade | — | measured/reported ≈ nil | — | dead levers |
| — | Python 3.14t free-threaded + torch 2.10 cp314t (no-IPC threaded envs) | both | future | — | watch list |

Key research facts pinned: Windows/Linux torch parity fixed since 2.4.1 (we
run 2.13); the M4's edge is Accelerate→AMX on small GEMMs; MKL-on-AMD penalty
is dead (oneDNN dispatches Zen properly); naive subprocess vec-envs LOSE for
cheap envs — shared-memory + multi-env-per-worker is the pattern that wins.
Full digest with sources: PR #43 comment (2026-07-21).

## Rig system checklist (round-2 research, 2026-07-21; sources in PR #43 comment)

1. **Adopt WSL2 Ubuntu as the training runtime** — likely-win: native Win11
   trails Linux ~10–15% on Zen 5 CPU work (Phoronix), WSL2 runs at ~94% of
   bare Linux → net ≈ +5–10% over native Windows, plus free `fork`, tmux +
   systemd ops, Defender-free ext4. Repo/venv/checkpoints in ext4, never
   /mnt/c. `.wslconfig`: memory=24GB, processors=16, swap=8GB,
   autoMemoryReclaim=gradual; systemd=true. Decided by the 3-way baseline.
2. **Windows Update reboot guard** (highest-stakes hygiene):
   NoAutoRebootWithLoggedOnUsers=1, notify-only updates, disconnect — never
   sign out — and pause updates before multi-day runs.
3. Crash resilience: supervising restart (systemd Restart=on-failure in WSL).
4. Defender exclusions for repo/venv/models/logs (moot inside ext4).
5. EXPO at DDR5-6000/6400 1:1 (small single digits, free).
6. GC tuning in-trainer: gc.freeze() after setup + gen0 threshold ~10–50k +
   periodic gc.collect(); NEVER blanket gc.disable() with tensors in play.
7. PBO + conservative Curve Optimizer (−15/−20) ~4–8% sustained — only with
   days-long stability validation; unattended-rig risk, defer.
8. Skip list: SMT off (need threads for parallel runs), telemetry/service
   culling (~0.3% measured), Game Mode (no-op), MKL numpy on AMD (keep
   OpenBLAS pip build), Memory Integrity off = optional 0–3%.

## Rules

1. Baseline before lever; lever lands only with a bench row proving it.
2. Throughput levers must not change learning semantics — same seeds must
   produce statistically indistinguishable ladder curves. Anything that alters
   step ordering or RNG consumption is flagged loudly in its row.
3. The rig's first production job stays the RUN-66 champion-trap replication —
   speed work must not delay banked-science work when both can run.
