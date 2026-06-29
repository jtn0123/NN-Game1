# Crystal Caves generalization — RUN LOG

Numbered, append-only log of every tracked experiment so we can refer to runs by ID.

**Convention going forward:** each new experiment gets the next `RUN-NN`. When the M4
agent reports, the result comment title is `## M4 RESULT RUN-NN <short-tag>`, and the
`--out` dir is `scratchpad/RUN-NN_<tag>/...`. Keep this file updated when a run lands.

All runs below: difficulty=tutorial (1 walk-reachable crystal + exit), `--cpu`,
`--truncation-bootstrap`, vec-envs 8, best-checkpoint + seed-averaged grading.
Baseline reference for "held-out crystals" memorization floor ≈ **0.033**.

| RUN | machine | config | held-out @ best ckpt (crystal / win) | verdict |
|-----|---------|--------|--------------------------------------|---------|
| RUN-01 | M4 | pool-size sweep: pool 24 & 256, 5 seeds, 1200 ep | p24: 0.000 / 0.03 · p256: 0.025 / 0.05 | pool size not the lever **at this budget**; 256 underfit (too few reps/level) |
| RUN-02 | M4 | learnceiling: pool 24, 3 seeds, 3000 ep | 0.00 / 0.00 (final 0.033 / 0.033); TRAIN cryst climbed 0.20→0.67 | learner IS capable (memorizes well); meanQ healthy +1.6→+4.6 (Q-divergence disproved); wall = **memorization** |
| RUN-03 | M4 | generalization-budget: pool 256, 3 seeds, 5000 ep | 0.033 / 0.00 (final TEST cryst 0.10, exit 0.27) | more diversity+budget → modest off-zero transfer, but wins ~0 and train weak; not enough |
| RUN-04 | M4 | infinite-deleak: pool 24, 3 seeds, 3000 ep, 2 arms | B (regenerate): 0.167 / 0.05 · C (regenerate+de-leak): 0.067 / 0.05 | fresh-level regeneration = **real but weak/noisy** lever (0.167 peak vs 0.033, spiky, regressed); de-leak **inconclusive**; agent UNDERfits in infinite regime → representation likely the ceiling |

## Detail

### RUN-01 — poolsize-sweep (M4)
`--episodes 1200 --seeds 0,1,2,3,4 --games 30 --checkpoint-every 300 --pool-size {24,256}`
- pool 24 best ep1200: TRAIN win 0.13 / cryst 0.287 · TEST win 0.03 / cryst 0.000 (huge gap = memorization).
- pool 256 best ep900: TRAIN win 0.05 / cryst 0.025 · TEST win 0.05 / cryst 0.025 (both ~0; budget-starved — only ~37 reps/level).
- Takeaway: at fixed short budget, more levels just dilutes memorization; not a clean win.

### RUN-02 — learnceiling (M4)
`--episodes 3000 --seeds 0,1,2 --games 20 --checkpoint-every 500 --pool-size 24`
- TRAIN curve: win 0.05→0.27, cryst 0.20→0.67 (still rising at ep3000); meanQ +1.6→+4.6.
- TEST stayed ~0.03; gap widened with training.
- Takeaway: capable learner (memorizes); not capability-limited at low diversity; Q-divergence/collapse hypothesis killed; the problem is memorization/transfer.

### RUN-03 — generalization-budget (M4)
`--episodes 5000 --seeds 0,1,2 --games 20 --checkpoint-every 500 --pool-size 256`
- Best ep2000: TRAIN win 0.12 / cryst 0.067 · TEST win 0.00 / cryst 0.033. Final ep5000: TEST cryst 0.10, exit-unlock 0.27, win 0.033.
- Takeaway: diverse pool + adequate budget lifts held-out crystals off zero (~0.10) and unlocks exits, but rarely converts to wins; train competence also weak.

### RUN-04 — infinite-deleak (M4)
`--episodes 3000 --seeds 0,1,2 --games 20 --checkpoint-every 500 --pool-size 24 --regenerate-each-episode [--drop-leak-features]`
- Arm B (regenerate only) best ep2500: TEST win 0.05 / cryst 0.167 — clears the ≥0.15 lift bar vs 0.033, but the curve is spiky and regressed to 0 by ep3000; TRAIN very weak (cryst ~0).
- Arm C (regenerate + de-leak) best ep1500: TEST win 0.05 / cryst 0.067 (below B at official best, though intermittent 0.167 spikes); better TRAIN depth 0.483.
- Note: under `--regenerate-each-episode` both diagnostic splits are holdouts, so the printed gap/verdict are ignored; judge absolute held-out.
- Takeaway: regeneration is a confirmed (if fragile) lever; de-leak unproven; the agent UNDERfits with unlimited levels → points at the observation/representation (flat MLP) as the ceiling, not the data pipeline.

## Pre-numbering (exploratory, cloud box — kept for reference)
- Long baseline vs `truncation_fix` (1200 ep, 3 seeds, easy): truncation looked slightly worse — later found confounded (graded at final/collapsed episode). Led to best-checkpoint grading.
- Phase-0 train-vs-test diagnostic (600 ep, 2 seeds) + learning-curve diagnostic (1500/3000 ep): established memorization diagnosis + that "collapse" was largely noise.
- Disconfirmed levers (earlier A/B): reward shaping, start-state reverse curriculum, CNN+global-average-pool, truncation-as-terminal-vs-bootstrap.

## Next (proposed) — pending direction
- RUN-05 candidates: (a) position-preserving CNN (flatten, multi-channel tile encoding) + regenerate — targets the suspected representation ceiling; (b) confirm RUN-04 Arm B with 5 seeds / longer budget before investing; (c) pause.
