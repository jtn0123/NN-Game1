# Crystal Caves generalization — RUN LOG

> ## ⚠️ METRIC CORRECTION (supersedes earlier "zero held-out transfer" reads)
> `crystal_frac` was aggregated with an **interquartile mean that floors any rate ≤25% to 0.000** (verified at N=12/20). On tutorial (1 crystal) the exit unlocks **iff** that crystal is collected, so the plain-mean `exit_unlocked_rate` ≈ the **true** held-out collect-rate. Re-reading that way, the agent has been **collecting crystals on ~0.15–0.33 of UNSEEN levels all along** — real generalization, NOT zero transfer. The wall is the **collect→exit conversion** (held-out win ≈ 0–0.08 ⇒ conversion ~0–25%): it solves leg 1 (find+collect) but fails leg 2 (route to the now-open exit). Fix shipped: `crystal_frac` now aggregated as a MEAN + a `collect->win conversion` line. Next lever = route-to-exit shaping (geodesic), NOT CNN/PPO. Earlier RUN-01..05 "held-out crystals ~0" verdicts are metric artifacts; trust `exit_unlocked_rate` / the new mean.


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

### RUN-06 — geodesic route-to-exit shaping A/B (M4) — DISCONFIRMED
`--regenerate-each-episode`, pool 24, 3 seeds, 3000 ep. Arm A control vs Arm B `--geodesic`.
- Arm A (control) best ep2500: held-out collect **0.300**, win **0.050**, conversion 0.17.
- Arm B (geodesic) best ep1000: held-out collect 0.150, win **0.000**, conversion 0.00; weaker train competence throughout, meanQ dipped negative.
- Verdict: dense geodesic potential **did not fix conversion and HURT learnability** (suppressed the base collect policy). Do not promote. The conversion wall (collect→exit ~0.17) stands; first targeted fix failed.

### RUN-07 — gentle geodesic (weight 0.1) (M4) — DISCONFIRMED
Same setup as RUN-06 Arm B but `--geodesic-weight 0.1`. Best ep1000: held-out collect 0.117, win **0.000**, conversion 0.00, meanQ −0.26; meanQ negative after ep500, train competence collapses after ep1000. Loses decisively to RUN-06 control (collect 0.30 / win 0.05 / conv 0.17 / meanQ +1.90). Verdict: geodesic shaping is the wrong tool for this learner at ANY weight — "too strong" was not the whole story. Fold condition met.

## Status: DECISION POINT (after RUN-07 — geodesic family disconfirmed)
Corrected, robust picture across RUN-04/05/06: the agent **generalizes collection** (~0.25–0.30 on unseen levels) but **fails collect→exit conversion** (held-out win stuck ~0.05, conversion ~0.17). Cheap data levers exhausted; first conversion fix (geodesic) failed/hurt. Open options (pending human direction):
- (a) lighter/after-unlock route-to-exit shaping (tune `CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT` down from 0.3, or shape only after exit unlock) — cheap, modest P.
- (b) CNN (`--cnn`, built) — lower priority; collection already generalizes so perception isn't the obvious conversion bottleneck.
- (c) accept near-ceiling for this DQN+task+budget; deliverable = rigorous investigation + metric fix + "collection generalizes, completion doesn't".

## (superseded) earlier next-plan
- **RUN-05 — KILL (salvage checkpoints).** Its decision metric was the broken IQM `crystal_frac`; finishing buys nothing. Re-grade its saved checkpoints with the fixed code (report held-out collect-rate via mean crystal_frac / `exit_unlocked_rate`, win, and conversion). Stop the training, keep artifacts.
- **RUN-06 (REVISED) — attack collect→exit conversion with geodesic route-to-exit shaping**, NOT the CNN. `CRYSTAL_CAVES_GEODESIC_POTENTIAL` (config.py) is already coded. A/B geodesic-on vs off, pool 24, tutorial, 3 seeds, ~1500–3000 ep. Success = held-out **win** rises (conversion up) while held-out collect-rate holds. Decision rule (from strategy panel): only if re-grade shows held-out collect genuinely ~0 do we instead go to the CNN (RUN-06-alt); if collect>0.15 & conversion<0.3 (current evidence), do geodesic.
- **CNN (`--cnn`) is built and parked** as the contingency if collect-rate turns out genuinely ~0.
- **PPO / bigger-net** deferred (wrong failure mode: this agent's signature is collect-generalizes / conversion-fails, not an overfit gap).
