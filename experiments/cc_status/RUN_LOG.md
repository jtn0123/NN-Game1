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
| RUN-12 | M4 | FAR reverse-exit curriculum p=0.5, 3 seeds, 4000 ep, 2 arms | ctrl: 0.242 / 0.033 (FAR probe 0.117) · curr: 0.192 / 0.033 (FAR probe 0.100) | **DISCONFIRMED** — drilling far-starts lowered train competence (win 0.375→0.20) AND the FAR probe; control wins. Confirms leg-2 is a **representation ceiling**, not a data/start-distribution problem → reward/curriculum family exhausted, observation change (RUN-13) is the live lever |
| RUN-13 | M4 | **geodesic corridor compass** (`--geo-compass`), 3 seeds, 4000 ep, 2 arms | ctrl: 0.242 / 0.033 (FAR 0.117) · compass: **0.817 / 0.483** (FAR **0.575**) | ✅ **BREAKTHROUGH** — held-out win 0.033→0.483 (14×), FAR probe 0.117→0.575, every seed, from ep500 on. The flat MLP CAN use an explicit route signal; the wall WAS missing route info. Caveat: oracle-fed → RUN-14 makes it learnable |
| RUN-14 | M4 | compass A/B **beyond tutorial**: easy + normal, 3 seeds, 4000 ep | easy compass: 0.658 cryst / **0.042** win (FAR 0.308) · normal compass: 0.833 cryst / **0.000** win (FAR 0.125, **exit_unlocked 0.000**) | ✅ compass lift **survives** (collection/routing big on both); ❌ **not sufficient for normal wins**. Normal wall MOVED downstream: exit never unlocks (never collects the LAST crystals) + train wins also 0 ⇒ **full-chain completion / capability**, not routing or memorization |
| RUN-15a | local | completion diagnostic (agent-free, structural), 40 held-out lvls/difficulty | normal: 100% winnable, perfect tour ~1103 steps (37% of budget), 0 no-progress traps | normal's 0 wins are NOT structural (budget/reachability fine) ⇒ **behavioral** completion wall. Rules out "raise MAX_STEPS"; validates curriculum/capability family |
| RUN-15 | M4 | difficulty curriculum: 2000 easy→2000 normal warm-start vs 4000 normal, both compass, 3 seeds | curr: 0.725 cryst / 0.000 win (FAR 0.058) · ctrl: 0.833 / 0.000 (FAR 0.125) | ❌ **DISCONFIRMED** — easy pretrain washes out on normal hazards; regressed the compass baseline (crystals, NEAR, FAR all down), exit_unlocked still 0.000. Simple warm-start dead. Wall = normal-specific hazards/enemies (suspect) → diagnose failure mode next |

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

### RUN-08 — after-unlock-only geodesic (M4) — DISCONFIRMED (terminal shot)
`--geodesic-after-unlock` (geodesic shapes leg-2 only; leg-1 keeps its normal approach reward), pool 24, 3 seeds, 3000 ep. Best ep2000: held-out collect 0.200, win **0.017**, conversion 0.08, meanQ +1.58 (positive throughout — learnability NOT damaged, unlike RUN-06/07). Still loses to RUN-06 control (collect 0.30 / win 0.05 / conv 0.17). Verdict: the gating fixed the learnability harm (confirming prior geodesic broke leg-1), but produced **no reliable held-out completion lift**. Shaping family fully exhausted. Per the pre-agreed terminal rule: STOP, consolidate.

---

## FINAL SUMMARY / CONCLUSION
**Original goal — raise held-out (unseen-level) WIN rate — NOT achieved.** Held-out wins stayed ~0.05 across all 8 runs. But the investigation produced real, durable results:

**What we established (high confidence):**
- The agent **generalizes leg 1**: on *unseen* levels it finds + collects the crystal **~0.25–0.30** of the time (this unlocks the exit). It is NOT a zero-transfer memorizer.
- The wall is **leg 2 — collect→exit conversion ~0.17**: after unlocking the exit it usually fails to route to it on unseen layouts.
- Value learning is healthy (meanQ positive); no Q-divergence/collapse (the earlier "collapse" was small-sample noise).

**The big bug we fixed:** `crystal_frac` was aggregated with an interquartile mean that floors any rate ≤25% to 0.000, which masked the real ~0.25–0.30 held-out collection and produced a false "zero transfer / pure memorizer" story for several runs. Fixed (mean aggregation + a `collect→win conversion` line). `exit_unlocked_rate` ≈ true tutorial collect-rate.

**Levers tried and DISCONFIRMED** (all measured with the corrected metric, best-checkpoint, seed-averaged): pool size / diversity (RUN-01,03), more budget (RUN-02,03), fresh-level regeneration (RUN-04,05), de-leak features (RUN-04C), reward shaping & start-state curriculum (pre-numbered), truncation-as-terminal vs bootstrap (bootstrap helped value stability only), CNN+global-avg-pool (pre-numbered), geodesic shaping at full / gentle / after-unlock strengths (RUN-06,07,08). Reward shaping consistently failed or harmed this fragile learner.

**Honest conclusion:** for this ~50K-param DQN on this hard multi-step procedural task, leg 1 generalizes and leg 2 does not, and cheap shaping/data levers do not move it. Real further gains would need a bigger swing — e.g. **PPO/on-policy** (the ProcGen-generalization standard), a **leg-2-specific curriculum** (train route-to-exit from post-collection starts in isolation), or **representation work** — all higher-effort with uncertain payoff.

**Durable deliverables kept on this branch:** the metric fix + conversion metric; the train-vs-held-out diagnostic with best-checkpoint + seed-averaging (`diagnose_gap.py`); per-seed aggregator (`aggregate_diag.py`); levers behind flags (truncation-bootstrap, regenerate-each-episode/infinite levels, drop-leak-features, weight-decay, CNN, geodesic variants); this RUN_LOG; and the distributed M4 experiment workflow.

## (historical) decision points
Corrected, robust picture across RUN-04/05/06: the agent **generalizes collection** (~0.25–0.30 on unseen levels) but **fails collect→exit conversion** (held-out win stuck ~0.05, conversion ~0.17). Cheap data levers exhausted; first conversion fix (geodesic) failed/hurt. Open options (pending human direction):
- (a) lighter/after-unlock route-to-exit shaping (tune `CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT` down from 0.3, or shape only after exit unlock) — cheap, modest P.
- (b) CNN (`--cnn`, built) — lower priority; collection already generalizes so perception isn't the obvious conversion bottleneck.
- (c) accept near-ceiling for this DQN+task+budget; deliverable = rigorous investigation + metric fix + "collection generalizes, completion doesn't".

## (superseded) earlier next-plan
- **RUN-05 — KILL (salvage checkpoints).** Its decision metric was the broken IQM `crystal_frac`; finishing buys nothing. Re-grade its saved checkpoints with the fixed code (report held-out collect-rate via mean crystal_frac / `exit_unlocked_rate`, win, and conversion). Stop the training, keep artifacts.
- **RUN-06 (REVISED) — attack collect→exit conversion with geodesic route-to-exit shaping**, NOT the CNN. `CRYSTAL_CAVES_GEODESIC_POTENTIAL` (config.py) is already coded. A/B geodesic-on vs off, pool 24, tutorial, 3 seeds, ~1500–3000 ep. Success = held-out **win** rises (conversion up) while held-out collect-rate holds. Decision rule (from strategy panel): only if re-grade shows held-out collect genuinely ~0 do we instead go to the CNN (RUN-06-alt); if collect>0.15 & conversion<0.3 (current evidence), do geodesic.
- **CNN (`--cnn`) is built and parked** as the contingency if collect-rate turns out genuinely ~0.
- **PPO / bigger-net** deferred (wrong failure mode: this agent's signature is collect-generalizes / conversion-fails, not an overfit gap).

---

## REOPEN — leg-2 probe → leg-2 curriculum

### RUN-09 — leg-2 probe (route-to-exit isolation) — BORDERLINE PASS
Added a held-out **leg-2 probe** (`--leg2-probe`): after training, drop the trained agent in the post-collection state right next to the now-open exit (reverse_exit start, oracle-verified reachable) on held-out levels and measure greedy reach-exit rate. This isolates leg-2 (route-to-exit) from leg-1 (find+collect). Also oracle-hardened the reverse-start placement so the agent is never dropped in an un-jumpable pocket (which would read as a false ceiling).
- Result: seed-avg reach-exit **0.50**, narrowly over the ≥0.5 bar.
- Read: the route-to-exit skill **exists** but is under-practiced in normal play → a leg-2 curriculum has *some* upside. Signal is borderline; not to be over-claimed.
- Decision rule (pre-agreed): reach ≥0.5 → earns **one** RUN-10 leg-2 curriculum attempt.

### RUN-10 — in-env reverse-exit curriculum (M4) — DISCONFIRMED
Built `CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM(+_P)` + `--reverse-exit-curriculum-p`. A/B: Arm A control (RUN-06 baseline) vs Arm B `--reverse-exit-curriculum-p 0.5`, tutorial, 3 seeds, 4000 ep, `--leg2-probe`, process-per-seed, CPU.
- **Arm A control:** held-out best-by-win ep2500 win **0.058** / collect 0.192 / conversion **0.30** / meanQ +4.72. Leg-2 probe **0.733** (per-seed [0.80, 0.675, 0.725]).
- **Arm B curriculum:** held-out best ep2000 win **0.050** / collect 0.183 / conversion **0.27** / meanQ +3.17. Leg-2 probe **0.767** (per-seed [0.775, 0.875, 0.65]).
- Curriculum also LOWERED train competence (train win 0.375→0.25) and final held-out win (0.033→0.025) — same fixed-p tax we saw on the base reverse curriculum (half the resets skip the full-from-spawn task).
- **Verdict: disconfirmed.** Drilling raised the isolated probe a hair (0.73→0.77) but did NOT lift held-out full-chain win or conversion; control is actually ahead on the best checkpoint. Skill-in-the-drill did not transfer.

### ⚠️ KEY INSIGHT — the probe & curriculum measured/drilled the WRONG thing
The leg-2 probe and the reverse-exit curriculum BOTH drop the agent **within ~5 tiles of the open exit** (`place_player_near_tile` / near-exit relocation, closest-first). That isolates only the **trivial final hop**, not the real leg-2 wall: **long-range navigation from the last-collected-crystal across the whole level to the exit.** This explains everything: control already scored 0.73 on the probe (the final hop IS easy), so RUN-09's 0.50 was a small-sample underestimate, and drilling the easy hop changed nothing because the agent was never bad at it. The genuine post-collection navigation skill was never isolated or drilled. So RUN-10 disconfirms *this* curriculum, but does NOT cleanly disconfirm "leg-2 navigation help" in general — that lever was mis-specified. A corrected diagnostic = drop at a RANDOM reachable standing tile (full-level distance) with the exit open; a corrected curriculum would drill from there, not adjacent to the exit.

### RUN-11 — corrected leg-2 diagnostic (NEAR vs FAR probe) (M4) — DECISIVE
Built a FAR probe to fix RUN-10's mismatch: `--leg2-probe` now runs NEAR (next to exit, ~1 tile, the old probe) AND FAR (random reachable standing tile, real navigation; oracle-verified the exit is reachable from it). Pushed `e04e872`. Control-only, 3 seeds, 4000 ep.
- **Result: NEAR = 0.733** (per-seed [0.80, 0.675, 0.725]) vs **FAR = 0.117** (per-seed [0.10, 0.075, 0.175]), mean FAR start distance **17.3 tiles**. Held-out win still 0.033, collect 0.242, conversion 0.14.
- **Read: NEAR high & FAR low → the wall IS long-range route-to-exit navigation.** The agent aces the final hop (0.73) but almost never navigates the ~17-tile route to the exit after collection (0.12). The old near-exit probe (and RUN-10's curriculum) measured/drilled a non-missing skill — fully explains RUN-10's null. The genuinely-missing skill is now isolated.
- Pointer: a CORRECTED far-start curriculum (drill from far reachable tiles, not adjacent) = RUN-12. Caveat: FAR=0.12 could also mean the flat-MLP representation can't do unseen-level navigation, in which case a curriculum hits the same ceiling — RUN-12 tests which.

### RUN-12 — FAR reverse-exit curriculum (M4) — DISCONFIRMED
Built `CRYSTAL_CAVES_REVERSE_EXIT_CURRICULUM_FAR` + `--reverse-exit-curriculum-far`: the reverse-exit curriculum now drops the player on a random reachable tile ≥6 tiles from the open exit (mean ~22), drilling the RUN-11-confirmed missing skill (long-range route-to-exit) instead of RUN-10's trivial hop. Pushed `aff36ea`. A/B: Arm A control vs Arm B `--reverse-exit-curriculum-p 0.5 --reverse-exit-curriculum-far`, 3 seeds, 4000 ep, `--leg2-probe`. (~2h36m on M4, six parallel seed workers, all exit 0.)
- **Official best ckpt (ep4000):** control held-out win **0.033** / collect **0.242** / conv 0.14 / FAR probe **0.117** · curriculum win **0.033** / collect **0.192** / conv 0.17 / FAR probe **0.100**.
- **Held-out win peak over ckpts:** control **0.058** @ep2500 vs curriculum **0.033** @ep4000.
- **Train competence dropped:** train win 0.375→**0.200** (the fixed-p=0.5 drill makes half the resets skip the full-from-spawn task → underfits). NEAR probe also dipped (0.733→0.625).
- **Verdict: disconfirmed — control wins.** Not even "FAR probe up but full-play flat": the FAR probe itself is *lower* with the curriculum (0.100 vs 0.117). Drilling far-starts harder did not lift navigation and reduced overall competence. This is the **representation-ceiling** branch RUN-11 flagged: the flat-MLP can't acquire unseen-level route-planning from more reps alone — so the fix must change what the agent *sees/computes*, not just its start distribution. → the reward/curriculum family for leg-2 is exhausted; the live lever is RUN-13 (give it the route as an observation).

### RUN-13 — geodesic corridor compass (M4) — ✅ BREAKTHROUGH (first real lift)
Brainstorm #1 (see NAV_EXPANSION_BRIEF.md). RUN-11 root cause: the agent's only directional signal is a EUCLIDEAN compass that points at the objective THROUGH walls, so it can't route around them (FAR probe 0.12). Fix: append 4 metadata scalars `[step_dx, step_dy, reachable, geo_dist_norm]` pointing down the real traversable route, read from the already-cached `_geodesic_distance_field()` BFS. Observation (computed identically at eval), NOT reward → does not re-trigger the disconfirmed geodesic-PBRS lever. Pushed `6f1d910`. Cheap (~1.3K params, +4 state dims), works on MLP + SpatialDQN. **Directly attacks RUN-12's representation-ceiling finding:** RUN-12 proved more reps won't teach route-planning, so this hands the agent the route signal directly instead.
- A/B: Arm A control vs Arm B `--geo-compass`, tutorial, 3 seeds, 4000 ep, `--leg2-probe`. (~1h51m on M4, all six workers exit 0.)
- **RESULT — every grade metric moved the right way, hugely, on every seed:**

| metric | control | geo-compass | Δ |
|---|---:|---:|---:|
| held-out win (best ckpt) | 0.033 | **0.483** | +0.450 |
| held-out win (final ep4000) | 0.033 | 0.475 | +0.442 |
| held-out crystals (best) | 0.242 | 0.817 | +0.575 |
| collect→win conversion (best) | 0.138 | 0.592 | +0.454 |
| FAR leg-2 probe | 0.117 | **0.575** | +0.458 |
| NEAR leg-2 probe | 0.733 | 0.850 | +0.117 |
| train win | 0.375 | 0.842 | +0.467 |

- **Read: the flat MLP CAN exploit an explicit corridor signal — the bottleneck was missing route information, exactly as RUN-11 diagnosed and RUN-12 (curriculum) failed to supply.** Not a one-checkpoint spike: compass beats control from ep500 onward, every seed, and it lifted leg-1 (collect, train win) too — knowing the route to objectives helps the whole chain, not just leg-2. Gap narrowed (win gap 0.342→0.300) AND absolute held-out rose. This clears the success branch.
- **Honest caveat (drives RUN-14):** the compass is *oracle-fed* — a BFS planner runs alongside the net and hands it the descending direction each step; the net learns to follow it (and to platform/execute, which is why it's 0.48 not ~1.0). It's eval-fair (computed identically per-level at eval, on unseen layouts, so it's real generalization not memorization) and cheap, but it leans on a hand-coded route oracle. RUN-14 should make the route signal *learnable* (aux route-distance head — note `CRYSTAL_CAVES_ROUTE_AUX_LOSS`/`route_aux_head` infra already exists — and/or a spatial trail channel / CNN that preserves structure), so the net builds a route-aware representation from raw observation rather than being spoon-fed. Also untested: whether the lift holds at `easy`/`normal` difficulty (multi-crystal + hazards) — RUN-13 is tutorial (1 crystal).
- Verified pre-run: compass step strictly reduces route-distance to exit in 30/30 far states.

### bench-net — CPU vs MPS device decision (M4) — RESOLVED: stay on CPU+vec-envs
Micro-benchmark (`bench_net.py`, commit `f1d114a`) timing a real train step for the 614K MLP / 798K flatten-CNN / 77K global-pool CNN. Both legs run on M4 (GPU leg alongside RUN-13 on the idle GPU; CPU leg after, on the freed box). Samples/s (steps/s×batch):
- **MLP: CPU wins 5–10×** at every batch (overhead-bound on MPS). No contest.
- **CNN: CPU wins at B128** (the recipe regime); MPS only edges ahead at B512+ (where conv compute finally amortizes GPU dispatch). Throttle detector flagged 2 MPS cells (early-vs-late drift) — methodology worked.
- **Decision: keep RUN-14 on CPU + `--vec-envs`** regardless of architecture. The real DQN loop (CPU env-step, replay sampling, per-step host sync) is even less MPS-favorable than this isolated bench. Neural Engine is unreachable from PyTorch training. MPS would only pay off for a multi-M-param encoder at batch ≥512 with a sync-free loop — not justified by any evidence.

### RUN-14 — compass beyond tutorial (easy + normal) (M4) — lift survives, but the wall MOVED
Repeat the RUN-13 compass A/B at `easy` (2–3 crystals, no hazards) then `normal` (10–14 + hazards + enemies), 3 seeds, 4000 ep, `--leg2-probe`. Commit `07f858c`.
- **Easy:** compass held-out crystals **0.200→0.658**, progress 0.56→0.91, NEAR 0.38→0.83, FAR 0.075→0.308, exit-unlock 0.00→0.158, **win 0.000→0.042**. Routing/collection lift clearly survives; wins modest.
- **Normal:** compass held-out crystals **0.542→0.833**, progress 0.48→0.90, depth 0.28→0.82, FAR 0.025→0.125 — big navigation/collection lift, **train/test gap shrinks** (cryst gap 0.083 vs 0.308) so not memorization. BUT **held-out win = 0.000 for BOTH arms**, and **exit_unlocked = 0.000 for both**.
- **Key read — the wall moved downstream of routing.** On normal the compass nails collection (0.83) and routing, but the exit *never unlocks* (it never collects the LAST crystals) and **train wins are also 0** → this is a **full-chain completion / capability wall**, not routing, not generalization. Two stacked obstacles now: (1) finish collecting all 10–14 crystals; (2) the post-collection FAR route-to-exit is still weak at normal (0.125). The compass solved the *navigation* sub-problem; the binding constraint is now *completing the long multi-objective chain* (and execution: hazards/enemies/timing).
- **Implication for RUN-15:** Op 2 (learnable route, built `26e33c3`) attacks routing — already solved by the compass — so it is **not** the normal unlock; best used as an easy/oracle-removal A/B. The normal-difficulty lever is **completion/capability**: a difficulty curriculum (warm-start normal from an easy/compass-trained agent — easy *does* produce wins; normal-from-scratch does not) and/or last-crystal completion + post-collection execution. NOTE the M4's suggested far-start exit curriculum ≈ RUN-12, which was DISCONFIRMED — re-tread only with the compass present and clear justification.

### RUN-15 — (pending user direction) candidates
1. **Difficulty curriculum (warm-start normal ⟵ easy)** — top pick; targets the confirmed "can't complete the long chain from scratch" wall (train wins 0 at normal, but easy produces wins).
2. **Op 2 learnable route** (`--route-aux`, built) — elegance/oracle-removal; A/B on tutorial/easy where routing→wins. Won't crack normal alone.
3. **Completion diagnostic** — why does collection stall at ~0.83 / exit never unlocks? (step-limit? last-crystal accessibility? hazards?) Cheap; de-risks before committing a build.

### RUN-15a — completion diagnostic (agent-free, structural) — DONE: the wall is BEHAVIORAL
`diagnose_completion.py` (commit `58d6dec`), 40 held-out levels/difficulty, perfect-agent upper bound (greedy geodesic collect-all-then-exit tour vs MAX_STEPS/no-progress window). No training/GPU.

| difficulty | crystals | reachable | tour steps (mean / max) | fits 3000 | budget used | longest leg < 720 |
|---|---:|---:|---:|---:|---:|---:|
| tutorial | 1.0 | 100% | 305 / 442 | 100% | 10% | 100% |
| easy | 2.4 | 100% | 531 / 907 | 100% | 18% | 100% |
| normal | 12.4 | 100% | 1103 / 1585 | 100% | 37% | 100% |

- **Result: the cheap structural causes are RULED OUT.** Every normal level is winnable; a perfect agent's tour uses only ~37% of the 3000-step budget (max 1585); no inter-crystal leg exceeds the 720 no-progress window. So normal's 0 wins are NOT a step-limit / unreachable-crystal / no-progress-trap problem.
- **⇒ The normal wall is purely BEHAVIORAL/capability:** the policy can't *execute* the full 12-crystal chain + hazard/enemy survival from scratch, even though it's comfortably doable within limits. (The tour also ignores hazards/enemies, so real play only adds DEATH risk + dodging overhead — a survival dimension, not a budget one.) This validates the **curriculum/capability** family for RUN-15 and rules out "just raise MAX_STEPS."
- Caveat: still behavioral-blind (agent-free). A loadable normal checkpoint (needs `--save-checkpoints`) would let a follow-up probe split the actual failure into timeout-wander vs death vs stuck — but structural feasibility being clean already justifies the curriculum build.

### RUN-15 — difficulty curriculum (warm-start normal ⟵ easy) (M4) — ❌ DISCONFIRMED
`--curriculum-easy-episodes` (commit `80cac5e`): 2000 easy pretrain → 2000 normal warm-start vs control 4000 normal from scratch, both `--geo-compass`, 3 seeds, held-out normal. (~1h38m, all exit 0.)
- **Result: did NOT crack completion AND regressed the compass baseline.** Both arms held-out win **0.000**, exit_unlocked **0.000**, conversion 0.00. Curriculum WORSE on the useful signals: crystals **0.725 vs 0.833**, NEAR **0.433 vs 0.633**, FAR **0.058 vs 0.125**, meanQ 4.4 vs 6.1.
- **Read:** easy pretrain is a weak/partial policy (held-out easy ~30-44% crystals) that **washes out / misaligns** on normal (which adds hazards/enemies the easy agent never saw). At equal total budget the curriculum starts behind on normal and never catches up — trading 2000 normal episodes for easy pretrain net-hurt. Simple difficulty warm-start, this recipe, is dead. (Lower-priority tweaks — epsilon-carry, lighter split — unlikely to overcome a baseline regression.)
- **Persistent wall stands:** exit_unlocked = 0.000 on held-out normal for BOTH arms → still **never collects all ~12 crystals**. RUN-15a proved it's structurally feasible ⇒ behavioral. Prime suspect: the **normal-specific hazards/enemies** (the one thing easy lacks, and easy *did* produce wins). → diagnose the failure mode (hazard exposure / death-vs-timeout) before the next lever; don't guess between survival/execution/shaping.

### RUN-15b — hazard-exposure probe (agent-free) — DONE: wall = hazard/enemy survival
Extended `diagnose_completion.py` (commit `5616202`) with hazard/enemy density + crystal-guarding, 40 held-out lvls/difficulty.
- **easy:** 0 hazards, 0 enemies, 0% crystals guarded. **normal:** 7.5 hazards + 3.6 enemies/level, **35% of crystals adjacent (≤1 tile) to a threat** (43% ≤2). To unlock the exit the agent must safely collect ALL ~12 crystals incl. ~4-5 threat-guarded, on 3 health.
- Confirmed the perception window already injects hazards+enemies (`_fill_window`) → the agent **SEES** threats but hasn't **learned** to survive them: a **capability gap, not an observation gap** ⇒ "add hazards to state" is NOT the fix. Explains RUN-15 (easy-pretrained policy never saw a hazard → washed out).

### RUN-16 — behavioral death-trace (BUILT, dispatched to M4)
`--death-trace` (commit `3e52d08`): after training, greedy-play held-out normal and report end-reason distribution (won/killed/timeout/stalled), the hazard-vs-enemy-vs-air split of deaths, and crystals-at-end. Added `last_damage_source` tracking to the game. Single arm (diagnostic), normal+compass, 3 seeds, 4000 ep.
- Decides RUN-17's lever: killed≫timeout & enemy-dominated → enemy-velocity obs / combat; killed & hazard-dominated → static-hazard pathing; timeout-dominated → exploration/credit (not death).
- Smoke-tested end-to-end; 111 crystal_caves tests pass.

### RUN-16-fixed — death-trace rerun on the audit-fixed harness (M4)
Reran the RUN-16 death-trace on `9adee82` (post audit-round-1). The CORRECTED metrics give an honest, different read.
- **Held-out (true means):** win 0.000, exit_unlocked 0.000, crystal_frac **0.186** (NOT the old "0.83" collected-≥1 rate), target_progress **0.487** (NOT the old saturated "0.90"). Train crystal_frac 0.293 — both low ⇒ a competence/early wall, not a clean memorisation gap.
- **Death-trace (fixed attribution):** **stalled 0.592**, killed 0.400 (hazard **0.250** > enemy **0.150**, both **0.000**), timeout 0.008, crystals-at-end 0.186, steps 999.
- **Key shift from the buggy run:** the B8 stall fix moved ~0.36 of episodes from "timeout" (0.367→0.008) to **"stalled" (0.225→0.592)** and dropped mean steps 1737→999 — confirming the agent OSCILLATES WITHOUT NET PROGRESS (gets stuck) in most episodes rather than running out of time while advancing. B6's `both=0.000` confirms the hazard>enemy split is real, not an overlap-attribution artifact.
- **Reframed wall:** collect ~19% of crystals, then ~59% **get STUCK** (stall: no net progress to the next objective for 720 steps) and ~40% **die** (hazard-dominated). The compass already gives the route, so this is an EXECUTION/getting-unstuck problem (jumps/hazard-dodging/maneuvering through complex terrain), plus hazard survival — NOT a routing-info gap. Leg-2 FAR still 0.125 but rarely reached in full play (exit never unlocks).
- RUN-17 lever (pending round-2 audit clearance): target getting-unstuck + hazard-execution, not more routing signal.
- M4 perf note: per-seed speed now bound by PER replay sampling (`np.random.choice` over priorities), not device — a code-level optimization target if needed.

### RUN-17 — R2 trusted baseline (post two audits, R2-A applied) (M4)
Reran normal+compass death-trace on `2a6d3fe` (22 audit bugs fixed; R2-A removed the farmable already-thrown-switch reward). This is the trusted pre-lever baseline. 3 seeds, 4000 ep, `--geo-compass --death-trace --leg2-probe`.
- **Held-out:** win **0.000**, exit_unlocked **0.000**, crystal_frac **0.199**, target_progress 0.514. Train crystal_frac 0.358. meanQ +6.38 (bounded — not Q-divergence).
- **Death-trace:** killed **0.483** / stalled **0.492** / timeout 0.025. Kills: hazard **0.283** > enemy **0.200**, both 0.000. crystals-at-end 0.199, steps 986.
- **R2-A effect:** stall dropped 0.592 → 0.492 (−0.10) vs RUN-16-fixed → ~a tenth of prior stalling WAS lever-camping. Baseline now trustworthy. Leg-2 NEAR 0.767 / FAR 0.242 (downstream — exit rarely unlocks).
- **Read:** failure ~50/50 killed/stalled; hazard is the largest single death source → motivated RUN-18 (hazard-aware compass).

### RUN-18 — hazard-aware corridor compass (A/B) (M4) — ❌ DISCONFIRMED
`--geo-compass-hazard-aware` (commit `431c694`): route the compass AROUND static hazards (separate hazard-weighted Dijkstra field; observation-only, same 4 dims). Control = plain compass. 3 seeds, 4000 ep, normal.
- **Reduced deaths but bought nothing.** killed 0.483 → 0.417, hazard 0.283 → 0.258, enemy 0.200 → 0.158; guardrails held (train flat 0.358→0.356, NEAR 0.767→0.792, FAR 0.242→0.258). BUT stalled **0.492 → 0.558** (+0.067 — deaths converted to stalls), mean steps 986→1090, wins still **0.000**, held-out crystals flat-to-worse 0.199→0.195.
- **Read:** the compass *was* steering into spikes (deaths drop when routed around), but suppressing deaths just converts them to stalls — no net objective gain. Don't stack. The binding constraint is the STALL, not survival per se.

### RUN-19 — stall diagnostic (single arm) (M4) — decisive: far+oscillating
`--stall-trace` (commit `fc0202e`): for each held-out STALLED episode, classify trapped (objective physics-unreachable from the stuck tile) / near-objective (≤3 tiles) / far / oscillating (≤3 tiles in trail). Normal+compass, 3 seeds, 4000 ep. Reproduces the RUN-17 baseline shape exactly.
- **Stall-trace (seed-avg over stalled eps):** trapped **0.018**, near-objective **0.137**, far-from-objective **0.863**, oscillating **0.645**, mean geodesic dist to objective **12.1 tiles**, crystals-at-stall 0.189.
- **Read:** NOT trapped (agent isn't wedged / compass isn't routing through impossible jumps) and NOT last-mile. It stalls **far from the objective, bouncing in a small local loop** — i.e. it has the route signal and still won't follow it. This is the **route-following / credit / exploration / representation** branch (a learning problem), not a control/jump-skill one. Disconfirms a jump-aware-routing or last-mile fix as the next lever; strengthens the case for a bigger swing (PPO / recurrence) or a route-following curriculum.
- Next: `--record-play` (commit `c852b96`) to WATCH the far+oscillating behavior on a deterministic replay before committing to the bigger swing.

### RUN-23 — reward calibration + NGU movement telemetry (M4) — CLIP-MASKED / C VALID
Protocol-v2 canonical `diagnose_gap --imported`, normal, geo-compass + hazard-aware, enemy motion off, 3 seeds, 4000 ep, checkpoints every 500. Training launched at `6332f9b`; aggregation/reporting pulled through `c894700`.
- **Critical finding:** default `REWARD_CLIP=5.0` clips sampled replay rewards at `-5`, so death (`-12`), timeout (`-8`), and stall (`-6`) have all trained as the same negative magnitude historically. The requested death-scale arm (`--death-penalty=-30`) was therefore masked: default `-12` and requested `-30` both train as `-5`. The D/combo arm matched C/NGU bit-identically for same seeds through observed checkpoints, confirming the clip masked death-scale behavior. D was stopped once this was confirmed.
- **B/death arm:** completed 3 seeds, but treat as clip-masked protocol-v2 baseline-ish, **not** a valid death-scale test. Best ep3500: win **0.021** (1/48, CI noise), crystals **0.520**, damage 2.58, tiles 105.3, idle 0.769. Final: win 0.000, crystals **0.533**, killed **0.500** (enemy 0.438, hazard 0.063), stalled **0.438**, timeout 0.063.
- **C/NGU arm:** valid movement/novelty test. Best ep3500: win **0.021** (1/48, CI noise), crystals **0.575**, damage 2.52, tiles **122.2**, idle 0.764. Final: win 0.000, crystals **0.562**, killed **0.729** (enemy 0.688), stalled **0.208**, timeout 0.063.
- **Read:** NGU raises movement/coverage and crystals modestly and cuts stalls, but converts much of the failure mass into enemy deaths; it does not create robust wins. Death-scale conclusions are blocked until rerun with a higher/disabled reward clip. Follow-up RUN-23b uses `--reward-clip 35` to control the clip and retest death scale.

### RUN-23b — reward-clip controlled death scale (M4) — ❌ DISCONFIRMED
Corrected RUN-23 after the clip finding. Protocol-v2 canonical `diagnose_gap --imported`, normal, geo-compass + hazard-aware, enemy motion off, 3 seeds, 4000 ep, checkpoints every 500. Arms: A′ `--reward-clip 35`, B′ `--reward-clip 35 --death-penalty=-30`, D′ `--reward-clip 35 --death-penalty=-30 --ngu-bonus`. Current branch head `37464d0`.
- **A′ clip-control:** best/final ep4000 win **0.000**, crystals **0.525**, damage 2.00, tiles 108.8, idle 0.740. Death trace: killed **0.458** (enemy 0.417, hazard 0.042), stalled **0.500**, timeout 0.042.
- **B′ death-scale:** best/final ep4000 win **0.000**, crystals **0.551**, damage 2.23, tiles 111.6, idle 0.773. Death trace: killed **0.500** (enemy 0.438, hazard 0.063), stalled **0.479**, timeout 0.021.
- **D′ combo:** best/final ep4000 win **0.000**, crystals **0.530**, damage 2.25, tiles 115.7, idle 0.728. Death trace: killed **0.563** (enemy 0.500, hazard 0.063), stalled **0.396**, timeout 0.042.
- **Read:** raising the clip alone does not unlock completion. Stronger death penalty gives a small crystal lift (+0.026 vs A′) but fails the decision rule because killed does not drop (0.458 → 0.500) and no wins appear. Combo lowers stalls (0.500 → 0.396) and visits more tiles, but converts failures into enemy deaths (0.458 → 0.563 killed) and still has 0 wins. Reward-side fixes are marginal-or-worse; move to the demonstration path / Tier 2 rather than more scalar reward tuning.

### RUN-24 — long-horizon D′ best-so-far config (M4) — LONGER HELPS CRYSTALS, NOT COMPLETION
Tested whether RUN-23b's 4000-episode horizon was simply too short. Protocol-v2 canonical `diagnose_gap --imported`, normal, geo-compass + hazard-aware, enemy motion off, 3 seeds, **16000 ep**, checkpoints every 1000, `--record-play 2`. Config: `--reward-clip 35 --death-penalty=-30 --ngu-bonus` on training/eval HEAD `980e40e`; PR head advanced to `34c6e18` while the run was active, but workers stayed on the launch head. Output: `scratchpad/RUN-24_long_horizon_Dprime/`.
- **Runtime:** 3 parallel seed workers completed cleanly; slowest seed set wall time at ~203m. Preflight passed (`14` eval/paired tests; `16/16` imported levels winnable).
- **Curve:** old 4k point win **0.000** / crystal **0.539**. Longer training lifted crystal collection into the ~0.60-0.66 band: ep8000 **0.000 / 0.611**, ep9000 **0.042 / 0.629**, ep11000 **0.062 / 0.651**, ep12000 **0.021 / 0.594**, ep15000 **0.021 / 0.658**, final ep16000 **0.000 / 0.658**.
- **Best checkpoint:** ep11000 reached **3/48 wins** (win **0.0625**, crystal **0.651**, exit unlock **0.104**, target progress **0.334**), but this was not sustained: ep12000 fell back to ~1/48 and final ep16000 was 0/48.
- **Final death/stall trace:** killed **0.438**, stalled **0.521**, timeout **0.042**. Deaths remain enemy-heavy (enemy **0.396**, hazard **0.042**). Stalls are far/trapped: trapped **0.354**, near-objective **0.000**, far **1.000**, mean geodesic distance **16.5** tiles, crystals at stall **0.703**.
- **Read:** 4000 episodes was somewhat short for crystal collection, but not for the completion decision. More budget produced occasional canonical wins and better crystals, yet no reliable finish conversion. This makes "just train longer" a weak lever for this config class; next work should change policy competence/control/planning or continue the demo/Tier-2 path, not another same-family longer run.
- **Artifact gap:** `diagnose_gap` evaluates per-level rows internally but only persists aggregate summaries; no checkpoints were saved, so the requested per-level-at-best table cannot be reconstructed after the fact. Before the next long run, persist per-level eval rows at checkpoints or save checkpoints for post-hoc `level_set_eval`.
