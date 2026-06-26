# Crystal Caves NN Experiment Tracker

**Last updated:** 2026-06-25
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
| B3s correction fine-tune v1 | `.Codex/artifacts/cc_sessions/20260624_215710_b3s_correction_finetune_b3s_ep300_1024_300` | 3/16 | 18.8% | 29.5% | policy-visited correction loss was active but regressed route performance; not promoted |
| B3v final-contact option eval | `.codex/artifacts/cc_sessions/20260624_222911_b3s_final_contact_option_eval30` | 13/30 | 43.3% | 50.5% | eval-only split policy improved first-crystal wins; validation 24/60, but gate returned `REGRESS` because depth 53.6% missed the 57.0% guardrail |
| B3w final-contact distance 1.5 | `.Codex/artifacts/cc_sessions/20260624_230035_b3s_final_contact_option_d15_commit8_eval30` | 8/30 | 26.7% | 61.9% | narrower trigger restored depth but under-fired; `REGRESS` vs B3s on selected wins |
| B3x final-contact commit 4 | `.Codex/artifacts/cc_sessions/20260624_230324_b3s_final_contact_option_d30_commit4_eval30` | 11/30 | 36.7% | 50.9% | selected `HOLD` vs B3s but still below B3v wins and fails depth profile; not worth expanded validation yet |
| B3y final-contact cancel outside | `.Codex/artifacts/cc_sessions/20260624_231322_b3s_final_contact_option_d30_commit8_cancel_eval30` | 13/30 | 43.3% | 50.5% | tied B3v selected metrics and improved instrumentation, but cancelled only 61 queued actions; not enough to fix depth |
| B4 contact-only correction weight 0.010 | `.Codex/artifacts/cc_sessions/20260625_030453_b4_contact_only_correction_finetune_w010_300` | 1/8 | 12.5% | 20.5% | interrupted at ep155 after ep150 eval; correction loss fit labels but route depth regressed; not promoted |
| B4 contact-only correction weight 0.005 | `.Codex/artifacts/cc_sessions/20260625_031606_b4_contact_only_correction_finetune_w005_300` | 0/8 | 0.0% | 16.1% | interrupted at ep102 after ep100 eval; lower weight still regressed route/contact; close this correction-loss lane |
| B5 anchored contact correction 0.02 | `.Codex/artifacts/cc_sessions/20260625_070641_b5_anchored_contact_correction_w005_a002_300` | 1/8 | 12.5% | 25.0% | interrupted at ep101 after ep100 eval; anchor active but too weak to preserve route; not promoted |
| B5 anchored contact correction 0.10 | `.Codex/artifacts/cc_sessions/20260625_071613_b5_anchored_contact_correction_w005_a010_300` | 2/8 | 25.0% | 22.3% | interrupted at ep108 after ep100 eval; stronger anchor improved teacher match but still failed depth; not promoted |
| B6 contact interleaved fixed levels | `.Codex/artifacts/cc_sessions/20260625_075416_b6_contact_interleaved_25pct_300` | 0/16 | 6.2% final, 12.5% selected-source | 35.7% final, 39.3% selected-source | early-stopped after ep150 source eval; contact lanes reached 100% but route/contact score was `0.799` vs B3s `1.821`, so this is a route regression, not a contact breakthrough |
| B7 generated contact-pool interleaved | `.Codex/artifacts/cc_sessions/20260625_084134_b7_contact_pool_interleaved_12pct_300` | 0/16 | 25.0% | 30.8% | early-stopped after ep100 source eval; generated contact pool improved over fixed B6 but still scored `1.308` vs B3s `1.821` and failed depth preservation |
| B8 history-state smoke | `.Codex/artifacts/cc_sessions/20260625_085713_b8_history_state_smoke` | 0/1 | 0.0% | 85.7% | mechanics smoke only; validated opt-in 323-feature state, demo BC, selected checkpoint save/eval, route/contact scorecard, and artifact validation |
| B8 history-state full | `.Codex/artifacts/cc_sessions/20260625_085924_b8_history_state_conservative_pool512_select30_300` | 10/30 | 33.3% | 59.0% | tied B3s wins/crystals but route/contact score `1.757` vs B3s `1.821`; compare gate `REGRESS` because only min target distance improved while depth/score regressed |
| B9 C51 distributional smoke | `.Codex/artifacts/cc_sessions/20260625_113254_b9_c51_distributional_smoke` | 0/1 | 0.0% | 92.9% | mechanics smoke only; validated opt-in C51 head/loss, selected checkpoint save/eval, config snapshot, and artifact validation |
| B9 C51 distributional full | `.Codex/artifacts/cc_sessions/20260625_113409_b9_c51_distributional_pool512_select30_300` | 10/30 | 33.3% | 59.8% | tied B3s wins/crystals but route/contact score `1.714` vs B3s `1.821`; compare gate `REGRESS` because depth, loop-after-close, and score regressed |
| B10 advantage-gated final-contact smoke | `.Codex/artifacts/cc_sessions/20260625_121800_b10_final_contact_advantage_gate_smoke/20260625_121346_b10_final_contact_advantage_gate_smoke` | 2/4 | 50.0% | 64.3% | mechanics smoke only; old B3s checkpoint compatibility bug found/fixed; gate rejected `24/27` option candidates and artifact validation passed |
| B10 advantage-gated final-contact selected | `.Codex/artifacts/cc_sessions/20260625_122000_b10_final_contact_advantage_gate_eval30/20260625_121437_b10_final_contact_advantage_gate_eval30` | 15/30 | 50.0% | 53.6% | strongest selected contact signal yet; option takeover only `0.8%` of steps, route/contact score `2.702` vs B3s `1.821`; selected eval was `HOLD` before validation |
| B10 advantage-gated final-contact validation | `.Codex/artifacts/cc_sessions/20260625_122500_b10_final_contact_advantage_gate_val60/20260625_121631_b10_final_contact_advantage_gate_val60` | 30/60 | 50.0% | 54.3% | metric audit showed raw depth was biased by early successes; non-success depth `69.5%` cleared the route guardrail and updated `compare-artifact --validation` returns `PROMOTE` |
| B11 advantage-gate correction collect | `.Codex/artifacts/cc_sessions/20260625_125000_b11_advantage_gate_correction_collect/20260625_123917_b11_advantage_gate_correction_collect` | dataset | 162 labels | n/a | collected high-confidence B10 gate-accepted labels: `162/625` close-zone candidates kept, `80.2%` policy/label disagreement, `67.7%` gate rejection |
| B11 advantage-gate correction fine-tune | `.Codex/artifacts/cc_sessions/20260625_125500_b11_advantage_gate_correction_finetune/20260625_124115_b11_advantage_gate_correction_finetune_w001_a005_150` | 4/16 | 25.0% | 44.2% | learned labels (`78.8%` correction accuracy) but route/contact score fell to `1.223`; compare `HOLD` only because sample was 16 games, support metrics regressed |
| B12 metric audit / promotion gate fix | code/docs | n/a | n/a | n/a | added `metric-audit` and outcome-conditioned non-success depth to promotion snapshots; B10 is now promoted as the best eval-time controller baseline, while B3s remains the pure-NN training baseline |
| B13 route-masked correction fine-tune | `.Codex/artifacts/cc_sessions/20260625_134602_b13_route_masked_correction_finetune_w001_a010_150` | 4/16 | 25.0% | 60.3% | kept B11 wins and recovered route depth; correction accuracy `80.4%`, non-success depth `64.9%`, but route/contact score `1.228` still below B3s due to stuck/loop-after-close |
| B14 detached contact-head fine-tune | `.Codex/artifacts/cc_sessions/20260625_144008_b14_contact_head_finetune_w002_150` | 1/16 | 6.2% | 37.5% | learned local labels (`80.2%` head accuracy) but selector eval regressed hard; head fired on `32.1%` of steps and overused `JUMP` (`11139` head actions), route/contact score `0.344` |
| B15 offline/head-only contact selector selected | `.Codex/artifacts/cc_sessions/20260625_150413_b15_contact_head_offline_balanced_conf075_500_eval30` | 13/30 | 43.3% | 56.9% | **promoted learned adapter candidate**; frozen B3s route weights, balanced offline head fit, confidence-gated selector fired on only `0.5%` of actions; route/contact score `2.302` vs B3s `1.821` |
| B15 offline/head-only contact selector validation | `.Codex/artifacts/cc_sessions/20260625_150521_b15_contact_head_offline_balanced_conf075_500_val60` | 22/60 | 36.7% | 57.0% | expanded validation clears B3s (`19/60`) and non-success depth guardrail (`69.4%` vs B3s `69.9%`); `compare-artifact --validation` returns `PROMOTE` |
| B3s seed-1 selected control | `.Codex/artifacts/cc_sessions/20260625_162938_b3s_selected_seed1_eval30` | 8/30 | 26.7% | 54.3% | same-seed control for B15 robustness check; seed 1 is materially harder than the frozen seed-0 reference |
| B15 seed-1 selected check | `.Codex/artifacts/cc_sessions/20260625_162756_b15_contact_head_offline_balanced_conf075_500_seed1_eval30` | 8/30 | 26.7% | 52.9% | ties same-seed B3s wins/crystals and barely improves scorecard (`1.362` vs `1.359`), but trails frozen B3s; second seed is neutral, not a robustness promotion |
| B16 jump-gated contact head seed-1 check | `.Codex/artifacts/cc_sessions/20260625_164745_b16_contact_head_jump_conf085_seed1_eval30` | 9/30 | 30.0% | 54.3% | stricter `0.85` confidence for jump-variant head actions improves same-seed B3s/B15 by one win and scorecard (`1.526` vs `1.359`/`1.362`), but compare remains `HOLD` pending stronger evidence |
| B16 jump-gated contact head seed-0 guardrail | `.Codex/artifacts/cc_sessions/20260625_164903_b16_contact_head_jump_conf085_seed0_eval30` | 12/30 | 40.0% | 57.6% | still beats B3s seed 0, but regresses versus B15 seed 0 (`13/30`, score `2.302` -> `2.193`); do not promote over B15 or run expanded validation yet |
| B17 hard-seed B10-gated label collect | `.Codex/artifacts/cc_sessions/20260625_165300_b17_advantage_gate_correction_collect_seed1` | dataset | 222 labels | n/a | seed-1 dataset-only pass; kept more labels than B11 (`222` vs `162`) and adds much more `LEFT_JUMP`/`RIGHT_JUMP` coverage (`51`/`60` vs `18`/`21`), with `83.8%` disagreement and `82.4%` gate rejection |
| B18 combined contact-head calibration | `.Codex/artifacts/cc_sessions/20260625_173639_b18_contact_head_combined_calibration_b11_b17` | calibration | 384 labels | n/a | combines B11+B17 into `288` train / `96` held-out labels; route weights stayed fixed, but calibration accuracy was only `58.3%` vs `70%` gate, with `LEFT` `23.5%` and `LEFT_JUMP` `52.9%`; **do not run a selected eval from this head yet** |
| B19 contact-label quality audit | `.Codex/artifacts/cc_sessions/20260625_180033_b19_contact_label_quality_audit_b11_b17` | audit | 384 labels | n/a | no rounded duplicate-state conflicts, but severe label instability: `36` semantic ambiguity groups covering `295/384` labels, `76/313` adjacent-frame flips (`24.3%`), and `87/242` horizontal direction mismatches (`36.0%`); confirms B18 failed because one-step labels are noisy/phase-dependent |
| B20 stable contact-label filter | `.Codex/artifacts/cc_sessions/20260625_180829_b20_stable_contact_label_filter_b11_b17` | filter | 117 labels | n/a | kept only `67%+` semantic-majority rows matching the majority label and dropped adjacent-flip rows; retained all five classes (`RIGHT` min class `13`) |
| B20 stable-label contact calibration | `.Codex/artifacts/cc_sessions/20260625_180944_b20_stable_contact_head_calibration_b11_b17` | calibration | 88 train / 29 held-out | n/a | held-out label accuracy improved from B18 `58.3%` to `82.8%`; route delta `0.00e+00`; calibration decision `pass` |
| B21 stable-label contact selector selected | `.Codex/artifacts/cc_sessions/20260625_181012_b21_stable_contact_head_offline_conf075_eval30` | 13/30 | 43.3% | 56.7% | ties B15 selected wins but improves route/contact score (`2.333` vs `2.302`) and loop-after-close (`26.7%` vs `30.0%`); selected result required expanded validation |
| B21 stable-label contact selector validation | `.Codex/artifacts/cc_sessions/20260625_181129_b21_stable_contact_head_offline_conf075_val60` | 23/60 | 38.3% | 56.9% | **promoted learned adapter baseline**; beats B15 validation wins (`23/60` vs `22/60`) and score (`2.036` vs `1.920`); raw depth misses by `0.001`, but metric audit shows non-success route depth `70.8%` clears B3s guardrail `66.9%`; fixed promotion snapshot extraction so `compare-artifact --validation` returns `PROMOTE` |
| B21 stable-label seed-1 selected | `.Codex/artifacts/cc_sessions/20260625_182515_b21_stable_contact_head_offline_conf075_seed1_eval30` | 10/30 | 33.3% | 50.9% | improves same-seed B3s/B15 selected wins (`8/30` -> `10/30`) and score (`1.693` vs `1.359`/`1.362`), but raw/non-success depth regress; required matched validation |
| B21 stable-label seed-1 validation | `.Codex/artifacts/cc_sessions/20260625_182705_b21_stable_contact_head_offline_conf075_seed1_val60` | 18/60 | 30.0% | 53.0% | modest seed-1 robustness lift vs matched B3s seed-1 val60 (`16/60`, `26.7%`, `55.5%`), but route depth regresses (`non-success 62.1%` vs `64.4%`); selector heavily overuses `LEFT_JUMP` (`2265` accepted actions) |
| B3s seed-1 validation control | `.Codex/artifacts/cc_sessions/20260625_182902_b3s_selected_seed1_val60_control` | 16/60 | 26.7% | 55.5% | matched 60-game seed-1 control for B21 robustness; confirms B21 seed-1 lift is `+2/60` wins with route-depth cost |
| B22 stable-label jump confidence 0.90 seed-1 selected | `.Codex/artifacts/cc_sessions/20260625_183234_b22_stable_contact_head_jump_conf090_seed1_eval30` | 9/30 | 30.0% | 51.2% | rejected; global jump threshold reduced jump firing and improved loop/non-success depth, but lost one win versus B21 seed-1 selected (`10/30` -> `9/30`) |
| B23 stable-label jump confidence 0.85 seed-1 selected | `.Codex/artifacts/cc_sessions/20260625_183355_b23_stable_contact_head_jump_conf085_seed1_eval30` | 9/30 | 30.0% | 51.2% | rejected; milder global jump threshold still trailed B21 selected (`REGRESS`), so do not spend on global jump-threshold validation |
| B24 per-action `LEFT_JUMP` confidence 0.90 seed-1 selected | `.Codex/artifacts/cc_sessions/20260625_184641_b24_stable_contact_head_left_jump_conf090_seed1_eval30` | 10/30 | 33.3% | 50.2% | tied B21 seed-1 selected wins and improved route/contact score (`1.719` vs `1.693`) plus loop-after-close (`23.3%` vs `26.7%`); selected comparator returned `HOLD`, so expanded validation was required |
| B24 per-action `LEFT_JUMP` confidence 0.90 seed-1 validation | `.Codex/artifacts/cc_sessions/20260625_184808_b24_stable_contact_head_left_jump_conf090_seed1_val60` | 19/60 | 31.7% | 52.6% | small positive versus B21 seed-1 val60 (`18/60`, score `1.555` -> `1.643`) and much less `LEFT_JUMP` over-fire (`312` accepted vs B21 `2265`), but non-success depth still trails B21/B3s (`61.7%` vs `62.1%`/`64.4%`); do not promote as new baseline |
| B24 per-action `LEFT_JUMP` confidence 0.90 seed-0 guardrail | `.Codex/artifacts/cc_sessions/20260625_185016_b24_stable_contact_head_left_jump_conf090_seed0_eval30` | 13/30 | 43.3% | 56.7% | did not break B21 seed-0 selected wins/depth/score, but did not improve enough to promote; formal comparator returned `REGRESS` because tied wins had fewer than two support improvements |
| B25 B24-policy-visited contact label collect | `.Codex/artifacts/cc_sessions/20260625_190140_b25_b24_policy_visited_contact_collect_seed1` | dataset | 225 labels | n/a | collected B10 advantage-gated labels from states visited by the B24-style contact-head selector; all five classes represented (`JUMP 71`, `LEFT 42`, `LEFT_JUMP 30`, `RIGHT 32`, `RIGHT_JUMP 50`), `79.2%` disagreement, gate rejected `1115/1399` candidates |
| B26 B20+B25 contact-label audit/filter | audit/filter | 184 stable labels | n/a | aggregated B20+B25 into `342` labels; audit still showed `30` semantic ambiguity groups and `24.8%` adjacent flips; B20-style filter kept `184` stable labels with all five classes represented |
| B27 B20+B25 stable-label calibration | `.Codex/artifacts/cc_sessions/20260625_190527_b27_b20_b25_stable_contact_head_calibration` | calibration | 137 train / 47 held-out | n/a | calibration improved modestly over B20 (`82.8%` -> `85.1%`) and passed, but mean confidence dropped (`0.826` -> `0.730`), especially `LEFT 0.616` and `RIGHT 0.686` |
| B28 B20+B25 stable head selected | `.Codex/artifacts/cc_sessions/20260625_190558_b28_b20_b25_stable_contact_head_left_jump_conf090_seed1_eval30` | 9/30 | 30.0% | 50.0% | rejected; despite better calibration, game eval trailed B24 (`10/30`); selector over-fired `JUMP` (`1477` accepted head actions) and almost stopped using `LEFT` (`2`) |
| B29 B20+B25 stable head with `JUMP`+`LEFT_JUMP` gates | `.Codex/artifacts/cc_sessions/20260625_190718_b29_b20_b25_stable_contact_head_jump_leftjump_conf090_seed1_eval30` | 9/30 | 30.0% | 50.0% | rejected; stricter `JUMP` gate fixed the jump over-fire (`1477` -> `9`) but did not recover wins, confirming the aggregate head lost useful behavior beyond jump spam |
| B30 B20+B25 class-calibrated thresholds | `.Codex/artifacts/cc_sessions/20260625_190858_b30_b20_b25_stable_contact_head_class_calibrated_seed1_eval30` | 9/30 | 30.0% | 50.0% | rejected; lowering base threshold to recover `LEFT` produced `523` accepted `LEFT` actions but still trailed B24, so the B26 aggregate labels should not be used as direct head-training data |

## Updated Ranked Backlog

This replaces the older A1-A8 backlog. Several items from that list have now been
implemented and rejected, so the active list should route future work away from repeated
correction/option variants.

## Next Recommendation

**Current top recommendation:** use B10 as the promoted eval-time controller baseline,
but keep B3s as the pure-NN training baseline. The metric audit found that B10's raw
depth regression is mostly an early-success artifact: successful first-crystal episodes
end around `39%` depth, while failed B10 validation episodes still average `69.5%`
depth, essentially matching B3s's `69.9%` non-success validation depth. The promotion
gate now records outcome-conditioned non-success depth and treats B10 as `PROMOTE`.

**Latest useful NN work:** B21 supersedes B15 as the best learned contact-head adapter.
It starts from the same frozen B3s route policy, but trains the detached contact head on
the B20 stable-label subset (`117` labels) instead of raw B11 labels. Calibration improved
from B18 `58.3%` to `82.8%`, selected eval tied B15 at `13/30`, and expanded validation
improved to `23/60` versus B15 `22/60` and B3s `19/60`. Route weights did not move
(`route delta 0.00e+00`). `compare-artifact --validation` now returns `PROMOTE` after the
promotion snapshot fix reads artifact-level non-success depth.

- B3s pure-NN baseline for learned-policy comparisons.
- B21 learned contact-head adapter baseline for B3s+NN-selector comparisons.
- B10 eval-time controller baseline for best current first-crystal outcomes.

**Next lane:** B20/B21 confirmed that data cleanliness mattered more than raw label count:
filtering raw B11+B17 from `384` labels to `117` stable labels lifted held-out calibration
from `58.3%` to `82.8%` and produced a promoted learned adapter. The seed-1 robustness
check is now done: B21 improves matched seed-1 validation from `16/60` to `18/60`, but
route depth regresses and the selector overuses `LEFT_JUMP` on seed 1. B22/B23 then showed
that a broad jump-variant threshold (`0.90` or `0.85`) over-corrects: it improves some
route/depth support metrics but loses the B21 win lift. Do not run more raw-label
contact-head evals or broad jump-threshold validation. B24 added that targeted
**per-action threshold** path and showed it is safer than broad jump gating, but it is not
enough: seed-1 validation improved only `18/60` -> `19/60` and route depth still trails
B3s. B25-B30 then tested the policy-visited aggregation path. It improved offline
calibration, but direct training on the aggregate labels regressed selected game outcomes.
Do not promote B26/B28-B30. The next highest-value path is source-aware/phase-aware label
use: keep B20 as the positive head-training anchor, and use B25 policy-visited data for
diagnostics, rejection thresholds, or sequence/phase-aware labels rather than as a flat
single-action training set.

### Strategic Reanalysis After B30

**Current performance picture:**

- Pure B3s remains the route-preserving neural baseline: seed 0 `19/60`, seed 1 matched
  control `16/60`.
- B10 remains the best overall eval-time controller: `30/60`, but it is not pure learned
  policy behavior.
- B21 is the best learned NN contact adapter: seed 0 `23/60`, seed 1 `18/60`. It improves
  wins on both seeds, but seed 1 shows route-depth cost.
- B22/B23 show the latest failure mode is not "all jump actions are bad." A global jump
  threshold reduced useful contact actions and lost wins.
- B24 shows class-specific `LEFT_JUMP` gating is safer than broad jump gating: it preserved
  B21 seed-1 selected wins, improved seed-1 validation to `19/60`, and reduced accepted
  `LEFT_JUMP` actions from `2265` to `312`. It still did not recover non-success route
  depth, so it is an evidence-positive tweak, not a promoted baseline.
- B25-B30 show policy-visited aggregation is not automatically beneficial. B25 collected
  useful-looking labels and B27 improved held-out label accuracy to `85.1%`, but B28-B30
  all regressed to `9/30`. The mismatch is between label calibration and game behavior:
  the aggregate head changed selector action distribution too much.

**What is actually working:**

- Stable-label filtering worked. B20 improved held-out label calibration from `58.3%` to
  `82.8%`, then B21 translated that into real game gains.
- Frozen-route/head-only adaptation worked better than online fine-tuning. Route delta
  stayed `0.00e+00`, avoiding the broad policy damage seen in B14 and earlier correction
  fine-tunes.
- Outcome-conditioned metric audit is required. Raw depth can look worse because success
  episodes end earlier; non-success depth is the safer route-preservation signal.

**What is not worth more time right now:**

- Raw B11+B17 contact-head training: B18 and B19 already showed label instability.
- Broad confidence thresholds: B22/B23 missed the class-specific nature of the problem.
- Threshold-only selector tuning: B24 is useful as a guardrail/tooling improvement, but the
  remaining error is not solved by confidence gating alone.
- Directly training the contact head on B26 aggregate labels: B28 over-fired `JUMP`, B29
  suppressed `JUMP` but still lost wins, and B30 restored `LEFT` actions but still lost.
  The B25 data should not be treated as ordinary positive imitation labels.
- More reward-terminal tuning: prior win-reward changes did nothing because the agent rarely
  reaches the terminal event.
- Generic architecture sweeps before fixing selector calibration: B8 history state and B9
  C51 did not move the failure mode.

**Ranked next work:**

1. **Source-aware/phase-aware contact-label work.** Do not flatten B20+B25 into one
   balanced action dataset again. Keep B20 as the positive anchor and use B25 rows to learn
   when to reject/raise thresholds, or convert close-zone trajectories into short sequence
   labels so phase-dependent `JUMP`/`LEFT` decisions are not treated as interchangeable.
2. **Head selector diagnostics before another game eval.** Add per-action accepted/rejected
   confidence histograms and success-conditioned head-action counts. B28-B30 prove aggregate
   calibration accuracy is not enough; the decisive signal is how the selector action mix
   changes in successful versus failed games.
3. **Keep B24 per-action gates available as a selector guardrail.** They are useful
   infrastructure and should stay documented, but do not promote B24 over B21 yet.
4. **Full-level completion work.** After first-crystal contact stabilizes, shift from
   first-crystal terminal evals back toward collect-all-and-exit completion. Do not confuse
   first-crystal gains with full Crystal Caves completion yet.

**Why this is next:** B6 proved the model can master fixed contact training lanes while
still failing held-out tutorial caves. That means the next unknown is not "can it learn
contact in isolation?" but **where the full-policy failure happens**:

- does the route policy stop reaching the first objective?
- does it reach close-zone but choose the wrong action?
- does it collect the first crystal but fail to preserve route depth afterward?
- does a candidate increase contact while damaging depth, like B3v/B4/B5/B6?

The scorecard and metric audit are now implemented and backfilled. They report:

- B3s conservative demo-Q: route/contact score `1.821`, verdict `contact regression`;
  it preserves route depth (`60.5%`) and reaches close-zone (`60.0%`) but loops/stalls
  after close contact. Metric audit: selected non-success depth `71.4%`; validation
  non-success depth `69.9%`.
- B6 fixed contact interleave: route/contact score `0.799`, verdict `route regression`;
  it mastered contact lanes but selected-source held-out route progress fell to `12.5%`
  first-objective and `39.3%` depth.

**Next NN implication:** B6 and B7 both show contact lanes can be learned while held-out
route depth degrades. B7's generated pool was less bad than fixed B6 (`1.308` vs `0.799`
route/contact score), but still below B3s (`1.821`) with `30.8%` depth. B8 showed that
short action/approach history is safe and nearly neutral, but it did not fix the contact
decision: selected close-zone jump rate stayed `0%`, loop-after-close stayed `33.3%`,
and the route/contact score stayed below B3s. B9 showed that replacing scalar DQN with
an opt-in C51 value distribution also does not change the close-zone action failure:
selected wins/crystals tied B3s, but route/contact score fell to `1.714` and
loop-after-close worsened. B10 showed local action control can be made much more selective
than B3v/B3y: option takeover dropped from roughly `30-36%` of eval steps to `0.8%`, while
selected wins rose to `15/30` and validation wins rose to `30/60`.
B11 then proved the high-confidence labels are not enough by themselves: a low-weight
anchored correction fine-tune learned the labels but regressed held-out crystal/depth.
B12 showed the earlier B10 depth failure was a metric artifact: B10 selected non-success
depth was `72.4%`, validation non-success depth was `69.5%`, and the updated promotion
gate now promotes B10 while keeping raw depth regression visible.
B13 narrowed the failure: masking the frozen-policy anchor outside the close zone fixed
the B11 route-depth regression (`44.2%` -> `60.3%`) while keeping correction accuracy high
(`80.4%`), but completion stayed at `4/16` and loop/stuck-after-close worsened. That
points away from more scalar weight sweeps and toward a separate learned contact action
selector or gated network head.
B14 tested that separate head directly. The head learned the labels (`80.2%`), but the
selector regressed to `1/16` first-objective wins and `37.5%` depth, with most head
actions being `JUMP`. This means the mechanism should not be discarded, but the training
recipe should change: freeze B3s, train the head offline, balance classes, and gate by
confidence before using it in held-out eval.
B15 made that recipe change and is the first promoted learned adapter after B3s: selected
`13/30`, validation `22/60`, confidence-gated head action rate `0.5-0.6%`, and validation
non-success depth `69.4%`. It still trails B10's eval-time controller on validation wins
(`22/60` vs `30/60`), so B10 remains the best overall controller outcome while B15 becomes
the best learned contact-head adapter on seed 0. A same-method seed-1 check tied B3s at
`8/30` and should be treated as neutral rather than confirmation that B15 is robust across
seeds.

### Already Built; Do Not Rebuild

- Comparable status-session runner, isolated artifacts, live metrics, recipes, reports,
  and `compare-artifact`.
- B3s conservative demo-Q baseline and selected checkpoint workflow.
- Drill, bridge, interleaved drill, bridge-interleaved, reverse-start, archive-start,
  novelty, invalid-action, and final-contact-option experiment modes.
- Near-miss/contact diagnostics and selected-checkpoint failure diagnostics.
- Correction dataset collection, correction fine-tune, B4 contact-only dataset, and B5
  policy-anchor provider.

### R1. B6 Staged Contact Curriculum

**Status:** implemented, smoke-tested, then real run early-stopped. Fixed contact-level
interleaving is not promoted.

**Problem it targets:** B3s is the best route policy (`10/30`, `60.5%` depth). B3v/B3y
show close-zone control can raise first-crystal contact, but eval-time options take over
too much route behavior. B4/B5 show supervised contact labels are fitted but hurt route
depth. The missing method is contact practice through normal game rewards, not blended
action-label pressure.

**What already exists to reuse:**

- `B3s` selected checkpoint as the route-policy starting point.
- Interleaved vector-lane infrastructure for full lanes plus skill lanes.
- Drill/bridge config patterns.
- Final held-out eval and near-miss diagnostics.
- Recipe infrastructure and artifact validation.

**What was added on 2026-06-25:**

1. Added training-only contact levels, separate from final/eval levels:
   - `contact_floor`: crystal and exit on same floor, short horizontal approach.
   - `contact_jump_up`: crystal/exit requires one clean jump-up.
   - `contact_drop_return`: drop to crystal, climb or jump back to exit.
   - `contact_step_pair`: two short jumps, not a full staircase.
   - `contact_exit_after_crystal`: crystal collected first, exit one ledge away.
2. Added `CRYSTAL_CAVES_CONTACT_LEVELS`, `contact_config`, and
   `make_interleaved_contact_config` so contact levels stay separate from drills,
   bridges, and final procedural tutorial levels.
3. Added `contact-interleaved` mode:
   - restore B3s selected weights;
   - use `6` normal tutorial lanes and `2` contact lanes at `vec-envs=8`;
   - train with normal Crystal Caves rewards only;
   - do not enable correction, close-zone, or policy-anchor losses.
4. Added `b6_contact_interleaved` recipe:
   - `episodes=300`;
   - `seed=0`;
   - `eval_every=50`;
   - `train_eval_games=8`;
   - `selected_eval_games=30`;
   - `eval_games=16`;
   - `save_selected_checkpoint=True`;
   - required override: `--checkpoint` pointing to B3s selected weights.
5. Added contact-lane metrics to live JSON, summary JSON, and markdown reports:
   - `contact_lane_win_rate_100`;
   - `contact_lane_crystal_rate_100`;
   - `contact_lane_exit_rate_100`;
   - `full_lane_progress_100`;
   - `full_lane_first_crystal_rate_100`.
6. Added tests for contact cave well-formedness/solvability, contact config isolation,
   contact interleaving, recipe guardrails, and selected-checkpoint transfer wording.

**Not yet added:** randomized contact-layout variants. The first B6 build uses five
fixed compact contact caves so the plumbing and metric lane can be validated before
adding another moving part.

**Smoke artifact:** `.Codex/artifacts/cc_sessions/20260625_074529_b6_contact_interleaved_smoke_20`

- Command shape: `run-recipe b6_contact_interleaved --checkpoint <B3s selected>`,
  overridden to `20` episodes, tiny eval counts, and `eval_every=10`.
- Artifact validation: passed.
- Contact lane metrics: `100%` win, `100%` crystal, `100%` exit over `17` contact
  lane episodes.
- Full training lane metrics: `0.981` progress and `100%` first-crystal rate over
  `3` completed full-lane episodes.
- Held-out eval is not meaningful at this budget: final `0/4` wins, `0%` crystals,
  `10.7%` depth; selected ep10 `0/4` wins, `0%` crystals, `32.1%` depth.
- Interpretation: B6 plumbing works and the new metrics surface early lane learning.
  This smoke does not promote or reject the method.

**Real-run artifact:** `.Codex/artifacts/cc_sessions/20260625_075416_b6_contact_interleaved_25pct_300`

- Command shape: full `b6_contact_interleaved` recipe from the B3s selected checkpoint.
- Stopped manually after the ep150 source eval because it failed the documented
  continuation rule. The interrupt landed at ep182. A recovered partial `summary.json`,
  `report.md`, and `artifact_validation.json` now exist; validation passes with the
  expected interrupted-partial warning.
- Source eval history:
  - ep50: `0/16` wins, `12.5%` crystals, `36.2%` depth.
  - ep100: `0/16` wins, `12.5%` crystals, `39.3%` depth.
  - ep150: `0/16` wins, `6.2%` crystals, `35.7%` depth.
- Latest quick trainer eval at ep150 was worse: `0/8` wins, `0%` crystals, `12.5%`
  depth.
- Live metrics at interruption:
  - contact lanes: `100%` win, `100%` crystal, `100%` exit over the last 100 contact
    episodes;
  - full lanes: `55.6%` progress and `53.8%` first-crystal rate over 26 completed
    full-lane episodes.
- Interpretation: the training-only contact levels are too easy or too disconnected from
  the held-out tutorial route. The agent masters contact lanes, but that mastery does
  not transfer into full procedural caves. This reproduces the earlier pattern: training
  skill success can look excellent while held-out Crystal Caves remains weak.

**Do not rerun this exact fixed-level B6 recipe as-is.** It is now evidence that simple
contact interleaving alone is not the missing lever.

**Original real-run command, kept for reproducibility only:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b6_contact_interleaved \
  --checkpoint .codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300/tutorial_demo_conservative/models/crystal_caves/tutorial_demo_conservative_selected_ep300.pth
```

**Early-stop rule used:**

- At ep50: continue only if contact lanes show learning and full-lane progress is not
  clearly collapsing.
- At ep100: stop if held-out depth is below `35%` and selected crystals are below B3s.
- At ep150: stop unless depth is recovering toward the B3s guardrail or wins/crystals
  are materially above B3s.

**Promotion rule:**

- Selected eval must beat B3s `10/30` first-crystal wins or tie it with materially better
  close-zone/contact metrics.
- Expanded validation must beat or tie B3s `19/60` while keeping depth within `3`
  percentage points of B3s validation depth.
- Reject any run that improves contact but repeats the B3v/B4/B5 pattern of low depth.

### R2. Unified Route/Contact Scorecard + Composite Selection

**Status:** implemented and backfilled on 2026-06-25.

**What exists:** selected checkpoints, `compare-artifact`, near-miss eval, failure
traces, and live source-lane metrics already exist.

**Specific improvement:** package those metrics into one scorecard and select
checkpoints by a route-preserving score, not by wins alone:

`selection = 3.0 * first_crystal_rate + 1.5 * crystal_frac + 1.0 * depth_frac + 0.5 * close_zone_rate - 1.0 * loop_after_close_rate - 0.5 * stall_rate`

**Why:** many probes improved one headline metric while damaging depth. Selection should
make that tradeoff visible before final eval.

**Promotion value:** this does not directly improve the NN. It improves our ability to
avoid false positives and route the next NN change correctly.

**Implementation notes:**

- `experiments/cc_status/scorecard.py` builds the `route_contact_scorecard` block.
- Status-session summaries and reports now include the scorecard automatically.
- Partial/interrupted summaries also include the scorecard when live/source evals exist.
- Source checkpoint selection now uses the route/contact composite score instead of
  pure `(win, crystal, depth, score)` ordering.
- `compare-artifact` shows route/contact score support against the frozen B3s baseline.

### R3. Game-Faithful Contact Curriculum Variant

**Status:** implemented, smoke-tested, and early-stopped; not promoted.

**Specific test:** replace the five fixed contact maps with a larger randomized,
game-faithful contact pool:

- `contact_pool_size=128`;
- no exact repeated contact layouts inside an epoch;
- held-out contact eval pool of `32` layouts;
- place contact setups inside real procedural cave fragments when possible;
- still final-evaluate only on normal tutorial caves.

**Why:** fixed B6 contact lanes were mastered but did not transfer. A randomized or
fragment-based contact curriculum is the only contact-level follow-up worth testing.

**Implementation added on 2026-06-25:**

- `contact_pool_caves(pool_size, seed)` builds deterministic generated contact rooms.
- `CRYSTAL_CAVES_CONTACT_POOL_SIZE` and `CRYSTAL_CAVES_CONTACT_POOL_SEED` keep generated
  contact pools opt-in.
- `contact-interleaved` accepts `--contact-pool-size` and `--contact-eval-pool-size`.
- New recipes:
  - `b7_contact_pool_smoke`;
  - `b7_contact_pool_interleaved`.
- Reports show generated contact-pool train/eval sizes and held-out contact-pool eval
  rollups.

**Smoke artifact:** `.Codex/artifacts/cc_sessions/20260625_083828_b7_contact_pool_smoke_20`

- Artifact validation: ok.
- Training lanes: contact win/crystal/exit `92%/100%/100%`.
- Held-out contact-pool eval: `0%` win, `50%` any crystal, `50%` all crystals,
  progress `0.508`.
- Selected held-out tutorial eval was not meaningful at this budget: `0/4`, `0%`
  crystals, `10.7%` depth.

**Real-run artifact:** `.Codex/artifacts/cc_sessions/20260625_084134_b7_contact_pool_interleaved_12pct_300`

- Command shape: `run-recipe b7_contact_pool_interleaved --checkpoint <B3s selected>`.
- Mix: `7` full tutorial lanes, `1` generated contact-pool lane (`12.5%` contact).
- Contact pool: `128` generated training rooms; held-out contact-pool eval size `32`.
- Stopped at ep108 after the ep100 source eval failed the depth guardrail.
- Source eval history:
  - ep53: `0/16`, `18.8%` crystals, `35.3%` depth.
  - ep100: `0/16`, `25.0%` crystals, `30.8%` depth.
- Route/contact score: `1.308`, verdict `depth regression`, below B3s `1.821`.
- Lane metrics at interruption:
  - contact lane win/crystal/exit `92.5%/100%/100%`;
  - full-lane progress `0.436`;
  - full-lane first crystal `36.8%`.

**Decision:** do not promote B7 and do not run another plain contact-interleave variant.
Generated contact rooms helped relative to fixed B6, but the method still traded away
too much held-out route depth.

### R4. Route/Policy Architecture Probe

**Status:** implemented, smoke-tested, and full-run tested; not promoted.

**Why:** platformer timing and repeated close-zone loops may need short action history or
velocity/history beyond the current state.

**Specific minimum probe:** add an optional 4-frame metadata/action-history state
extension before changing algorithms.

**Implementation added on 2026-06-25:**

- `CRYSTAL_CAVES_HISTORY_STATE` and `CRYSTAL_CAVES_HISTORY_STEPS` stay opt-in.
- The default state remains checkpoint-compatible with B3s: `295` features.
- Enabling 4 history steps appends `28` metadata scalars and creates a `323`-feature
  state: recent idle/left/right/jump/shoot/interact flags plus normalized approach
  delta per step.
- `tutorial-demo-conservative` accepts `--history-state --history-steps 4`.
- New recipes:
  - `b8_history_state_smoke`;
  - `b8_history_state_conservative`.
- Because state size changes, B8 is a fresh architecture probe and cannot restore the
  B3s selected checkpoint directly.

**Smoke artifact:** `.Codex/artifacts/cc_sessions/20260625_085713_b8_history_state_smoke`

- Artifact validation: ok.
- Config snapshot: `history_state=true`, `history_steps=4`, `state_size=323`.
- Demo collection/BC worked with the expanded state: `9/16` scripted wins and `1625`
  transitions.
- Selected checkpoint save/eval, near-miss diagnostics, route/contact scorecard, and
  markdown/JSON reports all wrote successfully.
- Result was only `0/1` on a one-game smoke. Do not interpret this as performance
  evidence.

**Full artifact:** `.Codex/artifacts/cc_sessions/20260625_085924_b8_history_state_conservative_pool512_select30_300`

- Artifact validation: ok.
- Config/state evidence: `history_state=true`, `history_steps=4`, `state_size=323`.
- Source eval progression:
  - ep50: `1/16`, `6.2%` crystals, `36.2%` depth.
  - ep100: `3/16`, `18.8%` crystals, `55.4%` depth.
  - ep150: `3/16`, `18.8%` crystals, `60.7%` depth.
  - ep200: `3/16`, `18.8%` crystals, `64.3%` depth.
  - ep250: `3/16`, `18.8%` crystals, `58.0%` depth.
  - ep300: `4/16`, `25.0%` crystals, `62.9%` depth.
- Selected checkpoint eval: `10/30`, `33.3%` crystals, `59.0%` depth.
- Route/contact scorecard: `1.757`, `contact regression`, below B3s `1.821`.
- `compare-artifact`: `REGRESS`; B8 tied selected wins but improved only mean minimum
  target distance while regressing depth and route/contact score.

**Decision:** keep the history-state code as opt-in infrastructure, but do not promote
B8 and do not sweep history lengths by default. The trace still shows close-zone loop
behavior and `0%` close-zone jump rate, so the next mechanism should change the learning
algorithm/action-selection behavior rather than adding more short memory.

### R5. Algorithm Upgrade: Distributional or PPO-Style Probe

**Status:** first candidate implemented and rejected as a baseline promotion.

**Why:** current DQN has rare high-value wins and unstable greedy outcomes. But all
recent evidence now says data/state additions alone are not enough: the policy can reach
near contact, but the greedy action choice still loops or stalls instead of committing to
the needed jump/contact action.

**Implemented first candidate:** opt-in C51 distributional DQN with the existing B3s
recipe shape. The network still returns expected Q-values at the public `forward()`
boundary, while `Agent` uses C51 logits/probabilities and projected target
distributions when `USE_DISTRIBUTIONAL_DQN` is enabled.

Smoke artifact:
`.Codex/artifacts/cc_sessions/20260625_113254_b9_c51_distributional_smoke`

Full artifact:
`.Codex/artifacts/cc_sessions/20260625_113409_b9_c51_distributional_pool512_select30_300`

Full result:

- Source eval reached `5/16` crystals by ep101 and selected ep101.
- Selected checkpoint eval: `10/30`, `33.3%` crystals, `59.8%` depth.
- Route/contact scorecard: `1.714`, below B3s `1.821`.
- `compare-artifact`: `REGRESS`; selected wins/crystals tied B3s but depth,
  loop-after-close, and route/contact score regressed.
- Close-zone jump rate remained effectively absent (`0.1%`) while loop-after-close
  rose to `43.3%`.

**Decision:** keep the C51 implementation as opt-in architecture, but do not promote it,
do not mark the recipe recommended, and do not spend the next run sweeping atoms/support
ranges. The failure signature is still local action commitment near contact, not value
target expressiveness.

### R6. Action-Selection Control Probe: B10 Advantage-Gated Final Contact

**Status:** implemented, smoke-tested, selected-tested, validation-tested, and promoted
as the best eval-time controller baseline after the B12 metric audit.

**Why:** B3v/B3y proved local final-contact control increases first-crystal wins, but
they controlled about one-third of eval steps and hurt route depth. B10 changes the
mechanism: when the target is in the close zone, it simulates both the oracle local macro
and the NN policy rollout, and lets the option take over only when the option's simulated
score advantage clears `250`.

**Implementation details:**

- `close_zone_sequence_score(...)` now scores a specific local macro on a copied game
  state and remains side-effect-free.
- `final_contact_option_action(...)` accepts `gate_policy_advantage` and
  `min_option_advantage`.
- Reports record gate evaluations, rejections, rejection rate, and mean option advantage.
- C51 support buffers were changed to non-persistent buffers after the B10 smoke exposed
  that pre-C51 B3s checkpoints could no longer load.
- Recipes:
  - `b10_final_contact_advantage_gate_smoke`
  - `b10_final_contact_advantage_gate_eval`

Smoke artifact:
`.Codex/artifacts/cc_sessions/20260625_121800_b10_final_contact_advantage_gate_smoke/20260625_121346_b10_final_contact_advantage_gate_smoke`

- Result: `2/4` wins, `50.0%` crystals, `64.3%` depth.
- Gate: `27` evaluations, `24` rejections (`88.9%` rejected), option trigger rate `1.0%`.
- Artifact validation: `ok`.

Selected artifact:
`.Codex/artifacts/cc_sessions/20260625_122000_b10_final_contact_advantage_gate_eval30/20260625_121437_b10_final_contact_advantage_gate_eval30`

- Result: `15/30` first-crystal wins, `50.0%` crystals, `53.6%` depth.
- Route/contact scorecard: `2.702`, `depth regression`, above B3s `1.821`.
- Gate: `190` evaluations, `164` rejections (`86.3%` rejected), option trigger rate `0.8%`.
- `compare-artifact`: `HOLD`; promising selected result but required expanded validation.
- Metric audit: success depth `34.8%`; non-success depth `72.4%`. The low raw depth is
  mostly because more episodes end successfully early.

Validation artifact:
`.Codex/artifacts/cc_sessions/20260625_122500_b10_final_contact_advantage_gate_val60/20260625_121631_b10_final_contact_advantage_gate_val60`

- Result: `30/60` first-crystal wins, `50.0%` crystals, `54.3%` depth.
- Route/contact scorecard: `2.685`, `depth regression`.
- Gate: `496` evaluations, `444` rejections (`89.5%` rejected), option trigger rate `0.8%`.
- Metric audit: success depth `39.0%`; non-success depth `69.5%`. B3s validation
  non-success depth is `69.9%`, so B10 did **not** collapse the failed-episode route
  profile.
- Original `compare-artifact --validation`: `REGRESS`; validation wins improved strongly
  over B3s `19/60`, but raw validation depth `0.543` missed required `0.570`.
- Updated `compare-artifact --validation`: `PROMOTE`; raw depth regression remains a
  support regression, but non-success route depth `0.695` clears the outcome-conditioned
  route guardrail `0.669`.

Reproduction:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python experiments/cc_status_session.py metric-audit \
  .Codex/artifacts/cc_sessions/20260624_122423_b3s_selected_ep300_eval60 \
  .Codex/artifacts/cc_sessions/20260625_122500_b10_final_contact_advantage_gate_val60/20260625_121631_b10_final_contact_advantage_gate_val60

/Users/justin/.pyenv/versions/3.12.11/bin/python experiments/cc_status_session.py compare-artifact \
  .Codex/artifacts/cc_sessions/20260625_122000_b10_final_contact_advantage_gate_eval30/20260625_121437_b10_final_contact_advantage_gate_eval30 \
  --validation .Codex/artifacts/cc_sessions/20260625_122500_b10_final_contact_advantage_gate_val60/20260625_121631_b10_final_contact_advantage_gate_val60
```

**Decision:** promote B10 as the best current eval-time controller baseline. Do **not**
treat it as a pure-NN improvement. The active learned-policy comparison baseline remains
B3s until a network-only checkpoint matches or beats B10 without the eval-time option.

### R7. B11 Gate-Accepted Label Transfer

**Status:** implemented and tested; not promoted, and do not sweep correction weights by
default.

**Purpose:** test whether B10's high-confidence accepted decisions can teach the NN
directly without using the eval-time controller.

**Dataset collection artifact:**
`.Codex/artifacts/cc_sessions/20260625_125000_b11_advantage_gate_correction_collect/20260625_123917_b11_advantage_gate_correction_collect`

- `625` close-zone candidates evaluated.
- `423` rejected by the B10 advantage gate (`67.7%` rejection).
- `202` gate-accepted label candidates.
- `162` kept disagreement labels (`80.2%` policy/label disagreement).
- Label actions: `JUMP 70`, `LEFT 34`, `LEFT_JUMP 18`, `RIGHT 19`, `RIGHT_JUMP 21`.
- Artifact validation: `ok`.

**Fine-tune artifact:**
`.Codex/artifacts/cc_sessions/20260625_125500_b11_advantage_gate_correction_finetune/20260625_124115_b11_advantage_gate_correction_finetune_w001_a005_150`

- Restore B3s selected checkpoint.
- Train `150` episodes with correction action weight `0.001`, margin `0.6`, batch `32`.
- Add policy anchor weight `0.05`, temperature `1.5`.
- Correction label accuracy reached `78.8%` avg100, so the labels were being learned.
- Final 16-game eval: `4/16` first-crystal wins, `25.0%` crystals, `44.2%` depth.
- Route/contact scorecard: `1.223`, below B3s `1.821` and B10 `2.702`.
- `compare-artifact`: `HOLD` only because sample size was `16`; support metrics mostly
  regressed.

**Decision:** do not run more correction-weight or anchor-weight sweeps from this dataset
as the default next step. The failure is not label quality; the supervised correction loss
can fit the labels, but the resulting DQN policy still loses route/contact performance.
This points back to an objective/evaluation mismatch or to needing a different training
formulation than low-weight action-margin correction.

### R8. B13 Route-Masked Correction Transfer

**Status:** implemented and tested; not promoted, but keep the mechanism and diagnostics.

**Purpose:** test whether B11's failure came from the frozen-policy anchor fighting the
B10 contact labels inside close-zone states. B13 keeps the same B11 high-confidence
dataset and correction loss, but applies the frozen B3s policy anchor only when the
active target is at least `3.0` tiles away. In plain terms: preserve route behavior while
letting close-contact labels move.

**Smoke artifact:**
`.Codex/artifacts/cc_sessions/20260625_134345_b13_route_masked_smoke`

- Restore B3s selected checkpoint.
- Use the B11 `162`-state dataset.
- Train `5` episodes with `2` envs; artifact validation `ok`.
- Confirmed the report records `policy_anchor_min_target_distance_norm=0.0631` and
  `route mask >= 3.0 target tiles`.

**Fine-tune artifact:**
`.Codex/artifacts/cc_sessions/20260625_134602_b13_route_masked_correction_finetune_w001_a010_150`

- Restore B3s selected checkpoint.
- Train `150` episodes with correction action weight `0.001`, margin `0.6`, batch `32`.
- Add route-masked policy anchor weight `0.10`, temperature `1.5`, minimum target
  distance `3.0` tiles.
- Correction label accuracy reached `80.4%` avg100, so the local labels were learned.
- Policy-anchor teacher-action match was `47.1%` avg100 on masked route states.
- Final 16-game eval: `4/16` first-crystal wins, `25.0%` crystals, `60.3%` depth.
- Metric audit: non-success depth `64.9%`, clearing the `57.0%` route-depth guardrail;
  B11 non-success depth was only `49.4%`.
- Route/contact scorecard: `1.228`, still below B3s `1.821`, with stuck-after-close
  `43.8%` and loop-after-close `56.2%`.
- `compare-artifact`: `HOLD`; sample size was `16`, wins did not beat B3s, and support
  regressions remained in crystal rate, stuck-after-close, loop-after-close, and
  route/contact score.

**Decision:** keep the route mask and policy-anchor metadata because it fixed the route
depth regression relative to B11. Do not promote B13 and do not sweep B13 weights as the
next default. The remaining failure is not "can we fit the labels?" or "can we preserve
route depth?" It is that the learned Q policy still reaches close-zone and loops/stalls
instead of selecting the right final contact action. The next higher-value lane is a
separate learned close-zone action head or gated adapter trained on B10-accepted labels,
active only inside the same target-distance mask.

### R9. B14 Detached Contact-Head Selector

**Status:** implemented and tested; not promoted. Keep the infrastructure, but archive
this exact RL fine-tune recipe.

**Purpose:** test the R8 recommendation directly: train a separate close-zone action head
from the B10 gate-accepted labels and use that selector only inside the target-distance
mask, instead of pushing the main Q head with correction-action margins.

**Implementation:**

- Added opt-in `CRYSTAL_CAVES_CONTACT_ACTION_HEAD`.
- Added a detached `SpatialDQN.contact_action_logits(...)` head. The contact loss updates
  only the head parameters; it does not backprop through the shared route trunk.
- Added `contact-head-finetune` status-session mode and `b14_contact_head_finetune`
  recipe.
- Added selector stats: head actions, policy actions, head action counts, mean head
  confidence, and contact-head training loss/accuracy.
- Added checkpoint partial-load support so old B3s selected weights restore while the new
  head initializes fresh.

**Smoke artifact:**
`.Codex/artifacts/cc_sessions/20260625_143831_b14_contact_head_smoke`

- Train `5` episodes with `2` envs from B3s plus the B11 `162`-state dataset.
- Artifact validation: `ok`.
- Head accuracy reached `76.2%`.
- Selector was invoked (`1467` head actions), but early eval showed a strong over-jump
  pattern (`JUMP 1466`, `RIGHT_JUMP 1`).

**Fine-tune artifact:**
`.Codex/artifacts/cc_sessions/20260625_144008_b14_contact_head_finetune_w002_150`

- Restore B3s selected checkpoint.
- Train `150` episodes with contact action head weight `0.02`, batch `32`, active within
  `3.0` target tiles.
- Dataset: same B11 `162` high-confidence B10 labels.
- Head label accuracy reached `80.2%` avg100, so the local classifier learned the label
  set.
- Selector eval: `1/16` first-crystal wins, `6.2%` crystals, `37.5%` depth.
- Route/contact scorecard: `0.344`, below B13 `1.228` and B3s `1.821`.
- Selector use: `13255` head actions (`32.1%` action rate), mean confidence `0.703`.
- Head action mix: `JUMP 11139`, `RIGHT 854`, `LEFT 627`, `LEFT_JUMP 623`,
  `RIGHT_JUMP 12`.
- `compare-artifact` vs B13: `REGRESS`.

**Decision:** reject the B14 RL fine-tune selector. The useful signal is that the detached
head can learn labels without needing the main Q head, but the selector is too broad and
class-biased. The next version should be **head-only/offline**, not another full RL
fine-tune: freeze B3s, train only the contact head with balanced batches, add a confidence
threshold before overriding B3s, and evaluate immediately against B13/B3s before any
online RL updates.

### R10. B15 Offline/Head-Only Contact Selector

**Status:** implemented, selected-tested, validation-tested, and promoted as the best
learned contact-head adapter on top of B3s. It does not replace B10 as the best eval-time
controller outcome.

**Purpose:** test whether B14 failed because the online RL fine-tune moved too broadly and
the selector lacked a confidence gate. B15 freezes B3s, trains only
`contact_action_head.*` offline, balances the five action classes in the B11 label set,
and lets the head override B3s only inside the `3.0` tile target mask when softmax
confidence is at least `0.75`.

**Implementation:**

- Added `contact-head-offline` status-session mode and `b15_contact_head_offline` recipe.
- Added offline head-only training with explicit non-head route-weight delta reporting.
- Added class-balanced contact-head batches and dataset action-count reporting.
- Added confidence-gated selector fallback plus rejected-head-action counts.
- Offline runs now write `live_metrics.json/jsonl` so artifact validation stays strict.

**Smoke artifact:**
`.Codex/artifacts/cc_sessions/20260625_150206_b15_contact_head_offline_smoke`

- Trained `20` offline steps and evaluated `2` games.
- Artifact validation: `ok`.
- Route weights stayed fixed (`route delta 0.00e+00`).
- Confidence gate rejected all weak head decisions, so the smoke under-fired as expected
  rather than repeating B14's broad override.

**Selected artifact:**
`.Codex/artifacts/cc_sessions/20260625_150413_b15_contact_head_offline_balanced_conf075_500_eval30`

- Trained `500` offline head-only steps from the B11 `162`-state dataset.
- Dataset action counts: `JUMP 70`, `LEFT 34`, `RIGHT 19`, `LEFT_JUMP 18`,
  `RIGHT_JUMP 21`.
- Dataset eval accuracy `74.1%`; recent training accuracy `80.2%`; route delta
  `0.00e+00`; head delta `3.36e-01`.
- Selector eval: `13/30` first-crystal wins, `43.3%` crystals, `56.9%` depth.
- Route/contact scorecard: `2.302` vs B3s `1.821`.
- Selector use: `142` head actions, `28265` policy actions, `582` confidence rejects,
  head action rate `0.5%`, mean accepted confidence `0.807`.
- `compare-artifact`: `HOLD` only because expanded validation was required.

**Validation artifact:**
`.Codex/artifacts/cc_sessions/20260625_150521_b15_contact_head_offline_balanced_conf075_500_val60`

- Result: `22/60` first-crystal wins, `36.7%` crystals, `57.0%` depth.
- Metric audit: success depth `35.7%`; non-success depth `69.4%`, clearing the route
  guardrail and nearly matching B3s validation non-success depth `69.9%`.
- Selector use: `425` head actions, `66317` policy actions, `9317` confidence rejects,
  head action rate `0.6%`, mean accepted confidence `0.833`.
- `compare-artifact --validation`: `PROMOTE`.

**Decision:** promote B15 as the learned contact-head adapter baseline. Next work should
improve its data coverage for the classes the head still confuses. Do not follow it with
an online RL fine-tune until the offline adapter remains stable across validation samples.

**Second-seed check:**

- B3s same-seed control:
  `.Codex/artifacts/cc_sessions/20260625_162938_b3s_selected_seed1_eval30`
- B15 seed-1 artifact:
  `.Codex/artifacts/cc_sessions/20260625_162756_b15_contact_head_offline_balanced_conf075_500_seed1_eval30`
- B3s seed 1 result: `8/30` first-crystal wins, `26.7%` crystals, `54.3%` depth,
  route/contact score `1.359`.
- B15 seed 1 result: `8/30` first-crystal wins, `26.7%` crystals, `52.9%` depth,
  route/contact score `1.362`.
- Same-seed compare: `HOLD`, not promote. B15 improved close-zone jump and
  stuck-after-close, but regressed depth, near-miss distance, and <=1.5 near-miss rate.
- Metric audit: both clear non-success route depth (`64.9%` B3s, `61.4%` B15), so the
  adapter did not collapse routing, but it also did not create a second-seed win lift.
- Decision: do not run seed-1 expanded validation yet. Treat B15 as a seed-0 promoted
  learned-adapter baseline that still needs data/gating work before being cemented as
  robust.

### R11. B16 Jump-Gated Offline Contact Head

**Goal:** test the most direct B15 robustness hypothesis one change at a time. B15 seed 1
tied B3s and accepted mostly jump-variant contact-head overrides. B16 keeps the same
offline/head-only training recipe but requires `0.85` confidence before `JUMP`,
`LEFT_JUMP`, or `RIGHT_JUMP` head actions can override B3s; non-jump actions keep the
base `0.75` confidence threshold.

**Code change:** added `--contact-head-jump-confidence`, recipe
`b16_contact_head_jump_gated`, selector/report payload fields, and tests for
jump-specific confidence rejection. The recipe defaults to seed 1 because it targets the
B15 robustness failure case.

**Seed-1 selected artifact:**
`.Codex/artifacts/cc_sessions/20260625_164745_b16_contact_head_jump_conf085_seed1_eval30`

- Result: `9/30` first-crystal wins, `30.0%` crystals, `54.3%` depth.
- Same-seed B3s control was `8/30`, `26.7%` crystals, `54.3%` depth, score `1.359`.
- Same-seed B15 was `8/30`, `26.7%` crystals, `52.9%` depth, score `1.362`.
- B16 score: `1.526`.
- Selector behavior: only `6` head actions, `29070` policy actions, `1737` confidence
  rejects, mean accepted confidence `0.853`.
- Compare versus B3s and B15: both `HOLD`, with support improvements but selected-only
  evidence.
- Metric audit: raw mean depth `54.3%` misses the old raw guardrail, but non-success
  depth is `63.9%`, so route is not collapsing.

**Seed-0 guardrail artifact:**
`.Codex/artifacts/cc_sessions/20260625_164903_b16_contact_head_jump_conf085_seed0_eval30`

- Result: `12/30` first-crystal wins, `40.0%` crystals, `57.6%` depth.
- Versus B15 seed 0, this is a regression: B15 was `13/30`, `43.3%` crystals, `56.9%`
  depth, score `2.302`; B16 score is `2.193`.
- Versus B3s seed 0, this still improves selected wins (`12/30` vs `10/30`) and score
  (`2.193` vs `1.821`), but it does not beat the current learned-adapter baseline.
- Selector behavior: `23` head actions, `30589` policy actions, `306` confidence rejects,
  mean accepted confidence `0.868`.

**Decision:** keep B16 as a useful gating diagnostic, but do not promote it over B15 and
do not run expanded validation yet. The signal says jump over-acceptance was part of the
seed-1 issue, but a threshold-only fix is too blunt: it helps seed 1 slightly while
removing enough useful seed-0 head actions to lose one win. Next work should improve the
contact-label dataset and calibrate class thresholds from held-out labels, not continue
blind confidence sweeps. B17 completed the dataset-only hard-seed collection pass; the
next work is B18 combine/calibration before any new head run.

### R12. B17 Hard-Seed B10-Gated Label Collection

**Goal:** collect the hard-seed contact labels that B16 suggests are missing. This is a
dataset-only pass, not a trained-policy result.

**Artifact:**
`.Codex/artifacts/cc_sessions/20260625_165300_b17_advantage_gate_correction_collect_seed1`

- Validated: `artifact_validation.json` is `ok`.
- Kept labels: `222` states from `60` seed-1 games, compared with B11's `162` states from
  `60` seed-0 games.
- Candidate states: `1503`, compared with B11's `625`.
- Disagreement rate: `83.8%`, compared with B11's `80.2%`.
- Advantage gate rejection rate: `82.4%`, compared with B11's `67.7%`.
- Label action mix: `JUMP 56`, `LEFT 33`, `LEFT_JUMP 51`, `RIGHT 22`,
  `RIGHT_JUMP 60`.
- B11 label action mix was `JUMP 70`, `LEFT 34`, `LEFT_JUMP 18`, `RIGHT 19`,
  `RIGHT_JUMP 21`.
- Dataset path:
  `.Codex/artifacts/cc_sessions/20260625_165300_b17_advantage_gate_correction_collect_seed1/b17_advantage_gate_correction_collect_seed1/corrections/b17_advantage_gate_correction_collect_seed1_heldout/correction_examples.npz`

**Decision:** B17 is useful and should be kept. It gives the missing hard-seed jump-label
coverage that B16 exposed. Do not train directly from only B17 yet; the next step should
combine B11+B17 and add a calibration/audit layer so thresholds are chosen from held-out
label evidence rather than another blind confidence sweep.

### R13. B18 Combined Contact-Head Calibration

**Goal:** combine B11+B17, split off held-out labels, fit only the detached contact head,
and decide whether the improved data is good enough to justify another selected held-out
selector run.

**Artifacts:**

- Failed plumbing attempt:
  `.Codex/artifacts/cc_sessions/20260625_173600_b18_contact_head_combined_calibration_b11_b17`
  failed artifact validation because the recipe had positive offline training time but
  no live metrics. Fixed by setting `heartbeat_seconds=0` for the offline calibration
  recipe.
- Valid run:
  `.Codex/artifacts/cc_sessions/20260625_173639_b18_contact_head_combined_calibration_b11_b17`

**Result:**

- Combined dataset: `384` labels from B11+B17.
- Split: `288` train / `96` held-out calibration labels.
- Action coverage: `JUMP 126`, `LEFT 67`, `LEFT_JUMP 69`, `RIGHT 41`,
  `RIGHT_JUMP 81`.
- Train action counts: `JUMP 94`, `LEFT 50`, `LEFT_JUMP 52`, `RIGHT 31`,
  `RIGHT_JUMP 61`.
- Calibration action counts: `JUMP 32`, `LEFT 17`, `LEFT_JUMP 17`, `RIGHT 10`,
  `RIGHT_JUMP 20`.
- Route delta: `0.00e+00`; the offline-head path still preserves B3s route weights.
- Train-label accuracy: `75.7%`.
- Held-out calibration accuracy: `58.3%`, below the `70%` gate.
- Per-class calibration accuracy: `LEFT 23.5%`, `RIGHT 60.0%`, `JUMP 65.6%`,
  `LEFT_JUMP 52.9%`, `RIGHT_JUMP 80.0%`.
- Decision: `HOLD`.

**Interpretation:** B17 fixed raw jump-label coverage, but the combined B11+B17 labels
are not internally clean enough for another selected held-out selector run. The weak
classes are `LEFT` and `LEFT_JUMP`, not `RIGHT_JUMP`. Next work should audit label
quality and ambiguity before more training: likely duplicate/near-duplicate states with
conflicting labels, left/right symmetry errors, or B10 option labels that are valid in
rollout but too noisy as one-step supervised targets.

### R14. B19-B21 Stable Contact-Label Filter And Adapter

**Status:** implemented, calibrated, selected-tested, validation-tested, and promoted over
B3s as the current learned contact-head adapter baseline.

**B19 label-quality audit artifact:**
`.Codex/artifacts/cc_sessions/20260625_180033_b19_contact_label_quality_audit_b11_b17`

- Dataset: same B11+B17 `384` labels used by B18.
- Rounded duplicate-state conflicts: `0`, so the failure was not exact duplicated states
  with different labels.
- Semantic ambiguity: `36` local-geometry groups covering `295/384` labels had multiple
  one-step labels.
- Adjacent-frame flips: `76/313` checked pairs (`24.3%`) changed label while targeting the
  same tile.
- Direction mismatch heuristic: `87/242` (`36.0%`) horizontal labels moved opposite the
  target tile x-direction.
- Decision: B10 option labels are often valid short rollouts but too phase-dependent as
  raw one-step labels.

**B20 stable-filter artifact:**
`.Codex/artifacts/cc_sessions/20260625_180829_b20_stable_contact_label_filter_b11_b17`

- Filter: keep semantic groups with at least `0.67` majority label share, keep only rows
  matching the majority label, and drop adjacent label-flip rows.
- Retained `117/384` labels.
- Kept action counts: `JUMP 30`, `LEFT 26`, `LEFT_JUMP 23`, `RIGHT 13`,
  `RIGHT_JUMP 25`.
- Drop reasons: `low_semantic_majority 208`, `adjacent_label_flip 126`,
  `minority_label 22` (non-exclusive counts).

**B20 calibration artifact:**
`.Codex/artifacts/cc_sessions/20260625_180944_b20_stable_contact_head_calibration_b11_b17`

- Split: `88` train / `29` held-out labels.
- Held-out label accuracy: `82.8%`, above the `70%` gate and far above B18 `58.3%`.
- Per-class held-out accuracy: `JUMP 87.5%`, `LEFT 83.3%`, `LEFT_JUMP 83.3%`,
  `RIGHT 66.7%`, `RIGHT_JUMP 83.3%`.
- Train min class count: `10` (`RIGHT`), exactly clearing the floor.
- Route delta: `0.00e+00`.
- Decision: calibration `pass`; justified a selected game eval.

**B21 selected artifact:**
`.Codex/artifacts/cc_sessions/20260625_181012_b21_stable_contact_head_offline_conf075_eval30`

- Result: `13/30` first-crystal wins, `43.3%` crystals, `56.7%` depth.
- Ties B15 selected wins/crystals and slightly improves route/contact score
  (`2.333` vs `2.302`) plus loop-after-close (`26.7%` vs `30.0%`).
- Selector fired much more often than B15: `1898` head actions over 30 games vs B15's
  `142`, mostly `LEFT` and `JUMP`.
- `compare-artifact`: `HOLD` before validation.

**B21 validation artifact:**
`.Codex/artifacts/cc_sessions/20260625_181129_b21_stable_contact_head_offline_conf075_val60`

- Result: `23/60` first-crystal wins, `38.3%` crystals, `56.9%` raw depth.
- Beats B15 validation wins (`22/60`), crystal rate (`36.7%`), and route/contact score
  (`2.036` vs `1.920`).
- Raw depth missed the `57.0%` guardrail by about `0.001`, but metric audit showed
  success depth `34.5%` and non-success depth `70.8%`, clearing B3s's non-success route
  guardrail.
- Fixed promotion snapshot extraction so `compare-artifact --validation` reads
  artifact-level per-game rows for success/non-success depth. Corrected result:
  `PROMOTE`.

**Decision:** promote B21 as the current learned B3s+NN contact-head adapter baseline.
Keep B10 as the best overall eval-time controller because its validation outcome remains
stronger (`30/60`).

**B21 seed-1 robustness artifacts:**

- Selected:
  `.Codex/artifacts/cc_sessions/20260625_182515_b21_stable_contact_head_offline_conf075_seed1_eval30`
- Validation:
  `.Codex/artifacts/cc_sessions/20260625_182705_b21_stable_contact_head_offline_conf075_seed1_val60`
- Matched B3s seed-1 validation control:
  `.Codex/artifacts/cc_sessions/20260625_182902_b3s_selected_seed1_val60_control`
- Selected result: B21 improved same-seed selected wins from B3s/B15 `8/30` to `10/30`.
- Matched validation result: B21 improved wins from B3s `16/60` to `18/60` and score from
  `1.413` to `1.555`.
- Risk: raw depth regressed from `55.5%` to `53.0%`; non-success depth regressed from
  `64.4%` to `62.1%`; mean minimum target distance worsened slightly.
- Selector evidence: seed-1 B21 accepted `2767` head actions over 60 games, dominated by
  `LEFT_JUMP 2265`. This is a much stronger jump-variant skew than seed 0 and likely
  explains the route-depth cost.

**Decision update:** B21 has a modest second-seed win lift, not just seed-0 noise. Keep it
as the learned adapter baseline, but treat route-depth cost as the next blocker. The next
one-change test was broad jump confidence on top of B21 stable labels.

**B22/B23 broad jump-threshold probes:**

- B22 `jump_conf=0.90`:
  `.Codex/artifacts/cc_sessions/20260625_183234_b22_stable_contact_head_jump_conf090_seed1_eval30`
- B23 `jump_conf=0.85`:
  `.Codex/artifacts/cc_sessions/20260625_183355_b23_stable_contact_head_jump_conf085_seed1_eval30`
- Both selected checks landed at `9/30`, below B21 seed-1 selected `10/30`.
- B22 reduced loop-after-close (`23.3%` vs B21 `26.7%`) and improved non-success depth
  (`61.9%` vs `60.7%`), but it also reduced close-zone jump rate and lost a win.
- B23 was nearly identical to B22 and also regressed by the promotion gate.

**Decision update:** broad jump-threshold gating is not the lever. The over-firing problem
is class-specific (`LEFT_JUMP` on seed 1), so the next useful mechanism should allow
per-action thresholds or class-specific calibration rather than one threshold for all jump
variants.

**B24 per-action `LEFT_JUMP` threshold probe:**

- Seed-1 selected:
  `.Codex/artifacts/cc_sessions/20260625_184641_b24_stable_contact_head_left_jump_conf090_seed1_eval30`
- Seed-1 validation:
  `.Codex/artifacts/cc_sessions/20260625_184808_b24_stable_contact_head_left_jump_conf090_seed1_val60`
- Seed-0 guardrail:
  `.Codex/artifacts/cc_sessions/20260625_185016_b24_stable_contact_head_left_jump_conf090_seed0_eval30`
- Mechanism added: `--contact-head-action-thresholds ACTION:confidence`, with B24 using
  `LEFT_JUMP:0.90` while keeping the base confidence threshold at `0.75` and leaving
  `JUMP`/`RIGHT_JUMP` ungated beyond the base threshold.
- Seed-1 selected result tied B21 selected wins (`10/30`) and improved route/contact
  score (`1.719` vs `1.693`), near-miss `<=1.5` tiles (`40.0%` vs `33.3%`), and
  loop-after-close (`23.3%` vs `26.7%`), but raw depth slipped (`50.2%` vs `50.9%`).
- Seed-1 validation result improved B21 seed-1 validation from `18/60` to `19/60` and
  score from `1.555` to `1.643`. It also reduced accepted `LEFT_JUMP` actions from
  B21's `2265` to `312`, confirming the gate hit the intended failure mode.
- The remaining problem: non-success route depth stayed weak (`61.7%`) and still trails
  both B21 (`62.1%`) and the matched B3s seed-1 control (`64.4%`). This means threshold
  tuning reduced bad selector behavior, but did not restore the route policy's broader
  depth behavior.
- Seed-0 guardrail tied B21 seed-0 selected wins/depth/score exactly (`13/30`, `56.7%`,
  score `2.333`) and removed accepted `LEFT_JUMP` actions without breaking the strong
  seed-0 outcome.

**Decision update:** keep the per-action threshold mechanism and B24 recipe, but do not
promote B24 as the learned adapter baseline. B21 remains the baseline; the next useful
work is policy-visited label aggregation from B21/B24 failures, then B20-style stability
filtering.

**B25-B30 policy-visited label aggregation probe:**

- B25 collection:
  `.Codex/artifacts/cc_sessions/20260625_190140_b25_b24_policy_visited_contact_collect_seed1`
- B26 audit:
  `.Codex/artifacts/cc_sessions/20260625_190457_b26_b20_b25_policy_visited_contact_audit`
- B26 filter:
  `.Codex/artifacts/cc_sessions/20260625_190502_b26_b20_b25_policy_visited_stable_filter`
- B27 calibration:
  `.Codex/artifacts/cc_sessions/20260625_190527_b27_b20_b25_stable_contact_head_calibration`
- B28 selected eval:
  `.Codex/artifacts/cc_sessions/20260625_190558_b28_b20_b25_stable_contact_head_left_jump_conf090_seed1_eval30`
- B29 selected eval:
  `.Codex/artifacts/cc_sessions/20260625_190718_b29_b20_b25_stable_contact_head_jump_leftjump_conf090_seed1_eval30`
- B30 selected eval:
  `.Codex/artifacts/cc_sessions/20260625_190858_b30_b20_b25_stable_contact_head_class_calibrated_seed1_eval30`
- Mechanism added: `collect-contact-head-corrections` mode and
  `b25_contact_head_policy_collect` recipe. This fits the frozen-route contact head from
  a supplied contact-label dataset, rolls out that selector, then collects B10
  advantage-gated labels from the selector's own visited states. Rows now record
  `policy_action_source` and `policy_action_meta`, so future audits can separate base
  policy actions from contact-head overrides.
- B25 collected `225` kept labels from the B24-style selector rollout on seed 1. It covered
  all five classes (`JUMP 71`, `LEFT 42`, `LEFT_JUMP 30`, `RIGHT 32`, `RIGHT_JUMP 50`) and
  reproduced the B24 seed-1 outcome distribution (`19/60` first-crystal successes during
  collection).
- B26 aggregated B20+B25 into `342` labels. The audit still showed severe phase noise:
  `30` semantic ambiguity groups, `200` ambiguous examples, `62/250` adjacent label flips
  (`24.8%`), and `88/220` horizontal direction mismatches (`40.0%`). The B20-style filter
  retained `184` stable labels with all five classes represented.
- B27 passed calibration and improved held-out label accuracy from B20's `82.8%` to
  `85.1%`, but mean confidence fell from `0.826` to `0.730`. This was an early warning:
  useful lateral labels such as `LEFT` had low mean confidence (`0.616`).
- B28 trained directly on B26 and regressed to `9/30`. It over-fired `JUMP` (`1477`
  accepted head actions) and almost stopped accepting `LEFT` (`2`), despite the better
  offline calibration.
- B29 raised both `JUMP` and `LEFT_JUMP` thresholds to `0.90`; this fixed `JUMP` over-fire
  (`1477` -> `9`) but still ended at `9/30`, so the issue was not just jump spam.
- B30 lowered the base threshold to `0.60` to recover low-confidence `LEFT` while keeping
  risky classes gated (`JUMP/LEFT_JUMP 0.90`, `RIGHT/RIGHT_JUMP 0.75`). It accepted
  `523` `LEFT` actions but still ended at `9/30`, confirming the B26 aggregate changed the
  contact-head behavior in a way that confidence thresholds alone do not fix.

**Decision update:** keep B25's collection mechanism and B26 data for diagnostics, but do
not train the production contact head directly on B26. B21 remains the learned baseline,
with B24 available as a non-promoted selector guardrail. The next useful work is not "more
policy-visited labels"; it is source-aware/phase-aware use of those labels.

### Paths To Stop Spending Time On

- More correction-action weight sweeps.
- More global policy-anchor weight sweeps; keep the B13 route mask available, but do not
  sweep it as the only change.
- The B14 online `contact-head-finetune` recipe as-is; keep the head infrastructure, but
  do not repeat it without head-only training, class balance, and confidence gating.
- More final-contact threshold/commit variants without a new mechanism or a gate-accepted
  label/training plan.
- More broad beam/scripted demo coverage as default training data.
- More terminal reward tuning.
- Direct B20+B25 aggregate head training without source/phase controls.

## Detailed History Archives

The long B-series run notes and findings log were split into smaller archive files so
future sessions can read the relevant section without reopening a 3,000+ line tracker.
No findings were deleted.

- `docs/cc_nn_experiment_tracker/b_series_route_mastery.md` - B1 through B3h route-mastery plan and results.
- `docs/cc_nn_experiment_tracker/findings_2026_06_23_part1.md` - A-series findings, metrics review, and rerun decision.
- `docs/cc_nn_experiment_tracker/findings_2026_06_24_part1.md` - B3l through B3q follow-up runs and evaluation/demo-selection work.
- `docs/cc_nn_experiment_tracker/findings_2026_06_24_part2.md` - B3r through B3u filtered demos, conservative demo-Q, and close-zone label findings.

## Cleanup Note

There are now two baselines:

- **Pure-NN training baseline:** B3s conservative demo-Q. Use this when comparing learned
  checkpoints with no eval-time option.
- **Best learned contact adapter:** B21 stable-label offline/head-only contact selector. Use this when
  comparing B3s plus a learned NN contact head, with no oracle/planner at eval time.
- **Best eval-time outcome baseline:** B10 advantage-gated final-contact controller. Use
  this when asking what the current system can achieve with the controller layered on top
  of B3s.

B3t and B3u are archived as non-promoted close-zone variants because they did not improve
the expanded validation outcome enough to replace B3s. B3v is also not promoted: it is a
useful split-policy proof that close-zone control can increase first-crystal wins, but its
validation non-success depth remained too shallow after broad option takeover.

## 2026-06-24 B3s Policy-Visited Correction Pass

**Status:** completed; not promoted over B3s.

**Purpose:** test the top-ranked DAgger-style correction idea one step at a time. The
hypothesis was that B3s fails because scripted successful-demo states do not match the
states the learned policy actually visits. The experiment collected labels from
policy-visited loop/stale/close-zone states and fine-tuned the B3s selected checkpoint
with a low-weight correction action-margin loss.

**Baseline:** B3s conservative demo-Q remains the comparison point:
`10/30` selected first-crystal wins, `33.3%` crystals, `60.5%` depth, and `19/60`
expanded validation.

**Correction dataset artifact:**
`.Codex/artifacts/cc_sessions/20260624_215607_b3s_correction_collect_b3s_ep300_30g`

- Source checkpoint:
  `.codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300/tutorial_demo_conservative/models/crystal_caves/tutorial_demo_conservative_selected_ep300.pth`
- Kept examples: `1024` states from `20` games; artifact validation `ok: true`.
- Policy/label disagreement: `82.3%`.
- Triggers: `loop=844`, `stale=448`, `close_zone=182`.
- Label sources: `route_recovery=842`, `close_zone_oracle=182`.
- Label actions: `RIGHT_JUMP=251`, `LEFT_JUMP=216`, `LEFT=214`, `JUMP=185`,
  `RIGHT=97`, `IDLE=61`.

**Fine-tune artifact:**
`.Codex/artifacts/cc_sessions/20260624_215710_b3s_correction_finetune_b3s_ep300_1024_300`

- Training budget: `300` episodes, `8` envs, seed `0`, same B3s checkpoint, correction
  weight `0.020`, margin `0.60`, batch `64`.
- Correction loss was active: `1024` states, `correction_action_samples_100=100`,
  avg correction loss `0.2821`, correction accuracy `75.1%`.
- Final held-out eval: `3/16` first-crystal wins (`18.8%`), `18.8%` crystals,
  `29.5%` depth, end reasons `{'timeout': 10, 'stalled': 3, 'first_crystal_goal': 3}`.
- Near-miss rollup: `43.8% <=3 tiles`, `25.0% <=1.5 tiles`,
  mean min target distance `5.42`, close-zone jump `9.2%`, stuck-after-close `6.2%`,
  loop-after-close `25.0%`.

**Promotion gate:** `compare-artifact` returned `HOLD` because the final eval sample
was only `3/16` versus the frozen B3s `10/30` selected sample. The support metrics were
mixed: close-zone jump, stuck-after-close, and loop-after-close improved, but wins,
crystals, depth, near-miss rates, and mean target distance all regressed materially.

**Decision:** do not promote this correction-fine-tune recipe. Do not repeat the same
`1024`-state disagreement dataset at weight `0.020` as the next default path. If the
correction idea is revisited, change one variable deliberately: lower the correction
weight, stratify/downsample repeated loop states, and make the mode save/evaluate a
selected checkpoint with the normal `30`-game selected gate. The next ranked approach at
that point was the final-contact option family; B3v through B3y record those outcomes
below.

## 2026-06-24 B3v Final-Contact Option Evaluation

**Status:** completed; not promoted over B3s. It still fails under the outcome-conditioned
gate because broad option takeover leaves failed validation episodes too shallow.

**Purpose:** test the next-ranked split-policy idea without training a new network. The
B3s selected checkpoint still chooses route actions, but inside the close-zone threshold
the runner switches to a local final-contact option that simulates short action macros
from copied game states. The option commits each chosen macro for `8` actions before
replanning, which keeps eval runtime tractable and matches an options-style controller
better than replanning every frame.

**Code path added:** `eval-final-contact-option` status-session mode.

- Restores a selected-weight checkpoint with the same config restore path as
  `eval-checkpoint`.
- Writes a normal `final_eval` payload plus `final_contact_option` action counts,
  trigger rates, and per-level rows.
- Reuses near-miss diagnostics with the same option action selector.
- Adds report lines and artifact validation coverage for the option evidence.

**Smoke artifact:**
`.codex/artifacts/cc_sessions/20260624_222824_b3s_final_contact_option_smoke_v3`

- Result: `1/4` wins, `25.0%` crystals, `60.7%` depth.
- Artifact validation: `ok`.

**Selected comparison artifact:**
`.codex/artifacts/cc_sessions/20260624_222911_b3s_final_contact_option_eval30`

- Result: `13/30` first-crystal wins (`43.3%`) versus B3s `10/30`.
- Crystals/depth: `43.3%` / `50.5%`.
- End reasons: `{'first_crystal_goal': 13, 'stalled': 13, 'timeout': 4}`.
- Option activity: `13,015` option-controlled actions, `35.9%` of eval steps,
  `8.8%` simulated target-completion rate.
- Near-miss support: `46.7% <=1.5 tiles`, mean min target distance `3.49`,
  stuck-after-close `13.3%`, loop-after-close `36.7%`.

**Expanded validation artifact:**
`.codex/artifacts/cc_sessions/20260624_223318_b3s_final_contact_option_val60`

- Result: `24/60` first-crystal wins (`40.0%`) versus B3s validation `19/60`
  (`31.7%`).
- Crystals/depth: `40.0%` / `53.6%`.
- End reasons: `{'first_crystal_goal': 24, 'stalled': 28, 'timeout': 8}`.
- Option activity: `22,523` option-controlled actions, `29.8%` of eval steps,
  `9.5%` simulated target-completion rate.

**Promotion gate:** `compare-artifact <selected> --validation <validation>` returned
`REGRESS`.

- Improvements: selected wins, validation wins, crystal fraction, `<=1.5` near-miss
  rate, mean min target distance, close-zone jump rate, and stuck-after-close rate.
- Regressions: selected depth and loop-after-close rate.
- Blocking reason: expanded validation depth was `0.536`, below the required `0.570`
  guardrail (`B3s validation depth 0.600 - 0.030`). Its validation non-success depth was
  only about `60.9%`, below the B3s outcome-conditioned route guardrail.

**Decision:** do not replace B3s with this eval-only option. Keep the mode because it is
a strong diagnostic and a useful architecture probe: close-zone control can increase
first-crystal wins, but this version takes over too much of the route (`~30-36%` of
steps) and hurts the depth profile.

## 2026-06-24 B3w/B3x Final-Contact Follow-Ups

**Status:** completed; neither promoted.

These two probes changed one final-contact option variable at a time on top of B3s.
They used the same B3s selected checkpoint as B3v and the same `30`-game selected eval
shape. The CLI wrapper now passes `--label` through to the inner run summary so follow-up
artifacts are self-describing.

**B3w: narrower trigger distance (`1.5` tiles, commit `8`)**

- Artifact:
  `.Codex/artifacts/cc_sessions/20260624_230035_b3s_final_contact_option_d15_commit8_eval30`
- Result: `8/30` first-crystal wins (`26.7%`), `26.7%` crystals, `61.9%` depth.
- End reasons: `{'first_crystal_goal': 8, 'stalled': 17, 'timeout': 5}`.
- Option activity: `1,613` option actions, only `4.3%` of eval steps, `12.6%`
  simulated target-completion rate.
- Near-miss support: `60.0% <=3 tiles`, `40.0% <=1.5 tiles`, mean min target distance
  `3.54`, stuck-after-close `26.7%`, loop-after-close `46.7%`.
- Gate: `REGRESS`, because selected wins trail B3s `10/30`.

**Interpretation:** the overactive-option hypothesis was partly right for depth, but
`1.5` tiles under-fires. It gives the NN more route control and clears the depth
guardrail, but loses the contact benefit that made B3v interesting.

**B3x: shorter committed macro (`3.0` tiles, commit `4`)**

- Artifact:
  `.Codex/artifacts/cc_sessions/20260624_230324_b3s_final_contact_option_d30_commit4_eval30`
- Result: `11/30` first-crystal wins (`36.7%`), `36.7%` crystals, `50.9%` depth.
- End reasons: `{'first_crystal_goal': 11, 'stalled': 15, 'timeout': 4}`.
- Option activity: `11,049` option actions, `29.6%` of eval steps, `19.9%` simulated
  target-completion rate.
- Near-miss support: `60.0% <=3 tiles`, `43.3% <=1.5 tiles`, mean min target distance
  `3.56`, stuck-after-close `20.0%`, loop-after-close `46.7%`.
- Gate: `HOLD`, because selected wins beat B3s by one but need expanded validation
  before promotion.

**Interpretation:** `commit=4` is still too overactive to fix the depth issue, and it is
worse than B3v on the primary selected signal (`11/30` versus `13/30`). Do not spend a
60-game validation budget on B3x unless a later cancellation variant also improves the
depth profile.

**B3y: cancel committed macros outside the close zone (`3.0` tiles, commit `8`)**

- Artifact:
  `.Codex/artifacts/cc_sessions/20260624_231322_b3s_final_contact_option_d30_commit8_cancel_eval30`
- Result: `13/30` first-crystal wins (`43.3%`), `43.3%` crystals, `50.5%` depth.
- End reasons: `{'first_crystal_goal': 13, 'stalled': 13, 'timeout': 4}`.
- Option activity: `12,953` option actions, `35.8%` of eval steps, `8.8%` simulated
  target-completion rate.
- Cancellation activity: only `61` queued actions cancelled across `55` plans.
- Near-miss support: `60.0% <=3 tiles`, `46.7% <=1.5 tiles`, mean min target distance
  `3.49`, stuck-after-close `13.3%`, loop-after-close `36.7%`.
- Gate: `HOLD`, because selected wins are promising versus B3s but expanded validation
  would still be blocked by the same low-depth profile unless validation surprises.

**Interpretation:** B3y ties B3v on the headline selected metrics and preserves the
better stuck/near-miss profile, but cancellation barely triggers and does not address
the depth failure. This makes the final-contact option a good diagnostic and possible
future hybrid-controller component, not the current route to a promoted baseline.

**Decision for this option family:** do not run more small threshold/commit variants
without a stronger hypothesis. The repeated pattern is now clear: close-zone control
raises first-crystal contact, while broad option takeover hurts route depth. The next
default work should move back toward NN/curriculum changes that teach contact behavior
without replacing ~30-36% of the policy's actions at evaluation time.

## 2026-06-25 B4 Contact-Only Correction Dataset

**Status:** collection passed; both low-weight fine-tunes failed early-stop gates.

**Purpose:** move from eval-time option takeover to teaching the learned policy the
contact behavior. The previous correction fine-tune regressed badly because the dataset
was mostly loop/stale recovery states (`loop=844`, `stale=448`) with only `182`
close-zone examples. The final-contact option family then proved close-zone control can
raise first-crystal contact, but broad option takeover hurts depth.

**Plan:** collect a policy-visited correction dataset from the B3s selected checkpoint
with stale and loop triggers disabled:

- `--correction-stale-steps 999999`
- `--correction-loop-tile-visits 999999`
- default close-zone threshold (`3.0` tiles)
- disagreement-only examples, so the loss targets states where the policy differs from
  the close-zone oracle

**Promotion rule for this step:** this is a dataset-quality gate, not a promoted model.
Proceed to a low-weight fine-tune only if the artifact has enough close-zone examples,
label mix is not dominated by one repeated action, and disagreement rate shows the NN is
actually missing contact choices.

**Collection result:**
`.Codex/artifacts/cc_sessions/20260625_030239_b4_contact_only_correction_collect`

- `275` kept correction states from `30` held-out games.
- Trigger mix was clean: `275/275` kept examples came from `close_zone`; stale and loop
  triggers were disabled.
- Candidate states: `337`; agreement states: `62`; disagreement rate: `81.6%`.
- Label source: `close_zone_oracle` for all kept examples.
- Label action mix: `IDLE=81`, `JUMP=74`, `RIGHT_JUMP=40`, `LEFT_JUMP=35`, `LEFT=24`,
  `RIGHT=21`.
- Policy actions on the same states were mostly non-jump route actions:
  `RIGHT=161`, `LEFT=94`, `IDLE=20`.
- Collection endpoint mix: `10` first-crystal goals, `11` stalls, `9` timeouts.
- Dataset path:
  `.Codex/artifacts/cc_sessions/20260625_030239_b4_contact_only_correction_collect/b4_contact_only_correction_collect/corrections/b4_contact_only_correction_collect_heldout/correction_examples.npz`

**Collection decision:** this is a real improvement over the old broad correction dataset quality.
It isolates the exact final-contact disagreement, has enough examples for a small
auxiliary loss, and is not dominated by one label. Proceed to one low-weight fine-tune
from B3s at `--correction-action-weight 0.010`. If that overpowers route behavior again,
retry only once at `0.005`; do not return to the old loop-heavy `1024`-state recipe.

**Fine-tune result at weight `0.010`:**
`.Codex/artifacts/cc_sessions/20260625_030453_b4_contact_only_correction_finetune_w010_300`

- Run was intentionally interrupted at episode `155/300` after the episode-150 held-out
  eval regressed.
- Correction objective was active and learnable: `275` correction transitions,
  `correction_action_samples_100=100`, avg correction loss `0.2233`, avg correction
  accuracy `82.1%`.
- Held-out eval checkpoints:
  - ep50: `2/8` wins, `25.0%` crystals, `17.9%` depth.
  - ep100: `2/8` wins, `25.0%` crystals, `30.4%` depth.
  - ep150: `1/8` wins, `12.5%` crystals, `20.5%` depth.
- The final live metrics had rolling training progress near `0.451`, but held-out depth
  was far below B3s's route profile (`60.5%` selected depth, `60.0%` validation depth).

**Fine-tune result at weight `0.005`:**
`.Codex/artifacts/cc_sessions/20260625_031606_b4_contact_only_correction_finetune_w005_300`

- Run was intentionally interrupted at episode `102/300` after the episode-100 held-out
  eval regressed further than `0.010`.
- Correction objective was still active: `275` correction transitions,
  `correction_action_samples_100=100`, avg correction loss `0.2317`, avg correction
  accuracy `81.3%`.
- Held-out eval checkpoints:
  - ep50: `1/8` wins, `12.5%` crystals, `9.8%` depth.
  - ep100: `0/8` wins, `0.0%` crystals, `16.1%` depth.

**Fine-tune decision:** do not promote either correction fine-tune. This closes the
current correction-loss lane. The narrowed finding is useful: the contact-only dataset is
cleaner, and the auxiliary action-margin loss is definitely being sampled and fit, but
blending supervised close-zone labels into the active DQN update still damages the route
policy before it improves held-out contact outcomes. The next default work should move
to staged curriculum or network/curriculum architecture changes that preserve route
behavior while teaching contact, not more small correction-weight sweeps.

## 2026-06-25 B5 Anchored Contact Correction

**Status:** plumbing and smoke validated; both anchor calibrations failed early-stop gates.

**Purpose:** test the smallest concrete mechanism that addresses the B4 failure mode.
B4 proved the contact labels can be fit, but route behavior collapses. B5 keeps the B4
contact-only correction dataset and adds a frozen-teacher policy anchor: a KL loss that
keeps the current policy close to the restored B3s checkpoint on replay states while the
contact correction loss trains close-zone states.

**Recipe:** `b5_anchored_contact_correction`

- Starts from the B3s selected checkpoint.
- Uses the B4 contact-only dataset:
  `.Codex/artifacts/cc_sessions/20260625_030239_b4_contact_only_correction_collect/b4_contact_only_correction_collect/corrections/b4_contact_only_correction_collect_heldout/correction_examples.npz`
- Correction weight: `0.005`, margin `0.6`, batch `64`.
- Policy anchor weight: `0.020`, temperature `1.0`.
- Early-stop rule: stop if held-out depth stays shallow at ep100/150; promote only if it
  beats B3s first-crystal wins without losing the route-depth guardrail.

**Smoke artifact:**
`.Codex/artifacts/cc_sessions/20260625_070236_b5_anchored_contact_smoke_20`

- Artifact validation: `ok`.
- Final 4-game smoke eval: `0/4` wins, `0.0%` crystals, `37.5%` depth. This is not a
  decision sample; it only verifies the mechanism runs.
- Correction metrics were active: `275` states, `correction_action_samples_100=100`,
  avg correction loss `0.2712`, correction accuracy `78.6%`.
- Anchor metrics were active: `policy_anchor_samples_100=100`, avg anchor loss
  `0.0924`, teacher-action match `40.4%`.

**Next decision run:** run the full B5 recipe on the same B3s checkpoint and B4 dataset.
Do not promote from the 20-episode smoke.

**Comparison artifact at anchor weight `0.020`:**
`.Codex/artifacts/cc_sessions/20260625_070641_b5_anchored_contact_correction_w005_a002_300`

- Run was intentionally interrupted at episode `101/300` after the episode-100 held-out
  eval stayed shallow.
- Correction objective was active: `275` states, `correction_action_samples_100=100`,
  avg correction loss `0.2221`, correction accuracy `81.6%`.
- Policy anchor was active but weak: `policy_anchor_samples_100=100`, avg anchor loss
  `0.0834`, teacher-action match only `33.4%`.
- Held-out eval checkpoints:
  - ep50: `1/8` wins, `12.5%` crystals, `8.9%` depth.
  - ep100: `1/8` wins, `12.5%` crystals, `25.0%` depth.

**Decision:** do not promote B5 at anchor weight `0.020`. This is a useful negative:
anchoring is wired correctly, but the route-preservation pressure was not strong enough
to keep the policy close to B3s. One stronger-anchor calibration is justified by the
measured low teacher-action match; if that also fails, stop correction/anchor methods
and move to a staged curriculum method instead.

**Comparison artifact at anchor weight `0.100`:**
`.Codex/artifacts/cc_sessions/20260625_071613_b5_anchored_contact_correction_w005_a010_300`

- Run was intentionally interrupted at episode `108/300` after the episode-100 held-out
  eval still failed the depth gate.
- Correction objective remained active: `275` states, `correction_action_samples_100=100`,
  avg correction loss `0.2298`, correction accuracy `81.6%`.
- Stronger anchor increased teacher matching versus `0.020`: `policy_anchor_samples_100=100`,
  avg anchor loss `0.0455`, teacher-action match `54.3%`.
- Held-out eval checkpoints:
  - ep50: `2/8` wins, `25.0%` crystals, `34.8%` depth.
  - ep100: `2/8` wins, `25.0%` crystals, `22.3%` depth.

**B5 decision:** do not promote anchored correction. The stronger anchor made the
teacher-preservation metric move in the intended direction, but route-depth outcomes did
not recover. This suggests the problem is not just catastrophic drift from correction
labels. Stop correction/anchor weight sweeps here and move to a staged curriculum or
environment-scheduling method where contact practice happens in isolated training levels
without blending supervised contact labels into every replay update.

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
