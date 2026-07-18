# Crystal Caves NN Run-Info Results

Execution evidence for `CC_NN_RUN_INFO_TASKS.md` rev 3. Runs were performed on
the PR #39 branch (`claude/nn-performance-grading-2840b4`) on 2026-07-07
America/Los_Angeles, with artifacts written under the main checkout's
`.Codex/artifacts/cc_sessions/` folder.

## Task 0.1 - B3s drift check post-#37

- artifact/out dir: `/Users/justin/Documents/Github/NN-Game1/.Codex/artifacts/cc_sessions/20260707_172317_t01_b3s_drift_check_post37_eval30`
- validation/preflight: ok (`artifact_validation.json` ok, no warnings)
- command: `/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py eval-checkpoint --checkpoint /Users/justin/Documents/Github/NN-Game1/.Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300/tutorial_demo_conservative/models/crystal_caves/tutorial_demo_conservative_selected_ep300.pth --eval-games 30 --seed 0 --label t01_b3s_drift_check_post37_eval30 --out-dir /Users/justin/Documents/Github/NN-Game1/.Codex/artifacts/cc_sessions`
- wall time: 0:01
- headline: wins 10/30 | crystals 33.3% | depth/progress 64.8% | exit-unlock 0.0%
- support: near-miss <=3 56.7% | <=1.5 36.7% | loop-after-close 30.0% | stuck-after-close 23.3%
- verdict vs baseline: HOLD - frozen B3s reproduced exactly at 10/30, so no baseline drift beyond the task's +/-3 tolerance.
- end reasons: `{"first_crystal_goal": 10, "stalled": 20}`
- anomalies: `eval-checkpoint` restored the checkpoint's first-crystal objective as expected; no full-level objective override was applied.

## Task 0.2 - Full-level completion eval of B3s/B21

- artifact/out dir: none; skipped per tooling/artifact gap
- validation/preflight: failed
- command: no run command was executed
- wall time: 0:00
- headline: wins n/a | crystals n/a | depth/progress n/a | exit-unlock n/a
- support: near-miss <=3 n/a | <=1.5 n/a | loop-after-close n/a | stuck-after-close n/a
- verdict vs baseline: HOLD - no comparable full-objective evidence was produced.
- end reasons: n/a
- anomalies: `eval-checkpoint` reconstructs objective mode from the selected checkpoint config and exposes no `--full-objective` / first-crystal disable override. The B3s snapshot records `first_crystal_goal: true`. The B21 artifact family has valid summaries, but no `.pth` checkpoint; each B21 summary points back to the B3s checkpoint and also records `first_crystal_goal: true`.

## Task 2.1 - B3s conservative demo-Q with geo-compass

- artifact/out dir: `/Users/justin/Documents/Github/NN-Game1/.Codex/artifacts/cc_sessions/20260707_172512_t21_b3s_geo_compass_300_seed0`
- validation/preflight: ok (`artifact_validation.json` ok, no warnings)
- command: `/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_conservative_demo_q --geo-compass --label t21_b3s_geo_compass_300_seed0 --out-dir /Users/justin/Documents/Github/NN-Game1/.Codex/artifacts/cc_sessions`
- wall time: 0:12
- headline: selected wins 10/30 | crystals 33.3% | depth/progress 47.4% | exit-unlock 0.0%
- support: near-miss <=3 53.3% | <=1.5 36.7% | loop-after-close 26.7% | stuck-after-close 20.0%
- verdict vs baseline: REGRESS - `compare-artifact` tied B3s wins at 10/30 but regressed support metrics, especially depth (47.4% vs 60.5%) and route/contact score (1.640 vs 1.821).
- end reasons: `{"first_crystal_goal": 10, "stalled": 20}`
- anomalies: selected checkpoint was episode 250. Final episode 300 source eval was only 3/16 (18.8%) with 37.5% depth. Selected trace was 0/4 wins with all four traced failures stalled/no-crystal/tile-loop.

## Blockers / tooling gaps

- Full-objective checkpoint eval is not currently exposed by `eval-checkpoint`; it always restores `first_crystal_goal` from the saved checkpoint config. Add an explicit, recorded objective-mode override before relying on Task 0.2.
- The named B21 artifact directory does not contain a `.pth` selected checkpoint. Its summaries are valid B21/contact-head evaluations layered on the B3s checkpoint, not a restorable B21 weight snapshot.
- RUN-26 remains blocked on owner-recorded human demos as described in the task brief; no agent run was started for Task 2.2.
