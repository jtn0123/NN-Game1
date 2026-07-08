# Crystal Caves NN — Run-Info Task Brief (2026-07-07)

**Who this is for:** an execution agent tasked with *gathering run evidence only*.
Do not change model code, rewards, or hyperparameters beyond the explicit CLI
flags listed per task. Do not promote/demote baselines. Your job is to run the
commands, wait, verify artifacts, and report numbers.

**Why:** the 2026-07-07 performance grade
(`.Codex/nn-performance-grade-2026-07-07.md`) found that the two most promising
levers merged on 2026-06-26 — the geodesic PBRS potential (PR #35) and
reverse-curriculum mid-solution starts (PR #36) — are off by default and have
never been exercised by a single training run, and that the learn-time reward
clamp (`REWARD_CLIP=5`) flattens death (−12) / timeout (−8) / stall (−6) into an
indistinguishable −5. The last training artifact on disk is dated 2026-06-25.
Everything below exists to close that evidence gap. The PR that ships this brief
also adds the CLI flags the tasks use: `--geodesic-potential`,
`--geodesic-potential-weight`, `--show-locked-exit`, `--reverse-curriculum-p`,
and `--reward-clip` on `experiments/cc_status_session.py` (they apply to the
`tutorial-demo-*` modes, including the B3s recipe).

**Ground rules (from the tracker — do not re-derive):**

- Python: `/Users/justin/.pyenv/versions/3.12.11/bin/python`.
- All runs go through `experiments/cc_status_session.py`; artifacts land under
  `.Codex/artifacts/cc_sessions/<timestamp>_<label>/`.
- Trust only runs whose `artifact_validation.json` says `ok: true`.
- The frozen pure-NN baseline is **B3s**: `10/30` selected first-crystal wins
  seed 0, `19/60` expanded validation, `60.5%` selected depth. Its selected
  checkpoint:
  `.Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300/tutorial_demo_conservative/models/crystal_caves/tutorial_demo_conservative_selected_ep300.pth`
- Judge with `compare-artifact` (and `--validation` when a val60 exists), plus
  the near-miss metrics. Never judge on training score alone.
- A 300-episode recipe run takes roughly 2–3 hours on the M4 (CPU). Eval-only
  tasks take minutes. Run tasks strictly in the order below; later tasks depend
  on earlier results.
- While a training run is live, report the artifact's `live_metrics.json` path
  so progress can be watched.
- After each task, append the result block (defined at the bottom) to
  `CC_NN_RUN_INFO_RESULTS.md` (create it on first write). Include artifact paths
  verbatim.

---

## Phase 0 — cheap eval-only sanity (no training; ~30 min total)

### Task 0.1 — Baseline drift check: does frozen B3s still reproduce on current main?

PRs #34/#35/#36 changed eval selection, shaping plumbing, and reset paths after
the B3s numbers were frozen. If the same checkpoint no longer reproduces
~`10/30` on the same seed-0 held-out eval, every future comparison is invalid.

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py eval-checkpoint \
  --checkpoint .Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300/tutorial_demo_conservative/models/crystal_caves/tutorial_demo_conservative_selected_ep300.pth \
  --eval-games 30 --seed 0 --label t01_b3s_drift_check_main_eval30
```

(If the exact flag names differ, run the mode with `--help` first; do not
guess.)

**Report:** wins/30, crystal %, depth %, end-reason counts, and whether the
numbers are within noise of `10/30 / 33.3% / 60.5%`. If wins differ by more than
±3, STOP and report — do not run Phase 1 on a drifted surface.

### Task 0.2 — Full-level completion eval (not first-crystal)

Every promoted number in the tracker is a *first-crystal* metric. We need the
actual full-game number (all crystals + exit) for B3s and B21, same seed-0
held-out surface, 30 games each:

- B3s selected checkpoint (path above).
- B21 adapter checkpoint (under
  `.Codex/artifacts/cc_sessions/20260625_181012_b21_stable_contact_head_offline_conf075_eval30/`
  — locate the `.pth` inside; report the exact path you used).

The first-crystal early terminal is an experiment-config setting
(`CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL`); `eval-checkpoint` restores the config the
checkpoint was trained with, so check the run's `summary.json` config snapshot
to confirm whether the eval ended at first crystal or ran the full objective.
If there is no supported way to run these checkpoints under a full-objective
eval without code changes, report that as a tooling gap and skip — do not
modify code.

**Report:** for each: full wins/30, crystal %, exit-unlocked rate, depth, end
reasons — plus which objective mode the eval actually ran under.

---

## Phase 1 — A/B the never-run levers (one 300-ep run each, ~2–3 h each)

All runs use the B3s recipe (`run-recipe b3s_conservative_demo_q`), seed 0, and
change exactly ONE thing via the new CLI flags. Confirm in each artifact's
`summary.json` config snapshot that the intended lever (and only that lever)
changed: the snapshot now records `geodesic_potential`, `reverse_curriculum`,
`reverse_curriculum_p`, `show_locked_exit`, and `reward_clip`.

### Task 1.1 — Geodesic PBRS potential ON

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_conservative_demo_q \
  --geodesic-potential --label t11_geodesic_pbrs_on_300_seed0
```

### Task 1.2 — Reverse curriculum ON (p=0.5)

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_conservative_demo_q \
  --reverse-curriculum-p 0.5 --label t12_reverse_curriculum_p05_300_seed0
```

### Task 1.3 — Reward clamp OFF

Tests whether the −5 clamp has been suppressing the terminal signal.

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_conservative_demo_q \
  --reward-clip 0 --label t13_reward_clip_off_300_seed0
```

For each run:

1. Verify `artifact_validation.json` is `ok`.
2. Verify the config snapshot shows the intended lever.
3. Run `compare-artifact <artifact>` against the frozen B3s bar.
4. Extract: selected wins/30, crystal %, depth %, route/contact score,
   near-miss `<=3`/`<=1.5` rates, loop-after-close, stuck-after-close, and
   non-success depth via `metric-audit`.

**Early-stop rule (same as tracker):** if the ep100 source eval shows depth
below `35%` AND crystals below B3s, stop the run and report it as a regression
probe rather than burning the full budget.

**Report:** the metric block for each arm plus the `compare-artifact` verdict
(`PROMOTE`/`HOLD`/`REGRESS`).

---

## Phase 2 — only if directed after Phase 1 review

Do NOT start these without explicit direction; they exist so the follow-up
instruction can be one line.

- **2.1 Combined lever run:** best two Phase 1 arms together, same recipe
  shape, seed 0 (label `t21_combined_300_seed0`).
- **2.2 Seed robustness:** repeat the winning Phase 1 arm on seed 1, then
  `paired-ab` between its selected checkpoint and B3s selected, seeds 0–1,
  30 games (see the `paired-ab` example in `CC_NN_EXPERIMENT_TRACKER.md`).
- **2.3 Expanded validation:** 60-game validation of the winning arm plus
  `compare-artifact --validation`.
- **2.4 Geodesic weight probe:** if 1.1 is positive, one run at
  `--geodesic-potential-weight 0.6`.
- **2.5 Locked-exit map probe:** `--show-locked-exit` alone.

---

## Result block format (append per task to `CC_NN_RUN_INFO_RESULTS.md`)

```markdown
## <task id> — <label>
- artifact: <path>
- validation: ok|failed
- command: <exact command run>
- wall time: <h:mm>
- config snapshot check: <lever values confirmed>
- wins: X/N | crystals: % | depth: % | non-success depth: %
- near-miss <=3: % | <=1.5: % | loop-after-close: % | stuck-after-close: %
- route/contact score: X.XXX
- compare-artifact: PROMOTE|HOLD|REGRESS — <one-line reason>
- end reasons: {...}
- anomalies: <anything unexpected, or "none">
```

Keep a running `## Blockers / tooling gaps` section at the bottom for anything
you were told to skip rather than fix.
