# Crystal Caves NN — Run-Info Task Brief (rev 3, 2026-07-07)

**Who this is for:** an execution agent tasked with *gathering run evidence only*.
Do not change model code, rewards, or hyperparameters beyond the explicit CLI
flags listed per task. Do not promote/demote baselines. Run the commands,
wait, verify artifacts, report numbers.

**Rev 2 note:** rev 1 of this brief queued A/Bs for geodesic PBRS (reward),
reverse curriculum, and the reward clamp. PR #37's branch already ran all
three families on its 16-level benchmark and closed them (RUN-06/07/08,
RUN-10/12, RUN-23/23b — see `experiments/cc_status/RUN_LOG.md`). Those tasks
are DROPPED. What remains open is (a) baseline integrity on the main track,
(b) the cross-track compass test, and (c) the demo-era prep the decision
brief calls RUN-26.

**Rev 3 status:** PR #37 is MERGED to main (2026-07-08, merge commit
`c03eddf`) and this PR is rebased on top of it. Post-merge verification is
already done on this branch: full suite **1484 passed** and
`python -m experiments.cc_status.compass_audit` reports **zero**
trapped/dead/compass-hard-lie cells across all 16 imported levels. The
`--geo-compass` / `--geo-compass-hazard-aware` flags are now wired into
`cc_status_session.py` (this PR), so Task 2.1 is directly runnable. Every
task below runs on merged main.

**Ground rules:**

- Python: `/Users/justin/.pyenv/versions/3.12.11/bin/python`.
- Track A (B-series) artifacts land under
  `.Codex/artifacts/cc_sessions/<timestamp>_<label>/`; trust only runs whose
  `artifact_validation.json` says `ok: true`. Track B (RUN-NN) runs follow the
  RUN_LOG convention (`## M4 RESULT RUN-NN <tag>`, `--out scratchpad/RUN-NN_<tag>/`)
  and protocol v2: best-checkpoint + seed-averaged grading, checkpoints saved,
  a "winning" checkpoint must repeat on a fresh seed set.
- Frozen Track A baseline **B3s**: `10/30` selected first-crystal wins seed 0,
  `19/60` expanded validation. Checkpoint:
  `.Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300/tutorial_demo_conservative/models/crystal_caves/tutorial_demo_conservative_selected_ep300.pth`
- A 300-episode Track A recipe run ≈ 2–3 h on the M4; a 3-seed 4000-ep Track B
  arm ≈ 1.5–2.5 h. Eval-only tasks take minutes.
- After each task, append the result block (bottom of this file) to
  `CC_NN_RUN_INFO_RESULTS.md`. Include artifact paths verbatim.

---

## Phase 0 — cheap eval-only sanity on main (no training; ~30 min)

### Task 0.1 — Baseline drift check: does frozen B3s still reproduce post-merge?

PRs #34/#35/#36/#37 all changed eval selection, reset paths, seeding
determinism, or shared game code after B3s was frozen. Same checkpoint, same
seed-0 held-out eval, now on merged main:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py eval-checkpoint \
  --checkpoint .Codex/artifacts/cc_sessions/20260624_120002_tutorial_demo_conservative_recovery_pool512_select30_300/tutorial_demo_conservative/models/crystal_caves/tutorial_demo_conservative_selected_ep300.pth \
  --eval-games 30 --seed 0 --label t01_b3s_drift_check_post37_eval30
```

**Report:** wins/30, crystal %, depth %, end reasons; flag if wins deviate
from 10/30 by more than ±3. If drifted, every Track A comparison needs
re-baselining before Task 2.1 is judged against the frozen B3s bar — report
and wait for direction rather than improvising a new baseline.

### Task 0.2 — Full-level completion eval of B3s/B21 (not first-crystal)

Unchanged from rev 1: every Track A promoted number is a first-crystal
metric. Evaluate B3s and the B21 adapter checkpoint (locate the `.pth` under
`.Codex/artifacts/cc_sessions/20260625_181012_b21_stable_contact_head_offline_conf075_eval30/`)
on the same seed-0 held-out surface, 30 games, full objective. The
first-crystal early terminal is `CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL` in the
config snapshot — confirm from the artifact's `summary.json` which objective
the eval actually ran; if full-objective eval isn't reachable without code
changes, log it as a tooling gap and skip.

**Report:** full wins/30, crystal %, exit-unlocked rate, depth, end reasons,
objective mode confirmed.

---

## Phase 1 — post-merge integrity — ✅ DONE (recorded here for the log)

Done on the PR #39 branch after the #37 merge and rebase, 2026-07-08:
full suite **1484 passed**; `compass_audit` clean (zero trapped / dead /
compass-hard-lie cells on all 16 imported levels). No agent action needed.

---

## Phase 2 — the open experiments

### Task 2.1 — Cross-track compass A/B (the one big untested combination)

Track B's biggest lever (`--geo-compass`, +4 state dims, held-out tutorial
wins 0.033 → 0.483) has never been tried on Track A's promoted lineage.
Run the B3s recipe with the compass enabled, seed 0. State size changes, so
this is a fresh train, not a checkpoint restore — same caveat as the B8
history-state probe. The flag is wired into the status-session CLI and the
run's config snapshot records `geo_compass` for attribution:

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py run-recipe b3s_conservative_demo_q \
  --geo-compass --label t21_b3s_geo_compass_300_seed0
```

Judge with `compare-artifact` against the frozen B3s bar plus near-miss
metrics. **This is the highest-information single run available:** if the
compass lifts the first-crystal surrogate the way it lifted tutorial wins,
Track A's whole lineage gets a step-change for four state dims.

### Task 2.2 — RUN-26 demo run (BLOCKED on owner playtest — do not start)

Per the branch decision brief, the demo path is the only high-evidence family
never tried. It needs human demos first:

- **Owner action (not the execution agent):**
  `python main.py --human --imported --record-demos` — play the 16 imported
  levels, wins matter most. Verify with
  `experiments/cc_status/human_demos.py` (`verify_stored`).
- Then RUN-26 shape (per DECISION_BRIEF.md): DQfD-lite margin loss and/or
  backward curriculum from demo states, WITH the fidelity fixes riding along:
  stall window 720 → ~1440 (needs `MAX_STEPS_WITHOUT_PROGRESS` made
  configurable) and truncation-aware bootstrapping. Protocol v2, 3 seeds,
  checkpoints saved, fresh-seed replication before any promotion claim.

### Task 2.3 — (only if directed) selection-score rebalance validation

After the evaluator's `selection_score` is rebalanced so win strictly
dominates exit-unlock, re-run Task 0.1 to confirm keep-best ordering is
unchanged for existing checkpoints.

---

## Result block format (append per task to `CC_NN_RUN_INFO_RESULTS.md`)

```markdown
## <task id> — <label>
- artifact/out dir: <path>
- validation/preflight: ok|failed
- command: <exact command run>
- wall time: <h:mm>
- headline: wins X/N | crystals % | depth/progress % | exit-unlock %
- support: near-miss <=3 % | <=1.5 % | loop-after-close % | stuck-after-close %  (Track A)
           killed/stalled/timeout split | FAR/NEAR probe                          (Track B)
- verdict vs baseline: PROMOTE|HOLD|REGRESS — <one line>
- end reasons: {...}
- anomalies: <anything unexpected, or "none">
```

Keep a running `## Blockers / tooling gaps` section at the bottom for anything
skipped rather than fixed.
