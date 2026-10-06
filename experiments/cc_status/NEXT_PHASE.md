# Crystal Caves DQN — Phase 2: The Opening-Starvation Program

Continuation of the completion-wall campaign (PR #39, merged 2026-07-18; see
`CAMPAIGN_LOG.md` for the full record). Phase 1 ended with one conquered,
exam-verified summit (Switchback, 42/100 from-spawn wins) and three levels
bracketed at 64–93% by a single named blocker: **opening starvation** — the
backward ladder's final rungs (the route's first steps from spawn) receive
almost no win signal, because banking them requires full-route wins that are
rare from deep starts. Two independent 72k-episode Stalactite campaigns
(gen-5: 92% ladder / 0.923 exam; gen-6: 88% / 0.877) established that
re-running the recipe does not cross this gap.

## Ranked program (expected value ÷ cost)

1. **Opening-focused imitation** — behavior-clone ONLY the first ~300 steps of
   each demo route (exactly the segment the ladder cannot reach), then hand off
   to the ladder. New lever on existing infrastructure. First target:
   Stalactite (closest summit, 92%).
2. **Shortest-route re-harvest** — the route-length law says summit difficulty
   tracks route length, not crystal count. Bias the Go-Explore harvester toward
   the shortest winning trace (or re-harvest parked levels under a tighter
   cap) to soften Ore Shaft (2,998) and Scaffold.
3. **Exam-in-the-loop promotion** — wire `big_exam.py` into the harness as a
   periodic 100-episode focus-level exam. The 9-episode checkpoint eval cannot
   see a 10–40% win rate; automatic champion detection prevents a repeat of the
   winner's-curse promotion mistake.
4. **Demo the five unsolved levels** (5, 6, 8, 9, 10) — no demo ⇒ no ladder ⇒
   no path to a win. Needs a stronger planner or human demos.
5. **Massively parallel workers** — Salimans & Chen ran ~1,000 workers against
   Montezuma; we run 8 vec-envs. A PPO/IMPALA-scale rewrite is the
   literature-backed cure for opening starvation. New-project-sized; the
   phase-2 flagship if levers 1–3 stall.
6. **Ops**: run long jobs under `caffeinate -i`; two machine sleeps and one
   battery death each cost hours in phase 1.

## Laws carried forward (violate at your peril — each cost us a run)

- Promote by WIN metric from ≥100-episode exams only (`big_exam.py`);
  champions are mid-run artifacts — never take the final net.
- Exams keep the full 16-level eval context (state includes level index).
- Never resume a run; fresh campaign per target, bank checkpoints.
- Dead levers stay dead without new evidence: demo gradients, win-at-K
  curricula, CNN representation, consolidation passes.

## Tracking convention

As with PR #39: every run launch, verdict, and protocol change gets a
milestone comment on the tracking PR; durable results graduate into
`CAMPAIGN_LOG.md`.
