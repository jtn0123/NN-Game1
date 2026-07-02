# M4 training runbook (Codex agent ↔ Claude via this PR)

This PR is a **communication channel**, not something to merge. A Codex agent on a
Mac M4 runs the heavy training experiments here (the M4 is several times faster than
the cloud box Claude runs on) and reports results back as **PR comments**. Claude
reads those comments and decides the next experiment.

## Operating protocol

1. **Always `git pull` first.** Claude pushes new flags/experiments to branch
   `claude/cc-reverse-curriculum`. Run from the latest commit.
2. **Run the experiment(s) in the "CURRENT TASK" section below.**
3. **Report by posting a PR comment** whose first line is exactly:
   `## M4 RESULT <short-tag>` (e.g. `## M4 RESULT poolsize-sweep`).
   Paste the printed `LEARNING CURVE`, `PHASE 0 DIAGNOSIS`, and `best checkpoint`
   blocks verbatim, plus total wall-clock. If a run fails, post
   `## M4 ERROR <tag>` with the traceback.
4. **Do not edit code** unless Claude explicitly asks in the task. Do not merge.
5. After reporting, **stop and wait** for Claude's next comment (Claude will reply
   with the next task in a comment starting `## NEXT TASK`).

## One-time setup

```bash
git checkout claude/cc-reverse-curriculum
git pull origin claude/cc-reverse-curriculum
python -m venv venv && source venv/bin/activate   # or your existing env
pip install -r requirements.txt
python config.py                                   # sanity check
```

## Speed notes (M4)

- Pass `--cpu` to every run. For this ~50K-param model, **CPU is faster than MPS**
  on Apple Silicon (confirmed in CLAUDE.md benchmarks). `--cpu` sets `FORCE_CPU`.
- `--vec-envs 8` is a good default (raise to 10 if you have headroom).
- Each `diagnose_gap` run writes `diagnosis.json` + per-checkpoint eval logs under
  `--out`; the human-readable report is printed to stdout (that's what to paste).

## What the diagnostic measures

`experiments.cc_status.diagnose_gap` trains the baseline agent, then scores it on
both its **training** levels and disjoint **held-out** levels, every
`--checkpoint-every` episodes. It reports the seed-averaged learning curve, the
**best checkpoint** (not the collapsed final one), and the train−test gap. The core
question is whether the **train ≫ test memorization gap** shrinks as we change a
lever.

---

## CURRENT TASK — pool-size sweep (does more level variety reduce memorization?)

Background: the one robust finding so far is a large train≫test gap (the agent
memorizes its training caves). The RL-generalization literature (ProcGen/CoinRun)
says the #1 lever is the **number of training levels**. We currently train on only
~24. Test whether scaling that up closes the gap.

Run these three (same everything except `--pool-size`), CPU-forced, 5 seeds:

```bash
python -m experiments.cc_status.diagnose_gap --difficulty tutorial --episodes 1200 \
  --seeds 0,1,2,3,4 --games 30 --checkpoint-every 300 --vec-envs 8 --cpu \
  --truncation-bootstrap --pool-size 24   --out scratchpad/m4_pool24

python -m experiments.cc_status.diagnose_gap --difficulty tutorial --episodes 1200 \
  --seeds 0,1,2,3,4 --games 30 --checkpoint-every 300 --vec-envs 8 --cpu \
  --truncation-bootstrap --pool-size 256  --out scratchpad/m4_pool256

python -m experiments.cc_status.diagnose_gap --difficulty tutorial --episodes 1200 \
  --seeds 0,1,2,3,4 --games 30 --checkpoint-every 300 --vec-envs 8 --cpu \
  --truncation-bootstrap --pool-size 1024 --out scratchpad/m4_pool1024
```

Then post one PR comment `## M4 RESULT poolsize-sweep` containing, for each pool
size: the `best checkpoint` line and the `PHASE 0 DIAGNOSIS` table (train, held-out,
gap), plus total wall-clock. The number we care about most: **held-out win rate and
held-out crystal fraction at the best checkpoint, and how the train−test gap changes
from pool 24 → 256 → 1024.**

(If 1024 is too slow to build/train, run 24 and 256 first, report those, and note it.)
