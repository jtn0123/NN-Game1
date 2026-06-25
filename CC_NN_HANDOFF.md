# Crystal Caves DQN — Investigation Handoff (for external review)

**Date:** 2026-06-22
**Repo:** NN-Game1 (Deep Q-Learning agent that learns to play a clean-room Crystal Caves
clone, with a live NN visualizer + Flask/JS dashboard).
**Purpose of this doc:** a self-contained summary of the current investigation so an
external reviewer can validate the reasoning, sanity-check the code claims, and suggest
improvements. Please be adversarial — several conclusions are single-seed and noisy.

---

## 1. The task & the agent

**Game (Crystal Caves clone, `src/game/crystal_caves*.py`):** a tile-based puzzle
platformer. Win condition: **collect EVERY crystal in a level → the exit unlocks →
reach the exit**. Mechanics: walk, fixed-height jump (~3 tiles), fall, colour-keyed
lever→door, elevators, hazards (spikes/acid), limited ammo. 10 discrete actions
(idle/left/right/jump/left-jump/right-jump/shoot/left-shoot/right-shoot/interact).

**Agent (`src/ai/`):** DQN with Dueling + NoisyNets + n-step returns (n=6) + a CNN over
a spatial state (`SpatialDQN` in `network.py`). State = 19×11 local tile window + 11×6
coarse global objective map + ~20–27 normalized metadata scalars. PER is configured ON
but is **silently disabled whenever n-step is on** (known limitation, see §4). Reward is
potential-based shaping (Φ = 0.56·crystals + 0.15·switch + 0.09·depth + 0.20·won, scaled
×10) plus discrete bonuses (crystal +5, all-crystals +16, switch +8, exit terminal +25)
and a small per-step living penalty (−0.01).

**Training:** vectorized (8 envs), CPU (faster than MPS for this small model). A staged
**curriculum** (`src/app/crystal_curriculum.py`) warm-starts each difficulty tier from
the prior tier's held-out eval-best, gated on held-out crystal/win/timeout evidence.

---

## 2. The core problem we are chasing

**The greedy (deterministic) policy collects crystals but cannot finish a level.**
On held-out caves it grabs the crystal in ~10–33% of levels, gets ~50% deep, then
**times out before reaching the exit → ~0 wins** all the way up to 800 training episodes.
The historical best ever recorded (in prior sessions, per project memory) was ~18.5%
held-out win rate with long training, but it was unstable (collapsed past the peak).

---

## 3. Branch / commit map (IMPORTANT — most work is NOT merged)

`origin/main` = **adde927 (v0.3.1)**. Contains merged PRs:
- **#27** gated Crystal Caves curriculum
- **#28** stall fix + dependency/dashboard/eval hardening
- **#29** N-step vectorized-path fix (N1) + faithful winnable-floor level design (L1/L3/L5)

Unmerged experiment branches (each branched from main):

| Branch | Commits over main | What it adds | Validation result |
|---|---|---|---|
| `claude/cc-drill-levels` (**current**) | `d1b3db5`, `3ed9731` | 6 hand-authored single-skill **drill levels** + a skill-diagnostic harness | see §6 — promising |
| `claude/cc-nav-features` | `9ae873b`, `b896430`, `014f664` | **O1/O2** nav-perception state features; **R1** geodesic (walk-distance) reward; **fix#2** memoize + **run-catalog** tooling | O1/O2 ≈ noise; R1 ≈ neutral/negative |
| `claude/cc-reward-balance` | `95a739b` | **R2** raise win terminal reward 25→50 | **no effect** (ruled out) |

All branches pass full test suite + mypy/ruff/black. None of the experiment branches
(beyond what's in main via #29) are merged, because none produced a clear, validated win.

---

## 4. Confirmed correctness findings

- **N1 (FIXED, merged in #29):** `NStepReplayBuffer` overrode `push()` for n-step
  accumulation but **not** `push_batch()`, so the **vectorized** path (which the
  curriculum forces) stored raw **1-step** transitions while `_compute_q_values` applied
  `gamma**6`. Result: every curriculum target was `r₁ + γ⁶·Q(s₁)` — a 1-step reward with
  a 6-step discount. Fixed with per-env n-step accumulation in `push_batch`. **Effect
  (A/B, same seed/caves):** training Q-values went from **−0.06 (pinned at the
  credit-assignment wall) → −0.58**, Φ-best 0.79→0.99, training score 3.3×. BUT held-out
  wins stayed ~0% at 300 episodes — N1 fixes *learning*, not the navigation wall, at that
  horizon. ⚠️ *Reviewer check:* the n-step buffer still uses a windowed flush + uniform
  `gamma**n` for truncated-tail transitions (a pre-existing approximation); is that
  biasing targets?
- **N2 (open):** n-step and PER are mutually exclusive (`agent.py` buffer-selection
  `if use_n_step … elif PER`). Both flags default True → PER silently off. Only a warning
  was added; a prioritized-n-step buffer is not implemented.

---

## 5. Experiments run this session (all single-seed, tutorial tier, 300 ep unless noted)

Held-out = 30-game greedy gate eval on unseen caves. ⚠️ **Single-seed; "wins" is
0–1/30, i.e. noise-dominated. Crystal% is the steadier signal.**

| Experiment | held-out crystals | depth | wins | verdict |
|---|---|---|---|---|
| Baseline (main: N1 + level design) | 17% | 48% | 0/30 | — |
| **Level design L1/L3/L5** (guarantee crystals walk-reachable) | **0%→17%** | ↑ | 0 | **clear win** (merged #29) |
| O1/O2 nav-perception features | 10% | 36% | **1/30** | flicker, ≈ noise |
| R1 geodesic reward (vs O1/O2) | 17% | 22% | 0/30 | neutral/negative |
| R2 raise win reward 25→50 | 17% | 48% | 0/30 | **no effect** — byte-identical to baseline; the reward only triggers on a win, which never happens → ruled out reward as the lever |
| Full stack (N1+L+O1/O2+R1), **800 ep** | bounced 25/33/17/17/8 | ~50% | **0** thru ep750 | longer training did NOT crack it |

**Methodology conclusion:** single-seed short runs cannot resolve the "wins" metric
(it's lost in noise). The only changes that gave clear signal did so on the **steadier
crystal-collection metric**, and the clearest win came from **level design** (changing
the levels), not agent-side tweaks. The run-catalog tooling (`cc-nav-features`) was built
to halve runs (save a baseline once, compare new runs to it).

---

## 6. Current direction: skill-drill levels (the promising part)

**Hypothesis:** the full levels never give the agent enough clean reps of the
"collect-then-jump-to-exit" moment, so it never learns it. Fix = a set of tiny levels
that each isolate ONE motor skill (true to real 1991 Crystal Caves, whose early levels
each teach one mechanic). Two uses: **diagnostic** (which skills can it learn?) and
**teaching** (pre-train the skills, transfer to full game).

**6 hand-authored drills** (`src/game/crystal_caves_drills.py`, 18×44 grids, all verified
solvable by the jump-aware oracle): walk+collect, jump-up, jump-a-gap, drop-and-climb,
staircase, collect-then-jump-to-exit. Loaded via `CRYSTAL_CAVES_DRILLS` config flag.

**Diagnostic result (600 episodes training directly on the drills, greedy per-skill eval,
`experiments/drill_train.py`):**

| Drill (skill) | Greedy solved? |
|---|---|
| walk + collect | ✅ |
| jump up onto a ledge | ✅ |
| drop into a pocket & climb out | ✅ |
| **collect, then jump up to the exit** | ✅ **(this is the full-game wall!)** |
| jump across a (lethal-acid) gap | ❌ |
| climb a 4-step staircase | ❌ |

**Interpretation:** the agent CAN learn the exact skill it fails in full levels (100% on
the dedicated reach-exit drill) — so it's **not physically incapable; it lacked clean
practice**. This validates the teaching-set approach. Two skills fail: the **gap** (working
hypothesis: lethal acid teaches jump-avoidance) and the **staircase** (working hypothesis:
chaining 4 jumps is a hard credit-assignment problem). ⚠️ *Reviewer check:* greedy eval is
deterministic, so each drill reads binary solved/not — is that masking "almost solves"?
Also train==eval caves for drills (fixed levels), so this measures mastery, not
generalization.

---

## 7. Open questions for review / what we're trying next

1. **Transfer test (recommended next):** pre-train on the working drills, drop the policy
   into full levels — does "collect-then-jump-to-exit" finally appear as real wins? This
   is the key unproven claim.
2. **Fix the 2 failing drills:** soften the gap (non-lethal pit) and add a gentler
   staircase; does the avoidance/chain hypothesis hold?
3. **Is the methodology sound?** Single-seed 300-ep A/Bs for a rare-event metric (wins) —
   should we be multi-seed / longer / eval with small noise for a graded signal?
4. **Is N1's n-step implementation fully correct** (truncated-tail gamma)? Should PER be
   restored on top of n-step (N2)?
5. **Was R1 (geodesic reward) given a fair test?** It was tile-lumpy (only changes when
   crossing a tile boundary) and tested only 300 ep. Is the implementation or the test the
   problem?
6. **Generalization vs memorization:** held-out caves use a disjoint seed-offset pool;
   is that a genuine generalization measure?

## 8. How to reproduce
- Drill diagnostic: `python experiments/drill_train.py --episodes 600 --seed 0`
- A/B harness (tutorial tier, seeded, deterministic caves):
  `python experiments/ab_tutorial.py --label X --episodes 300 --seed 0 [--difficulty tutorial]`
- Run catalog (on `cc-nav-features`): `python experiments/run_catalog.py show | compare A B`
- Full test suite: `python -m pytest -q` (≈950 tests, all green on each branch)

## 9. Key files
- `src/game/crystal_caves.py` / `crystal_caves_logic.py` — game, reward shaping, state
- `src/game/crystal_caves_gen.py` — procedural generator + `cave_reachable` solvability oracle
- `src/game/crystal_caves_drills.py` — the hand-authored drills (current work)
- `src/ai/network.py` — `SpatialDQN` / `DuelingDQN` / `NoisyLinear`
- `src/ai/replay_buffer.py` — `NStepReplayBuffer` (N1 fix here)
- `src/ai/agent.py` — DQN agent, `_compute_q_values` (double-DQN target, gamma**n)
- `src/app/crystal_curriculum.py` — staged curriculum + gate logic
- `experiments/drill_train.py`, `experiments/ab_tutorial.py`, `experiments/run_catalog.py`
