# Crystal Caves NN Tracker Archive: B-Series Route Mastery

Archived from `CC_NN_EXPERIMENT_TRACKER.md` during cleanup on 2026-06-24.

## B-Series: Route-Mastery Plan

The A-series result is consistent: reward pressure, replay seeding, alternate starts,
and broad novelty can improve training/source metrics, but they do not make the greedy
policy collect the first real held-out crystal from a normal full-cave start. The next
work should stop optimizing full clears directly and make first-objective routing a
gated curriculum stage.

### B1. Route-Floor First-Crystal Curriculum

**Status:** tested as `first-crystal-route`; not promoted as sufficient.

**Why this is top-rated:** curriculum-learning surveys emphasize sequencing tasks/data
when the target task is too difficult from scratch, and automatic curriculum work frames
task difficulty as something that should match the agent's current ability. Recent
goal-reaching work also treats goal-hit rate as an early indicator for whether sparse
goal learning will succeed. Locally, the previous `first-crystal-transfer` source policy
briefly reached 25% first-crystal held-out success, then regressed; full-objective
transfer failed. So the immediate gate is not full completion, it is stable first-crystal
routing.

**Implementation plan:**

- Add a `route_floor` procedural difficulty: same full cave shape and normal start, but
  one open-route crystal is placed near the entrance route.
- Add a standalone `first-crystal-route` status-session mode:
  - pretrain on `route_floor` with `CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL`
  - fine-tune/evaluate on normal tutorial `CRYSTAL_CAVES_FIRST_CRYSTAL_GOAL`
  - keep final eval and traces on normal tutorial held-out caves
- Gate promotion on the first-crystal objective, not full wins.

**Promotion rule:** worth promoting only if normal tutorial first-crystal held-out success
reaches at least 50% on the 8-game probe or clearly beats the older 25% peak; then rerun
with 30 held-out games and require roughly 60%+ before full-objective transfer.

**References:** Curriculum Learning for RL Domains
(`https://arxiv.org/abs/2003.04960`), Automatic Curriculum Learning for Deep RL
(`https://arxiv.org/abs/2003.04664`), Revisiting Sparse Rewards for Goal-Reaching RL
(`https://arxiv.org/html/2407.00324v2`).

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py first-crystal-route \
  --route-floor-episodes 75 \
  --episodes 150 \
  --seed 0 \
  --eval-games 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label first_crystal_route_floor_75_150
```

**Training/eval boundary:** route-floor maps are allowed as training scaffolding only.
The final B1 gate remains normal held-out tutorial caves with the first-crystal
terminal objective; later full-game gates should stay as close as possible to the 1991
Crystal Caves style.

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_110004_first_crystal_route_floor_75_150`

**Result:** not enough as implemented.

| Metric | Previous first-crystal source peak | B1 route floor | B1 tutorial route final | B1 tutorial route best |
|---|---:|---:|---:|---:|
| Held-out first-crystal wins | 2/8 | 0/8 | 0/8 | 2/8 |
| Held-out crystals | 25.0% | 0.0% | 0.0% | 25.0% |
| Held-out depth | 14.3% | 4.5% | 7.1% | 8.9% |
| Trace any crystal | n/a | n/a | 0/4 | n/a |

**Finding:** training maps are allowed as scaffolding, but this scaffold did not solve
the route-generalization problem. The route-floor phase reached 20% training wins but
0/8 held-out wins. The normal tutorial route phase reached 36% training wins and briefly
matched the old 2/8 held-out first-crystal peak at episode 51, then regressed to 0/8 by
episode 150. Final traces still collected 0/4 crystals.

**Verdict:** B1 confirms the real blocker is earlier than full-game completion: even
nearby first-crystal routing is not stable under held-out cave variation. Do not promote
route-floor curriculum alone. Next work should add demonstration/action guidance on the
route scaffold before returning to full-objective training.

### B2. Route Demonstrations / DQfD-Lite on Training Scaffolds

**Status:** implementing/testing as `route-demo-bc`.

**Why:** B1 showed that even a near-entrance, walk-reachable training crystal did not
generalize in held-out route-floor eval. This points to an action-selection/credit
assignment problem before a goal-relabeling problem. Demonstration-assisted DQN work
combines TD learning with supervised demonstrator-action pressure, and the key local
lesson from failed A2 is to avoid off-distribution bridge demos. The next version should
use route-floor or normal tutorial first-crystal demonstrations only, then keep final
eval on faithful held-out tutorial/full Crystal Caves levels.

**Lowest-risk test:**

- Generate or script short route-floor first-crystal demonstrations.
- Pretrain or seed replay on those demonstrations with a simple margin/supervised action
  loss if possible; plain replay seeding alone already failed for off-distribution bridge
  demos.
- Evaluate first on route-floor held-out, then normal tutorial first-crystal held-out.

**Promotion rule:** route-floor held-out first-crystal success must become clearly
nonzero and stable before this is allowed to touch full-objective training.

**Reference:** Deep Q-learning from Demonstrations
(`https://arxiv.org/abs/1704.03732`).

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py route-demo-bc \
  --route-floor-episodes 75 \
  --episodes 150 \
  --seed 0 \
  --eval-games 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --route-demo-levels 32 \
  --route-demo-max-steps 800 \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label route_demo_bc_75_150
```

**Training/eval boundary:** scripted route-floor maps are training-only scaffolds. The
final B2 gate remains normal held-out tutorial caves with the first-crystal terminal
objective; this should not change faithful Crystal Caves final levels.

**Artifact B2a:** `.Codex/artifacts/cc_sessions/20260623_130340_route_demo_bc_75_150`

**Result B2a:** weak intermediate signal, not promoted.

| Metric | B1 route floor | B2a route demo floor | B1 tutorial route final | B2a tutorial route best | B2a tutorial route final |
|---|---:|---:|---:|---:|---:|
| Held-out first-crystal wins | 0/8 | 1/8 | 0/8 | 2/8 | 0/8 |
| Held-out crystals | 0.0% | 12.5% | 0.0% | 25.0% | 0.0% |
| Held-out depth | 4.5% | 8.9% | 7.1% | 29.5% | 21.4% |
| Trace any crystal | n/a | n/a | 0/4 | n/a | 0/4 |

**Finding B2a:** scripted route-floor demonstrations are only half reliable (16/32,
1,663 kept transitions). Behavior cloning in the current implementation did not produce
an immediately useful greedy scaffold policy (after-BC route-floor eval 0/8), but RL
with seeded demos reached 1/8 route-floor held-out and transferred to a normal tutorial
route policy that held 2/8 at episodes 51 and 100 before regressing to 0/8 at episode
150. The traces still collected 0/4 crystals and show tile-loop/no-crystal failures.

**Next B2 tweak:** run the same experiment with deterministic behavior cloning through
the NoisyNet mean weights. The current BC pass used training mode, so NoisyLinear sampled
noise during supervised fitting; that likely made the demonstrator action target less
stable. This is a one-change rerun before moving to B3.

**Artifact B2b:** `.Codex/artifacts/cc_sessions/20260623_131607_route_demo_bc_deterministic_75_150`

**Result B2b:** rejected; worse than B2a.

| Metric | B2a original BC | B2b deterministic BC |
|---|---:|---:|
| After-BC route-floor wins | 0/8 | 0/8 |
| Route-floor final wins | 1/8 | 0/8 |
| Route-floor final crystals | 12.5% | 0.0% |
| Tutorial route best wins | 2/8 | 0/8 |
| Tutorial route final wins | 0/8 | 0/8 |
| Tutorial route final crystals | 0.0% | 0.0% |
| Tutorial route final depth | 21.4% | 23.2% |
| Trace any crystal | 0/4 | 0/4 |

**Finding B2b:** deterministic NoisyNet-mean behavior cloning reduced the final CE loss
slightly (`1.1713 -> 1.0877`) but harmed every held-out success signal. The runner was
restored to B2a behavior after this test. The bigger B2 blocker is not stochastic BC
noise; it is low-quality/partial route demonstrations and poor held-out route
generalization.

### B3. Goal-Conditioned/HER-Lite First-Target Relabeling

**Status:** parked behind B3a scaffold-control fix.

**Why:** HER turns failed trajectories into training signal for achieved goals and can be
combined with off-policy algorithms like DQN. For this repo, the safer version is not a
general goal-vector rewrite; it is relabeling short first-target outcomes such as
`reached target region`, `reduced target distance`, or `collected first crystal`.

**Risk:** more invasive replay/state contract changes. Do only after B1 tells us whether
route-floor curriculum can create a reliable first-crystal policy.

**Reference:** Hindsight Experience Replay
(`https://proceedings.neurips.cc/paper/7090-hindsight-experience-replay.pdf`).

### B3a. Shaft-Catch Route Scaffold

**Status:** tested; diagnostic, not promoted.

**Why this comes before HER:** B2 exposed a simpler blocker: the route-floor scaffold was
certified as walk-reachable, but both the scripted controller and the learned policy
often fell past the crystal and could not climb back. That means the training level was
not actually clean practice. Before changing replay/HER contracts, make the training
scaffold truly teachable: normal spawn, normal shaft descent, one crystal on the catch
ledge directly below the shaft. Training levels are allowed to be artificial; final eval
stays normal tutorial Crystal Caves.

**Implementation plan:**

- Add a training-only `route_catch` generator difficulty.
- Put the single first-crystal objective at `(shaft, surface + 2)` when that catch-ledge
  tile is valid, with fallback to the old route-floor nearest walkable placement.
- Add a `--route-scaffold-difficulty` runner option so B1/B2 remain reproducible with
  `route_floor`, while B3a can test `route_catch`.
- Run the same 75 scaffold -> 150 normal tutorial first-crystal probe.

**Promotion rule:** source scaffold held-out should be clearly stable (target: at least
6/8) before we trust transfer. Normal tutorial first-crystal should beat the old 2/8
transient and avoid final regression before moving to full-objective training.

**Pre-run scaffold sanity:** the existing scripted route demonstrator solves
`route_catch` 32/32 with 1,930 kept transitions. The same demonstrator solved only
16/32 on `route_floor`, confirming the old scaffold was not clean practice.

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py first-crystal-route \
  --route-scaffold-difficulty route_catch \
  --route-floor-episodes 75 \
  --episodes 150 \
  --seed 0 \
  --eval-games 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label route_catch_75_150
```

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_135440_route_catch_75_150`

**Result:** not promoted, but very diagnostic.

| Metric | B1 route floor | B2a route demo floor | B3a route catch |
|---|---:|---:|---:|
| Scaffold training win100 | 20% final-ish | 75% final | 92% final |
| Scaffold held-out wins | 0/8 | 1/8 | 0/8 |
| Scaffold held-out crystals | 0.0% | 12.5% | 0.0% |
| Tutorial route best wins | 2/8 | 2/8 | 0/8 |
| Tutorial route final wins | 0/8 | 0/8 | 0/8 |
| Tutorial route final crystals | 0.0% | 0.0% | 0.0% |
| Trace any crystal | 0/4 | 0/4 | 0/4 |

**Finding B3a:** the scaffold itself is now controllable (scripted 32/32), and training
source wins are excellent (92%), but held-out scaffold wins are still 0/8. That is the
clearest evidence so far that the policy is memorizing the small training cave pool
instead of learning a reusable "walk to shaft, descend, collect" rule. This explains why
so many earlier changes looked good in source/training metrics and then failed held-out.

**Next recommended test:** domain-randomized route training. Increase the training cave
pool or generate more varied route-catch/tutorial caves during training so memorization is
harder. If a larger pool improves held-out, keep pushing curriculum/data diversity. If it
does not, inspect the model/state representation and consider auxiliary navigation
supervision.

### B3b. Larger-Pool Route-Catch Domain Randomization

**Status:** tested; scaffold promoted as diagnosis, not enough for tutorial transfer.

**Why:** B3a produced 92% source training wins but 0/8 scaffold held-out wins. The
cleanest hypothesis is training-pool memorization. This probe increases the procedural
training pool from 64 to 512 and gives the scaffold phase 150 episodes, so the policy
sees more route-catch variation before normal tutorial transfer.

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py first-crystal-route \
  --route-scaffold-difficulty route_catch \
  --cave-pool-size 512 \
  --route-floor-episodes 150 \
  --episodes 150 \
  --seed 0 \
  --eval-games 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label route_catch_pool512_150_150
```

**Promotion rule:** scaffold held-out must move meaningfully above 0/8. If it does not,
the issue is not just small-pool memorization; move to architecture/auxiliary objective
supervision.

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_140704_route_catch_pool512_150_150`

**Result:** partially promoted as a diagnosis; not enough for tutorial transfer.

| Metric | B3a route_catch pool64 | B3b route_catch pool512 |
|---|---:|---:|
| Scaffold train win100 | 92% | 100% |
| Scaffold held-out ep50 | 0/8 | 4/8 |
| Scaffold held-out ep100 | n/a | 8/8 |
| Scaffold held-out final | 0/8 | 5/8 |
| Tutorial route ep50 | 0/8 | 0/8 |
| Tutorial route ep100 | 0/8 | 0/8 |
| Tutorial route final | 0/8 | 1/8 |
| Tutorial trace any crystal | 0/4 | 0/4 |

**Finding B3b:** larger-pool route-catch training is the first clean held-out win in this
sequence. Pool 512 reached 8/8 scaffold held-out at episode 100, while the same scaffold
with the default pool 64 stayed 0/8. This strongly supports the "small training pool
memorization" diagnosis. However, the generalized catch-ledge skill does not transfer
far enough to normal tutorial crystal placement: final tutorial route was only 1/8 and
traces still collected 0/4 crystals.

**Decision:** keep `route_catch` and the `--cave-pool-size` runner knob. Do not claim the
agent is ready for full-objective transfer. The next top test should train directly on
normal tutorial first-crystal routing with pool 512 (or add a middle scaffold whose
crystal is off-shaft but not timing-sensitive). The current evidence says data diversity
helps when the training task matches the eval task; catch-ledges are now solved, tutorial
placements are not.

### B3c. Direct Tutorial First-Crystal Pool 512

**Status:** tested; modest positive, not enough to promote.

**Why:** B3b proved data diversity can solve held-out route-catch, but catch placement is
too narrow to transfer to normal tutorial crystal placement. This run removes the
scaffold transfer step and trains directly on the faithful tutorial first-crystal task
with the same larger training pool.

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py first-crystal-direct \
  --episodes 300 \
  --seed 0 \
  --eval-games 8 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --cave-pool-size 512 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label tutorial_route_pool512_direct_300
```

**Promotion rule:** held-out first-crystal success should beat the old 2/8 transient and
avoid final collapse. If it reaches 4/8 or better, rerun with 30 held-out games before
full-objective transfer.

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_143058_tutorial_route_pool512_direct_300`

**Result:** modest improvement over scaffold transfer, but still unstable.

| Metric | B3b route-catch transfer | B3c direct tutorial route |
|---|---:|---:|
| Training win100/final route success | 100% scaffold / n/a tutorial | 46% tutorial |
| Tutorial route ep50 | 0/8 | 1/8 |
| Tutorial route ep100 | 0/8 | 0/8 |
| Tutorial route ep150 | n/a | 2/8 |
| Tutorial route ep200 | n/a | 1/8 |
| Tutorial route ep250 | n/a | 2/8 |
| Tutorial route final | 1/8 | 1/8 |
| Best selected checkpoint | n/a | ep150, 2/8 wins, 25% crystals, 21% depth |
| Tutorial trace any crystal | 0/4 | 0/4 |

**Finding B3c:** direct training on normal tutorial first-crystal routing with pool 512
does learn something useful, but not enough. The best checkpoint reached 2/8 held-out
first-crystal completions, matching the old transient peak and beating B3b's final
transfer signal, but the final checkpoint fell back to 1/8. Training success rose to
46%, while held-out stayed unstable, so this is still not a robust reusable route
policy. Failure traces on the final policy collected 0/4 crystals and remained dominated
by `no_crystal` + `tile_loop` modes; one trace was idle-heavy and one was
interact-heavy.

**Decision:** keep direct normal-tutorial route training as the preferred B-series
baseline, but do not transfer it to full-objective Crystal Caves yet. The next smart
step is to improve the measurement and checkpoint selection around this route gate:
run the same B3c setup with a larger held-out eval sample and save/restore the best
source checkpoint, or add a small amount of stochastic/noisy evaluation to detect
"almost works" policies before committing to longer runs.

### B3d. Selected-Checkpoint Route Gate Verification

**Status:** tested; promoted as the current route-gate measurement method.

**Why:** B3c showed a best checkpoint at episode 150 but final weights regressed. The
old harness recorded the best source snapshot but still traced/evaluated final weights.
This run changes the measurement: keep small/frequent checkpoint evals, restore the best
in-memory weights, then run a larger held-out eval on that selected checkpoint.

**Implementation:** `first-crystal-direct` now accepts `--selected-eval-games`. When set,
the runner restores the best `selected_source_episode` weights after training, evaluates
that selected policy on the requested held-out sample, traces it separately, records
`selected_checkpoint_eval`, then restores the final weights before summary completion.

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py first-crystal-direct \
  --episodes 300 \
  --seed 0 \
  --eval-games 16 \
  --selected-eval-games 30 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --cave-pool-size 512 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label tutorial_route_pool512_select30_300
```

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_173858_tutorial_route_pool512_select30_300`

**Result:** route skill is real but still weak.

| Metric | B3c 8-game probe | B3d selected-checkpoint verification |
|---|---:|---:|
| Checkpoint eval games | 8 | 16 |
| Best checkpoint | ep150, 2/8 | ep150, 3/16 |
| Best checkpoint crystals/depth | 25.0% / 21.4% | 18.8% / 27.7% |
| Final checkpoint | 1/8 | 2/16 |
| Restored selected expanded eval | n/a | 6/30 |
| Expanded eval crystals/depth | n/a | 20.0% / 28.6% |
| Final trace any crystal/depth | 0/4, 7.1% | 0/4, 7.1% |
| Selected trace any crystal/depth | n/a | 0/4, 26.8% |

**Finding B3d:** the old 8-game eval was under-sampling a weak but nonzero route skill.
With 16-game checkpoints, the policy is consistently around 12-19% first-crystal success,
and the selected episode-150 weights hold up at 6/30 on expanded eval. This is the
strongest trustworthy route-gate result so far. It is still far below the 50-60% route
success needed before full-objective transfer is likely to help. The selected trace
improves depth and removes idle-heavy/interact-heavy behavior, but still collects 0/4
crystals and remains dominated by tile loops.

**Decision:** keep selected-checkpoint expanded eval as the standard route-gate method.
Do not run full-objective transfer yet. The next improvement should target the remaining
navigation failure directly: add auxiliary direction/distance-to-first-crystal
supervision or a middle training scaffold with off-shaft but non-timing-sensitive
crystals. Another pure reward or longer-training run is unlikely to move the needle.

### B3e. Off-Shaft Middle Route Scaffold

**Status:** tested; scaffold works, tutorial transfer not promoted.

**Why:** B3b showed a catch-ledge scaffold can generalize when the training pool is
large, but it did not transfer to normal tutorial crystal placement. B3d showed direct
normal-tutorial route training has a real but weak selected-checkpoint signal (6/30).
The next lower-risk test is a middle scaffold: keep normal cave starts and footprint,
but place the single training crystal a short lateral walk away from the shaft catch
ledge. This should teach "descend, orient, walk to nearby crystal" without the full
random tutorial placement distribution.

**Implementation plan:**

- Add a training-only `route_offset` generator difficulty.
- Place the one crystal on a walk/fall-reachable tile near the catch ledge but not in
  the shaft column.
- Allow `--route-scaffold-difficulty route_offset`.
- Extend `first-crystal-route` to restore/evaluate the selected tutorial-route checkpoint
  with `--selected-eval-games`, matching the B3d measurement method.

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py first-crystal-route \
  --route-scaffold-difficulty route_offset \
  --cave-pool-size 512 \
  --route-floor-episodes 150 \
  --episodes 150 \
  --seed 0 \
  --eval-games 16 \
  --selected-eval-games 30 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label route_offset_pool512_select30_150_150
```

**Promotion rule:** promote only if the selected normal-tutorial route checkpoint beats
B3d's 6/30 expanded-eval result or materially improves selected trace any-crystal/depth.
If it only improves scaffold/source metrics, keep it as another scaffold-only success
and move to auxiliary route supervision.

**Artifacts:**

- Main source-history run: `.Codex/artifacts/cc_sessions/20260623_180932_route_offset_pool512_select30_150_150`
- Repro source-history run: `.Codex/artifacts/cc_sessions/20260623_182748_route_offset_pool512_select30_150_150_rerun`
- Selected-eval verification: `.Codex/artifacts/cc_sessions/20260623_184721_route_offset_pool512_select30_150_100`

**Result:** useful scaffold, not a better route policy.

| Metric | B3d direct tutorial | B3e route_offset scaffold |
|---|---:|---:|
| Scaffold held-out | n/a | 10/16 at ep150 |
| Tutorial selected source | 3/16 at ep150 | 4/16 at ep100 in 150-transfer runs |
| Tutorial final source | 2/16 at ep300 | 3/16 at ep150 in 150-transfer runs |
| Verified selected expanded eval | 6/30 | 5/30 |
| Verified selected trace any crystal | 0/4 | 0/4 |
| Verified selected trace depth | 26.8% | 21.4% |

**Finding B3e:** `route_offset` is a valid middle scaffold. With pool 512 it reaches
10/16 held-out first-crystal success after 150 scaffold episodes, so it is much more
teachable than random tutorial placement and less narrow than `route_catch`. However,
the transferred normal-tutorial policy does not beat direct tutorial route training.
Two full 150-transfer passes peaked at 4/16 around episode 100 but final weights
regressed to 3/16. After fixing the route-curriculum selected-eval dispatch, the
shortened verification run selected episode 50 and scored 5/30 on expanded eval, below
B3d's direct-route 6/30.

**Implementation note:** the first two B3e artifacts did not include
`selected_checkpoint_eval` because `--selected-eval-games` was accidentally wired to
unrelated modes and not to `first-crystal-route`. The runner is now fixed for both
`first-crystal-route` and `first-crystal-direct`.

**Decision:** keep `route_offset` as a diagnostic scaffold, but do not promote it as
the next route training baseline. The next top experiment should be auxiliary route
supervision: teach the network an explicit direction/distance-to-first-crystal signal
or add a supervised action head from generated shortest-path/oracle steps. Another
hand-placed first-crystal scaffold is unlikely to clear the remaining tutorial-route
generalization gap.

### B3f. Auxiliary Objective-Direction Supervision

**Status:** tested; not promoted.

**Why this is the next top-rated test:** B3b showed that more data solves a narrow
scaffold, B3d showed direct tutorial route training has a real but weak 6/30 selected
checkpoint signal, and B3e showed another hand-placed scaffold does not transfer better.
The remaining failure looks less like "needs another reward" and more like "the shared
representation does not consistently turn the target compass/objective map into useful
movement." The state already contains active-target `dx`, `dy`, distance, and kind
metadata, so we can add supervised representation pressure without changing faithful
final levels or replay format.

**Implementation:**

- Add opt-in `CRYSTAL_CAVES_ROUTE_AUX_LOSS`.
- Add a 9-way direction head to `SpatialDQN` from the same hidden features used for
  Q-values.
- Derive labels from replay-state metadata slots 15-18:
  `up-left/up/up-right/left/center/right/down-left/down/down-right`.
- Train with `total_loss = DQN_loss + route_aux_weight * CE(direction)`.
- Log `avg_route_aux_loss_100` and `avg_route_aux_accuracy_100` in live metrics and the
  final summary.
- Keep the game objective, procedural tutorial levels, held-out eval, and traces
  unchanged.

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py first-crystal-direct \
  --episodes 300 \
  --seed 0 \
  --eval-games 16 \
  --selected-eval-games 30 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --cave-pool-size 512 \
  --route-aux-weight 0.05 \
  --route-aux-deadband 0.01 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label tutorial_route_aux_w005_pool512_select30_300
```

**Promotion rule:** promote only if the selected normal-tutorial route checkpoint beats
B3d's expanded-eval `6/30` first-crystal result or materially improves selected trace
target-distance/depth without reducing first-crystal success. If aux accuracy rises but
route success does not, the direction signal is learnable but not the missing control
lever; next candidate should be supervised action guidance from oracle route steps.

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_191540_tutorial_route_aux_w005_pool512_select30_300`

**Result:** mechanically successful auxiliary task, worse route outcome than B3d.

| Metric | B3d direct tutorial route | B3f route aux w=0.05 |
|---|---:|---:|
| Selected source checkpoint | ep150 | ep150 |
| Selected source eval | 3/16 | 4/16 |
| Selected source crystals/depth | 18.8% / 27.7% | 25.0% / 18.8% |
| Final source eval | 2/16 | 0/16 |
| Expanded selected eval | 6/30 | 4/30 |
| Expanded selected crystals/depth | 20.0% / 28.6% | 13.3% / 21.7% |
| Selected trace any crystal | 0/4 | 0/4 |
| Selected trace depth | 26.8% | 10.7% |
| Aux loss / accuracy | n/a | 0.076 / 97.3% |

**Finding B3f:** the network learned the direction-label auxiliary task quickly and
reliably (`~97%` recent accuracy), but that did not translate into a better held-out
route policy. The best 16-game checkpoint looked slightly better than B3d (`4/16` vs
`3/16`), but expanded selected eval regressed to `4/30`, below B3d's `6/30`, and selected
traces still collected `0/4` crystals with lower depth. This rules out simple
"the model cannot read target direction" as the main blocker.

**Decision:** keep the metric plumbing because it gives useful early visibility, but do
not promote route-aux direction supervision as a default trainer setting. The next top
candidate should supervise *actions*, not just direction labels: generate oracle or
planner route steps for first-crystal tutorial states and add a small behavior-cloning
/ margin loss on those states, or train a short-horizon option/policy for "move toward
current objective" before full DQN fine-tuning.

**Verification:** focused route-aux tests passed, broader runner/agent/network tests
passed, and the full suite passed: `1012 passed in 20.72s`.

### B3g. Tutorial Route Demonstration BC

**Status:** tested; weak positive, current best route-gate result.

**Why this is next:** B3f proved the model can learn a target-direction representation,
but that did not reliably change control. The next smallest change is action-level
supervision on the actual target distribution: successful scripted first-crystal
trajectories from normal tutorial levels, not route-floor/route-offset scaffolds. This
keeps final levels faithful and tests whether the missing lever is "knowing what to do"
rather than "knowing where the target is."

**Pre-run demo sanity:** the existing scripted controller is not a full oracle on normal
tutorial levels, but it is good enough to produce some on-distribution successes:
`5/16` tutorial first-crystal wins and `404` successful transitions in a quick seed-0
probe. That makes this a valid low-risk test, but not a definitive imitation-learning
solution.

**Implementation plan:**

- Add a `tutorial-demo-bc` status-session mode.
- Generate successful scripted first-crystal trajectories directly on normal tutorial
  training caves.
- Run behavior cloning on those successful `(state, action)` pairs.
- Seed replay with those same successful transitions.
- Train/evaluate the same direct normal-tutorial first-crystal route gate with the B3d
  selected-checkpoint method.

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-bc \
  --episodes 300 \
  --seed 0 \
  --eval-games 16 \
  --selected-eval-games 30 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --cave-pool-size 512 \
  --route-demo-levels 128 \
  --route-demo-max-steps 800 \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label tutorial_demo_bc_pool512_select30_300
```

**Promotion rule:** promote only if selected expanded eval beats B3d's `6/30` and B3f's
`4/30`, or if selected traces finally collect at least one crystal. If it only improves
training/source metrics, the controller demos are too weak or too biased; move to a real
planner/action oracle or option policy instead of more cloning from this heuristic.

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_195536_tutorial_demo_bc_pool512_select30_300`

**Result:** small but real improvement over the prior selected-checkpoint gate; still
below the threshold needed for full-objective transfer.

| Metric | B3d direct tutorial | B3f route direction aux | B3g tutorial demo BC |
|---|---:|---:|---:|
| Demo source | none | none | 35/128 scripted wins, 4,149 transitions |
| After-BC source eval | n/a | n/a | 1/16 |
| Selected source checkpoint | ep150 | ep150 | ep250 |
| Selected source eval | 3/16 | 4/16 | 3/16 |
| Selected source crystals/depth | 18.8% / 27.7% | 25.0% / 18.8% | 18.8% / 40.2% |
| Final source eval | 2/16 | 0/16 | 3/16 |
| Expanded selected eval | 6/30 | 4/30 | 7/30 |
| Expanded selected crystals/depth | 20.0% / 28.6% | 13.3% / 21.7% | 23.3% / 36.2% |
| Selected trace any crystal | 0/4 | 0/4 | 0/4 |
| Selected trace depth | 26.8% | 10.7% | 48.2% |

**Finding B3g:** action-level supervision from successful *normal tutorial* demos helps
more than direction-label supervision. The selected expanded eval improved from B3d's
`6/30` to `7/30`, and selected depth improved to `36.2%` on expanded eval / `48.2%` in
traces. The final checkpoint did not collapse all the way down either (`3/16` source
eval at episode 300). However, the gain is small, the scripted controller only solves
`27%` of tutorial demo attempts, and selected traces still collected `0/4` crystals with
tile-loop/idle/interact-heavy failures. This is not strong enough for full-objective
transfer.

**Decision:** keep `tutorial-demo-bc` as the current best first-crystal route baseline
and as evidence that action guidance is the right direction. Do not keep cloning more
from this weak heuristic controller. The next improvement should replace the heuristic
demo source with a stronger planner/action oracle or add an online DQfD-style supervised
margin/minibatch loss from successful tutorial demo states during DQN training.

**Verification:** status-session tests passed (`31 passed in 2.79s`) and the full suite
passed (`1013 passed in 22.58s`).

### B3h. Online Demo Action Margin Loss

**Status:** tested; useful diagnostic, not promoted over B3g.

**Why this is next:** B3g showed that action-level tutorial demos are a better lever
than target-direction labels, but one-shot BC + replay seeding still lets the policy
drift into tile loops and idle/interact-heavy behavior. The next narrow test keeps the
same demo source and adds a small DQfD-style supervised margin loss from successful
tutorial demo states during every DQN update. This tests whether persistent action
pressure helps more than a warm-start-only clone.

**Implementation plan:**

- Keep the `tutorial-demo-bc` data source: successful normal tutorial first-crystal
  scripted trajectories.
- Add opt-in agent demo-action supervision:
  `max_a(Q(s,a) + margin[a != demo]) - Q(s,demo)`.
- Track demo action loss and greedy demo-action accuracy in live/final metrics.
- Add a `tutorial-demo-dqfd` status-session mode that runs the same B3g setup with the
  online margin loss enabled.
- Keep final eval and traces on normal held-out tutorial caves.

**Probe command:**

```bash
/Users/justin/.pyenv/versions/3.12.11/bin/python -u experiments/cc_status_session.py tutorial-demo-dqfd \
  --episodes 300 \
  --seed 0 \
  --eval-games 16 \
  --selected-eval-games 30 \
  --train-eval-games 8 \
  --eval-every 50 \
  --trace-eval-games 4 \
  --trace-max-steps 3000 \
  --trace-sample-every 25 \
  --trace-tail-steps 120 \
  --vec-envs 8 \
  --cave-pool-size 512 \
  --route-demo-levels 128 \
  --route-demo-max-steps 800 \
  --bc-epochs 6 \
  --bc-batch-size 128 \
  --demo-repeat 4 \
  --demo-action-weight 0.05 \
  --demo-action-margin 0.8 \
  --demo-action-batch-size 64 \
  --heartbeat-seconds 30 \
  --log-every 5 \
  --report-seconds 20 \
  --label tutorial_demo_dqfd_w005_pool512_select30_300
```

**Promotion rule:** promote only if selected expanded eval beats B3g's `7/30` or if
selected traces finally collect at least one crystal. If demo-action accuracy rises but
held-out route success does not, the weak heuristic controller is the bottleneck and the
next step should be a stronger planner/action oracle, not more pressure on these demos.

**Artifact:** `.Codex/artifacts/cc_sessions/20260623_202433_tutorial_demo_dqfd_w005_pool512_select30_300`

**Result:** online action pressure worked mechanically and improved checkpoint stability,
but did not beat the main B3g route gate.

| Metric | B3g tutorial demo BC | B3h online margin loss |
|---|---:|---:|
| Demo source | 35/128 wins, 4,149 transitions | same |
| Online demo loss / accuracy | n/a | 0.0004 / 100% |
| Selected source checkpoint | ep250 | ep250 |
| Selected source eval | 3/16 | 4/16 |
| Selected source crystals/depth | 18.8% / 40.2% | 25.0% / 40.2% |
| Final source eval | 3/16 | 4/16 |
| Expanded selected eval | 7/30 | 7/30 |
| Expanded selected crystals/depth | 23.3% / 36.2% | 23.3% / 35.7% |
| Expanded selected end reasons | 12 stalled / 11 timeout / 7 success | 23 stalled / 7 success |
| Selected trace any crystal | 0/4 | 0/4 |
| Selected trace depth | 48.2% | 53.6% |

**Finding B3h:** persistent margin supervision made the network obey the successful demo
actions almost perfectly (`100%` demo-action accuracy), and it stabilized 16-game source
evals at `4/16` late in training. It did not improve the expanded selected route gate:
`7/30`, tied with B3g, with slightly lower expanded depth and many more stalls. The
selected traces still collected `0/4` crystals. This suggests the current heuristic demo
controller is now the bottleneck: more pressure on the same actions mostly makes the
agent more confidently reproduce a policy that still gets stuck.

**Decision:** keep the online margin machinery and metrics because it is a useful tool,
but do not promote `tutorial-demo-dqfd` over B3g as the default baseline. The next step
should be a stronger action oracle/planner for tutorial first-crystal paths, or a
targeted anti-stall/mask applied to the demo-guided policy. Do not spend more runs on
higher weights for the same weak controller until the demo source improves.

**Verification:** focused demo-action tests passed (`4 passed in 1.41s`) and the full
suite passed (`1017 passed in 22.61s`).

### B4. Adaptive Reverse Route Curriculum

**Status:** queued after B2/B3.

**Why:** the static reverse-start probe did not transfer, but reverse curriculum research
uses start states increasingly far from a known goal. A better version would move starts
backward along actual successful first-crystal trajectories, not teleport near arbitrary
objectives.

**Risk:** needs trajectory/state serialization and careful off-distribution controls.

**Reference:** Reverse Curriculum Generation
(`https://arxiv.org/abs/1707.05300`).

### B5. Hierarchical First-Objective Options

**Status:** queued later.

**Why:** once first-crystal routing is measurable, options like "go to first crystal",
"go to exit", and "collect next crystal" become natural subpolicies.

**Risk:** high implementation surface; should follow a working first-crystal gate.
