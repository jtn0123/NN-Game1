# Nav-Expansion Brief — breaking the leg-2 route-to-exit wall

> Produced by a 5-lens brainstorm workflow (spatial-arch / memory / aux-objectives /
> input-features / algorithm) → adversarial reality-check → synthesis, all grounded by
> reading the actual repo. 15 proposals generated, filtered to the shortlist below.

## 1. The confirmed wall
The agent finds and collects crystals on unseen levels (leg-1 generalizes, ~0.24) and
finishes when standing next to the open exit (NEAR probe 0.733). The wall is **long-range
routing to the exit on unseen layouts**: dropped ~17 reachable tiles from the open exit it
reaches it only **0.117** of the time — because its only directional signal is a
**euclidean compass that points straight at the exit *through walls***.

## 2. Ranked shortlist

### #1 — BFS shortest-path next-step compass *(input feature)* — EV: HIGH, effort: S
Append ~4-6 scalars to metadata: `step_dx, step_dy` (unit vector toward the lowest-BFS-
distance neighbor = the true next corridor step), `on_path`, `reachable`, read from the
already-cached `_geodesic_distance_field()` via a few dict lookups/step. Keep the euclidean
compass too. NEAR=0.733 proves a *locally-aimed* hop is easy → this makes **every** tile a
locally-aimed hop by pointing down the traversable corridor instead of into a wall. It's a
function of *local connectivity*, not memorized geometry, so it transfers. ~1.3K params,
**zero network edits** (`state_size` derives from `METADATA_SIZE`; appending after meta
index 19 leaves route_aux indices 15-18 untouched). Observation, not reward → does **not**
re-trigger the disconfirmed geodesic-PBRS lever. Measure: FAR probe 0.117 → higher; NEAR
and collect-rate must not regress; conversion/win should rise.

### #2 — Geodesic next-step-direction aux head *(9-way classification)* — EV: MEDIUM, effort: M
A 9-way CE head off the shared trunk whose **label is the geodesic next-step direction**
(NOT euclidean — the existing route_aux head re-derives the euclidean dx/dy already in the
state, which is why it was a null). Forces the encoder to represent "which way the corridor
actually goes." Needs a new trainer hook to push `(state, geodesic_dir)` into a bounded
dataset (the label isn't reconstructible from replay state). Risk: "head learns it, policy
ignores it." **Judge on the FAR probe, not aux accuracy.** Stacks cleanly with #1 (shared
BFS field): #1 feeds the signal, #6 forces the trunk to internalize it.

### #3 — Visited-trail memory channel *(observation-only)* — EV: MEDIUM, effort: S
Per-episode decaying visit-count grid at the existing 11×6 global-map resolution
(increment player's cell, decay ~0.97/step), appended as a second channel. The env already
tracks visit grids at this resolution. Attacks re-entering the same dead-end. **Keep
`NOVELTY_REGION_BONUS` OFF** — observation only; any reward attached *is* the disconfirmed
novelty lever. Weaker/complementary to #1/#2 (says "where I've been," not "which way out");
best as a cheap add-on once #1 lands.

### Free baseline — C51 distributional DQN
One flag (`USE_DISTRIBUTIONAL_DQN=True`) + tune V_MIN/V_MAX≈-10..110. Fully implemented.
General value-learning lift, **not** a nav fix — expect a small conversion nudge, FAR gap
largely intact. ~10-20% per-step overhead; verify vec-envs 8 stays in budget. Run in
parallel, never as the fix.

## 3. Recommended sequence
1. **Ship #1 (BFS compass) first** — highest EV-per-effort, S, no network/eval changes,
   supplies exactly the signal the FAR probe says is missing. Clean complement to the
   in-flight RUN-12 FAR curriculum (curriculum *drills* the skill; #1 gives the agent the
   observation to *express* it). Run together.
2. If #1 moves FAR but plateaus → **stack #3 (trail channel)** (anti-backtrack memory).
3. For a representation-level bet → **add #2 (geodesic aux head)**; reuses #1's BFS field.
4. **Flip C51 in parallel** as a free Rainbow-completeness lift, never the fix.

**Pre-registration discipline:** every lever is an A/B judged on `diagnose_gap.py` **FAR
(0.117) and held-out win (0.033)** — never train-set numbers or aux accuracy (several
levers re-expose absolute position and look great on the fixed training pool while doing
nothing on unseen FAR starts).

## 4. The bigger swing (only if the cheap fixes all stall)
**Lightweight recurrent DQN — GRU head + R2D2-style sequence replay.** The principled
answer (true long-horizon memory: "I explored the left branch, it dead-ended 30 steps
ago"), which a Markov policy and a frame-stack fundamentally cannot represent. **But** it's
a multi-day invasive rewrite that breaks the per-transition buffer, PER sum-tree, n-step
accumulation, and flat training tensors (needs a new `SequenceReplayBuffer`); CPU BPTT
likely violates the steps/sec budget; recurrent DQN + NoisyNets is notoriously unstable for
a fragile learner. **Do not start here.** Only if #1+#2+#3 all fail to move FAR *and* the
time budget is explicitly relaxed. The #3 trail channel is the cheap recurrence-flavored
middle ground.

## Key files
- `src/game/crystal_caves.py` — `_geodesic_distance_field` (~719, cached), `state_size`/
  `METADATA_SIZE` (~432), `get_state`/`_fill_global_map`
- `src/ai/network.py` — route_aux head (~746), SpatialDQN `_features` (~757), distributional
  path (~797)
- `src/ai/agent_experiments.py` — `register_auxiliary_loss_provider` (~229), route_aux meta
  indices 15-18 (~374)
- `src/ai/trainer.py` — step loop for the aux dataset hook
- `config.py` — flags; `USE_DISTRIBUTIONAL_DQN` (~396)
- `experiments/cc_status/diagnose_gap.py` — FAR/NEAR acceptance test
