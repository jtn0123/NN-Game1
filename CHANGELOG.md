# CHANGELOG


## v0.7.0 (2026-06-26)

### Features

- **crystal-caves**: Reverse-curriculum mid-solution starts (roadmap #4, v1)
  ([`611e534`](https://github.com/jtn0123/NN-Game1/commit/611e5348cedd7e13e934e678b0f8a84e2bc53bb0))

Adds an opt-in reverse curriculum: on a fraction `p` of TRAINING resets (never in eval), the episode
  begins from a valid mid-solution state so the agent gets dense reps of finishing the collect ->
  ... -> exit chain instead of only ever seeing it from scratch (the documented full-game wall).

v1 (safe, no player relocation): pre-collect a random subset of crystals (those farthest from the
  exit) and open every gate, leaving the player at the level spawn. This is solvability-preserving
  by construction — a subset of the objectives with all doors open stays reachable from a spawn that
  could already clear the full level — so it never creates an unwinnable start. exit_unlocked is set
  consistent with the remaining crystals.

- config: CRYSTAL_CAVES_REVERSE_CURRICULUM (bool, default off) + CRYSTAL_CAVES_REVERSE_CURRICULUM_P
  (float in [0,1], validated in __post_init__). - CrystalCaves.set_reverse_curriculum_p() lets a
  trainer anneal p toward 0 over training so the policy finishes on full-length episodes. - Applied
  before the progress/closeness baselines so PBRS reflects the start; skipped entirely in eval mode
  so held-out eval always measures the full task.

Deferred to a follow-up (the oracle-verified part the reviewers flagged as the riskier infra):
  relocating the player toward the remaining objectives to also shorten the navigation horizon.

Off by default — an attributable A/B lever on the chain-progress surrogate.

Tests: off-by-default/p=0 keep the full start; p=1 pre-collects a strict non-empty subset, opens all
  gates, keeps state normalized; eval mode is unaffected; the p setter clamps to [0,1]. Full suite:
  1097 passed, 115 skipped. ruff/black/mypy clean.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_018jGv8TVpr6WFFbGgnZjAwk


## v0.6.0 (2026-06-26)

### Bug Fixes

- **crystal-caves**: Address review — validate geodesic weight, harden tests
  ([`3f35a2b`](https://github.com/jtn0123/NN-Game1/commit/3f35a2b3d410adc19cb4256ab9a1ebe6d227b06a))

CodeRabbit review on #35: - Validate CRYSTAL_CAVES_GEODESIC_POTENTIAL_WEIGHT in Config.__post_init__
  (must be finite and non-negative) so a bad override can't invert the shaping signal or produce NaN
  targets. - test_geodesic_flag_disables_additive_approach_reward: assert the genuine-approach
  precondition (same target, distance actually reduced) instead of leaving the second
  _current_target() result unused. - test_locked_exit_hidden_in_global_map_by_default: add the
  occupancy guard and assert the isolated exit cell is fully hidden (== 0.0), so a leaked 0.2 locked
  marker or a masking objective can't make the test pass spuriously. - Add type-hint annotations to
  the new test signatures per the repo typing rule.

Tests: tests/test_crystal_caves.py + tests/test_evaluator.py green (98 passed). ruff/black/mypy
  clean.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_018jGv8TVpr6WFFbGgnZjAwk

### Features

- **crystal-caves**: Geodesic PBRS potential + locked-exit map + win-rate selection tiebreaker
  ([`b6316c2`](https://github.com/jtn0123/NN-Game1/commit/b6316c245fe82eba87302abb9ee5c376e44f18a2))

Follow-up to #34 (n-step target fix, eval hardening, PBRS stage 1). Adds the next attributable
  learning levers, both off by default so each can be A/B'd cleanly on the chain-progress surrogate
  rather than changing default behavior silently.

- Eval selection (default-on, measurement fix): restore win_rate as the dominant tiebreaker in the
  Crystal Caves continuous selection score. #34 dropped win_rate entirely, so two policies that both
  unlock the exit scored equally even if only one actually reached it. Now wins are preferred once
  they appear, while the continuous terms still expose progress before wins do.

- Geodesic PBRS potential (CRYSTAL_CAVES_GEODESIC_POTENTIAL, default off): fold a telescoping
  "closeness to the current objective" term into the shaping potential. It re-aims with the
  phase-ordered compass (switch -> crystals -> exit), is a deterministic function of state
  (PBRS-valid), has terminal Phi=0 and subtracts an initial baseline so it telescopes to exactly 0
  over a full episode (policy- invariant). It replaces the farmable additive per-step approach
  reward when on, to avoid double-counting; the stall-timer mark on genuine approach is preserved so
  the agent is never timed out mid-approach to the exit.

- Locked-exit visibility (CRYSTAL_CAVES_SHOW_LOCKED_EXIT, default off): reveal the still-locked exit
  in the coarse global objective map at a distinct lower value so the agent can learn its route
  before the last crystal unlocks it.

Tests: closeness monotonicity, geodesic-potential telescoping + terminal zero, additive-approach
  disable with stall-timer preserved, locked-exit map visibility (hidden by default / shown when
  enabled), and the win-rate selection tiebreaker.

Full suite: 1092 passed, 115 skipped (web extras). ruff/black/mypy clean on changed files.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_018jGv8TVpr6WFFbGgnZjAwk

- **crystal-caves**: Make the closeness potential a true geodesic (BFS route distance)
  ([`9552546`](https://github.com/jtn0123/NN-Game1/commit/9552546e3a81cfc488529a10792cacf960a260f4))

Addresses the remaining CodeRabbit review point on #35: the closeness term named "geodesic" was
  using straight-line np.hypot distance, which in caves with walls or locked doors rewards pushing
  toward a blocked shortcut.

_target_closeness() now reads a real route distance from a multi-source BFS field
  (_geodesic_distance_field): distance over traversable (non-solid) tiles from the active-phase
  objective tiles, honouring walls and *locked* doors via _solid_at, and re-aiming switch ->
  crystals -> exit. The field is cached and recomputed only when the objective set or open-door
  state changes, so the per-step cost stays a dict lookup; closeness is normalized by the field's
  max distance and returns 0 when the objective is unreachable from the player's tile. It remains a
  deterministic function of state, so the PBRS term is still policy-invariant.

Tests updated to derive player positions from the BFS field (guaranteeing connectivity) and to
  assert closeness rises on a genuine geodesic approach.

Full suite: 1092 passed, 115 skipped. ruff/black/mypy clean.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_018jGv8TVpr6WFFbGgnZjAwk


## v0.5.1 (2026-06-26)

### Bug Fixes

- **crystal-caves**: Correct dqn targets and eval selection
  ([`70a8737`](https://github.com/jtn0123/NN-Game1/commit/70a87373fd14b109bf2c4298b28821c2e42d2039))

### Chores

- **deps**: Bump the github-actions group across 1 directory with 2 updates
  ([`c9a5af0`](https://github.com/jtn0123/NN-Game1/commit/c9a5af02b456d7837828744b01c19a3bc2fedd3d))

Bumps the github-actions group with 2 updates in the / directory:
  [actions/checkout](https://github.com/actions/checkout) and
  [actions/dependency-review-action](https://github.com/actions/dependency-review-action).

Updates `actions/checkout` from 6 to 7 - [Release
  notes](https://github.com/actions/checkout/releases) -
  [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/actions/checkout/compare/v6...v7)

Updates `actions/dependency-review-action` from 4 to 5 - [Release
  notes](https://github.com/actions/dependency-review-action/releases) -
  [Commits](https://github.com/actions/dependency-review-action/compare/v4...v5)

--- updated-dependencies: - dependency-name: actions/checkout dependency-version: '7'

dependency-type: direct:production

update-type: version-update:semver-major

dependency-group: github-actions

- dependency-name: actions/dependency-review-action dependency-version: '5'

dependency-group: github-actions ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump the python-minor-and-patch group across 1 directory with 6 updates
  ([`d817d28`](https://github.com/jtn0123/NN-Game1/commit/d817d28e971f2b4e840fc4e49f32185fed4b75c2))

Bumps the python-minor-and-patch group with 6 updates in the / directory:

| Package | From | To | | --- | --- | --- | | [torch](https://github.com/pytorch/pytorch) | `2.12.0`
  | `2.12.1` | | [tqdm](https://github.com/tqdm/tqdm) | `4.68.1` | `4.68.3` | |
  [pytest](https://github.com/pytest-dev/pytest) | `9.0.3` | `9.1.1` | |
  [coverage](https://github.com/coveragepy/coveragepy) | `7.14.1` | `7.14.3` | |
  [python-engineio](https://github.com/miguelgrinberg/python-engineio) | `4.13.2` | `4.13.3` | |
  [python-socketio](https://github.com/miguelgrinberg/python-socketio) | `5.16.2` | `5.16.3` |

Updates `torch` from 2.12.0 to 2.12.1 - [Release notes](https://github.com/pytorch/pytorch/releases)
  - [Changelog](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) -
  [Commits](https://github.com/pytorch/pytorch/compare/v2.12.0...v2.12.1)

Updates `tqdm` from 4.68.1 to 4.68.3 - [Release notes](https://github.com/tqdm/tqdm/releases) -
  [Commits](https://github.com/tqdm/tqdm/compare/v4.68.1...v4.68.3)

Updates `pytest` from 9.0.3 to 9.1.1 - [Release
  notes](https://github.com/pytest-dev/pytest/releases) -
  [Changelog](https://github.com/pytest-dev/pytest/blob/main/CHANGELOG.rst) -
  [Commits](https://github.com/pytest-dev/pytest/compare/9.0.3...9.1.1)

Updates `coverage` from 7.14.1 to 7.14.3 - [Release
  notes](https://github.com/coveragepy/coveragepy/releases) -
  [Changelog](https://github.com/coveragepy/coveragepy/blob/main/CHANGES.rst) -
  [Commits](https://github.com/coveragepy/coveragepy/compare/7.14.1...7.14.3)

Updates `python-engineio` from 4.13.2 to 4.13.3 - [Release
  notes](https://github.com/miguelgrinberg/python-engineio/releases) -
  [Changelog](https://github.com/miguelgrinberg/python-engineio/blob/main/CHANGES.md) -
  [Commits](https://github.com/miguelgrinberg/python-engineio/compare/v4.13.2...v4.13.3)

Updates `python-socketio` from 5.16.2 to 5.16.3 - [Release
  notes](https://github.com/miguelgrinberg/python-socketio/releases) -
  [Changelog](https://github.com/miguelgrinberg/python-socketio/blob/main/CHANGES.md) -
  [Commits](https://github.com/miguelgrinberg/python-socketio/compare/v5.16.2...v5.16.3)

--- updated-dependencies: - dependency-name: torch dependency-version: 2.12.1

dependency-type: direct:production

update-type: version-update:semver-patch

dependency-group: python-minor-and-patch

- dependency-name: tqdm dependency-version: 4.68.3

- dependency-name: pytest dependency-version: 9.1.1

update-type: version-update:semver-minor

- dependency-name: coverage dependency-version: 7.14.3

- dependency-name: python-engineio dependency-version: 4.13.3

- dependency-name: python-socketio dependency-version: 5.16.3

dependency-group: python-minor-and-patch ...

Signed-off-by: dependabot[bot] <support@github.com>

### Continuous Integration

- Align release config check with checkout v7
  ([`7ce119e`](https://github.com/jtn0123/NN-Game1/commit/7ce119e747d0c4e78d4cadbd6a512135d2423bc1))


## v0.5.0 (2026-06-26)

### Features

- Add Crystal Caves NN experiment architecture
  ([`19c1394`](https://github.com/jtn0123/NN-Game1/commit/19c139443b61181c63717dd51faa8b480691f865))


## v0.4.0 (2026-06-25)

### Documentation

- Investigation handoff summary for external review
  ([`67f5068`](https://github.com/jtn0123/NN-Game1/commit/67f506850a8e2febd4bf07409f2ea5d29ab46228))

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

### Features

- Drill skill-diagnostic harness (train on drills, report per-skill mastery)
  ([`3ed9731`](https://github.com/jtn0123/NN-Game1/commit/3ed9731e8eb829c18d3fec3ad02253c8970420c2))

experiments/drill_train.py trains on the single-skill drill set and then greedily evals each drill
  on its own, printing a per-skill table (win% / crystal% / reached-exit%). A skill that stays ~0%
  even on its dedicated drill is the real wall. Doubles as motor-skill pre-training (produces a
  skill-trained policy).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Hand-authored single-skill drill levels for Crystal Caves
  ([`d1b3db5`](https://github.com/jtn0123/NN-Game1/commit/d1b3db57ff04c4e2f539a636946c5840acbad816))

A starter set of 6 tiny, deliberately-shaped teaching levels (src/game/ crystal_caves_drills.py),
  each isolating ONE motor skill: walk+collect, jump up a ledge, jump a gap, drop-and-climb-out,
  climb a staircase, and collect-then-jump-to-exit (the exact wall the agent is stuck on). True to
  1991, whose opening levels each introduce one mechanic.

Two intended uses: (1) diagnostic — run a policy on each to read its per-skill win rate instead of
  inferring the missing skill; (2) teaching — pre-train/interleave so the agent enters full levels
  already knowing these motor skills.

CRYSTAL_CAVES_DRILLS config flag loads the drill set (authored, randomized). Tests verify every
  drill is a valid 18x44 grid, solvable under the jump-aware oracle, that the jump drills genuinely
  require jumps the walk drill does not, and that the game loads them in drill mode. Full suite
  green; mypy/ruff/black clean.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

- Harden Crystal Caves NN experiment workflow
  ([`45519d5`](https://github.com/jtn0123/NN-Game1/commit/45519d55f4278d84cf186a99943a8846d25f8022))


## v0.3.1 (2026-06-21)

### Bug Fixes

- Restore N-step on vectorized path + faithful winnable-floor level design
  ([#29](https://github.com/jtn0123/NN-Game1/pull/29),
  [`adf470a`](https://github.com/jtn0123/NN-Game1/commit/adf470a51557f26975402765751aa0f76096a319))

N1: per-env N-step accumulation in NStepReplayBuffer.push_batch (was silently storing 1-step on the
  vec path the curriculum forces). L1/L3/L5: easy_open rung + walk-reachability guarantees +
  per-tier solvability gate. A/B-validated: N1 lifts training Q -0.06->-0.58; level design moves
  held-out crystals 0%->17%.


## v0.3.0 (2026-06-21)

### Features

- Reduce Crystal Caves stalls + harden deps/dashboard/eval
  ([#28](https://github.com/jtn0123/NN-Game1/pull/28),
  [`ea9a2b6`](https://github.com/jtn0123/NN-Game1/commit/ea9a2b6a0fc14002716b2c69f0da843b7fd04179))

Final-approach reward gradient fix (A/B-validated: stall rate 53%->10%) + NOISY_STD 0.7->0.5, plus
  dependency caps, dashboard emit guards, eval determinism tests, batch eval-mode hardening, and a
  dated CVE reminder.


## v0.2.0 (2026-06-21)

### Features

- Gate Crystal Caves curriculum promotion ([#27](https://github.com/jtn0123/NN-Game1/pull/27),
  [`4485221`](https://github.com/jtn0123/NN-Game1/commit/44852218eecf0977563af27a6d65c37d41aab330))

Staged Crystal Caves curriculum with held-out gated promotion, win-rate-aware keep-best, true
  eval-best rollback, guarded exploration boost, per-stage exploration anneal, and forced-vectorized
  eval so the curriculum's eval/early-stop/keep-best machinery actually runs. Validated end-to-end
  in a live run.


## v0.1.0 (2026-06-20)

### Features

- Add Crystal Caves training and dashboard telemetry
  ([`df8d395`](https://github.com/jtn0123/NN-Game1/commit/df8d3954cb4819d2c7ada6e1c8b4efd39b6eac22))

Adds Crystal Caves training/evaluation/dashboard telemetry and desktop dashboard dogfood polish.

### Refactoring

- Complete grade improvement items
  ([`67f865e`](https://github.com/jtn0123/NN-Game1/commit/67f865e7a6575ea0873c573870c8db7bdebffa53))

Squashed PR #22: complete grade-driven refactors and keep source files below 1000 LOC.

- Complete grade polish and coverage ([#23](https://github.com/jtn0123/NN-Game1/pull/23),
  [`7d0701b`](https://github.com/jtn0123/NN-Game1/commit/7d0701bc9bfaf7b7f5c5604cff5c0f465943a97e))

- Split runtime and dashboard modules ([#21](https://github.com/jtn0123/NN-Game1/pull/21),
  [`0ad6136`](https://github.com/jtn0123/NN-Game1/commit/0ad6136a8a072464edffb48d2610731e832fd43d))


## v0.0.4 (2026-06-16)

### Bug Fixes

- Harden dashboard stability
  ([`2344274`](https://github.com/jtn0123/NN-Game1/commit/23442741e3959a15d81ea56322b9452cd1ce7c32))

Harden dashboard startup and control flows, extract launcher assets, add regression coverage, and
  align dependency audit handling.


## v0.0.3 (2026-06-08)

### Bug Fixes

- Polish runtime edges and CI actions
  ([`63c50ec`](https://github.com/jtn0123/NN-Game1/commit/63c50ec89365d4e04f08e30310050c0b7f205ff3))

### Testing

- Enforce strict mypy and coverage gate
  ([`9ba130d`](https://github.com/jtn0123/NN-Game1/commit/9ba130dbba22653e1858dce4142ff8b1b1aa0b94))


## v0.0.2 (2026-06-07)

### Bug Fixes

- Polish runtime edges and CI actions
  ([`1f5e245`](https://github.com/jtn0123/NN-Game1/commit/1f5e245a301c3e9721139174906502ac4d2072b6))


## v0.0.1 (2026-06-04)

### Bug Fixes

- Address graded reliability and dashboard issues
  ([`363764b`](https://github.com/jtn0123/NN-Game1/commit/363764b8ab25e8abe0f082f76de10d181f06bb1d))

- Address pr review findings
  ([`28497af`](https://github.com/jtn0123/NN-Game1/commit/28497afaf240be23e94c40c30c4eda6a54e80582))

- Address sonar quality gate blockers
  ([`64379e0`](https://github.com/jtn0123/NN-Game1/commit/64379e0eca82607388cfff9d55466084192d6009))

- Close validated grade report items
  ([`ff2ef70`](https://github.com/jtn0123/NN-Game1/commit/ff2ef70ddfa9868e7181a78dedcea35d6d0cbcb9))

- Finish remaining grade report items
  ([`15b11d5`](https://github.com/jtn0123/NN-Game1/commit/15b11d57ab3ec18d8ac2ded91e185d5c2e50683a))

- Harden bug hunt findings
  ([`2d1c59c`](https://github.com/jtn0123/NN-Game1/commit/2d1c59ce57827a15609407a420c6e50f3b179613))

- Harden dashboard model controls
  ([`42e5718`](https://github.com/jtn0123/NN-Game1/commit/42e57186aa75aded3ef1063dc80ffa6a9aa968d2))

- Harden dashboard token exposure
  ([`8bbd74b`](https://github.com/jtn0123/NN-Game1/commit/8bbd74bb5df111b56238d54b3fdcb55df9f30199))

- Resolve model deletions from safe paths
  ([`cb77595`](https://github.com/jtn0123/NN-Game1/commit/cb775953c6763c66ebe27690cb90631c390a7280))

- Satisfy vector env contract in ci
  ([`74677af`](https://github.com/jtn0123/NN-Game1/commit/74677afef4a9dd67c7283e13b0cecd7072d10eb1))

- Update dependency floors for dependabot alerts
  ([`68893f2`](https://github.com/jtn0123/NN-Game1/commit/68893f2e8d7f3e9479282b0e1501166a60b87d18))

### Chores

- Configure coderabbit and remove sonar project config
  ([`5399146`](https://github.com/jtn0123/NN-Game1/commit/5399146a7218599bf313278a9bd6de078b6ff545))

- Fix ci type checks and update grade backlog
  ([`fef86ef`](https://github.com/jtn0123/NN-Game1/commit/fef86effa735c183d4e49fb7967c6410a097b05e))

- Harden dashboard and improve tooling
  ([`7e0e6f0`](https://github.com/jtn0123/NN-Game1/commit/7e0e6f05f2dcc907b04cab0f8abddf961cc20cdb))

- Merge main into grade fixes
  ([`5d6d70d`](https://github.com/jtn0123/NN-Game1/commit/5d6d70d74c7f370b4ac6c002f855981580a3d7a3))

### Continuous Integration

- Add CodeQL code scanning workflow
  ([`4631bc5`](https://github.com/jtn0123/NN-Game1/commit/4631bc546e46b8683c5d60d3540febc7084dfc81))

- Add dependabot configuration for automated dependency updates
  ([`54fc49a`](https://github.com/jtn0123/NN-Game1/commit/54fc49a95e3fc857476ec4b9d9b7a8a1cbb53db2))

- Add release automation and workflow hardening
  ([`b3787d9`](https://github.com/jtn0123/NN-Game1/commit/b3787d92b888cd19ca1e5a9e1931927bcfd00f34))

- Add SonarCloud configuration
  ([`33731c4`](https://github.com/jtn0123/NN-Game1/commit/33731c471761d5fa0d8e4dd495115168ebc9b608))

- Add SonarCloud scan workflow
  ([`45153be`](https://github.com/jtn0123/NN-Game1/commit/45153be6ed918b8ba559030c9c39292e9f00c788))

### Documentation

- Add tech debt analysis for current changes
  ([`0df9372`](https://github.com/jtn0123/NN-Game1/commit/0df9372f0dd47fb0ce6e51fdd03ea1e4b54f3ed3))

- Regrade current app state
  ([`b2454ec`](https://github.com/jtn0123/NN-Game1/commit/b2454ecca95197606412c040faa8832575f82c63))

### Refactoring

- Address pre-merge cleanup items
  ([`396f8ce`](https://github.com/jtn0123/NN-Game1/commit/396f8cec96b9856523994dde52fbaf2939251059))

- Close remaining grade report items
  ([`ea3b992`](https://github.com/jtn0123/NN-Game1/commit/ea3b9923ea2c8b41dfacdfb294126412b5529a5c))

### Testing

- Add cli and dashboard smoke coverage
  ([`e123c73`](https://github.com/jtn0123/NN-Game1/commit/e123c738a37b21780b0b7f316ec84468261b18d5))

- Broaden app confidence coverage
  ([`b707805`](https://github.com/jtn0123/NN-Game1/commit/b7078055b8ea72a0145f94289efde51472b0eae9))

- Strengthen dashboard and runtime coverage
  ([`49281c6`](https://github.com/jtn0123/NN-Game1/commit/49281c6e459b7271384c52e78850b4126044218e))
