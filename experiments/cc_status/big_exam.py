"""Large-sample official-rules exam of a saved policy checkpoint on one level.

This is the campaign's champion-verification protocol (see CAMPAIGN_LOG.md):
the training harness's 9-episode checkpoint evals cannot resolve a 10-40%
win rate, so promotion decisions are made ONLY from 100+ episode exams run
with this script. Two protocol rules, both learned the hard way:

- The full 16-level eval context must be kept (state includes the level
  index; pinning the game to a single level silently breaks the policy).
  We keep the eval-level set intact and pin only the eval cursor.
- "noisy" mode (NoisyNet sampling, as trained) is the policy as defined;
  "det" disables exploration noise for a greedy read.

Usage:
    python experiments/cc_status/big_exam.py CKPT LEVEL [N] [MODE] [MAX_STEPS]

    CKPT       path to a policy state_dict (.pth)
    LEVEL      eval level index (e.g. 14 = The Switchback Spire)
    N          episode count (default 100)
    MODE       "noisy" (default) or "det"
    MAX_STEPS  episode cap (default 4500, the fidelity clock; 3000 = 1991 rules)

Example (the banked champion):
    python experiments/cc_status/big_exam.py \\
        experiments/cc_status/data/champions/switchback_champion_run39e_ep10000.pth \\
        14 100 noisy 4500
"""

import sys

import torch

sys.path.insert(0, ".")

from experiments.cc_status.lever_ab import make_config  # noqa: E402
from experiments.cc_status.training import prepare_trainer  # noqa: E402


def main() -> None:
    ckpt = sys.argv[1]
    level = int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    mode = sys.argv[4] if len(sys.argv) > 4 else "noisy"
    max_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 4500

    overrides = {
        "CRYSTAL_CAVES_GEO_COMPASS": True,
        "CRYSTAL_CAVES_GEO_COMPASS_HAZARD_AWARE": True,
        "REWARD_CLIP": 35.0,
        "CRYSTAL_CAVES_STALL_WINDOW_STEPS": 1440,
        "CRYSTAL_CAVES_NGU_BONUS": True,
        "CRYSTAL_CAVES_ENEMY_MOTION": True,
        "CRYSTAL_CAVES_MAX_STEPS_OVERRIDE": max_steps,
        "FORCE_CPU": True,
    }
    config = make_config(overrides, difficulty="tutorial", imported=True)
    trainer = prepare_trainer(config, episodes=1, vec_envs=1)
    sd = torch.load(ckpt, map_location="cpu")
    trainer.agent.policy_net.load_state_dict(sd)
    trainer.agent.epsilon = 0.0
    if mode == "det":
        trainer.agent.policy_net.eval()

    game = trainer.game if hasattr(trainer, "game") else trainer.games[0]
    game.use_eval_levels(len(game.CAVES) if not game._eval_caves else len(game._eval_caves))

    wins = 0
    best = 0.0
    for ep in range(n):
        game._eval_cursor = level
        state = game.reset()
        assert game.level_index == level
        done = False
        while not done:
            action = trainer.agent.select_action(state, training=(mode != "det"))
            state, _r, done, _info = game.step(int(action))
        frac = 1.0 - len(game.crystals) / max(1, game.initial_crystals)
        best = max(best, frac)
        if game.won:
            wins += 1
            print(f"  WIN on episode {ep} (steps={game.steps})", flush=True)
        if (ep + 1) % 25 == 0:
            print(f"  {ep + 1}/{n}: wins={wins} best_frac={best:.3f}", flush=True)
    print(f"FINAL: {wins}/{n} official wins | best crystal fraction {best:.3f}")


if __name__ == "__main__":
    main()
