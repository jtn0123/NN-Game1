#!/usr/bin/env python3
"""
Parallel Experiment Runner
==========================

Runs multiple training variants in parallel to find optimal hyperparameters.
Each variant tests ONE different tweak to isolate what helps.

Usage:
    python experiment_runner.py --experiments 8 --episodes 2000
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment variant."""
    name: str
    description: str
    config_overrides: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# EXPERIMENT DEFINITIONS - Each tests ONE variable
# ============================================================================

EXPERIMENTS_ROUND_1 = [
    ExperimentConfig(
        name="baseline",
        description="Current settings (control group)",
        config_overrides={}
    ),
    ExperimentConfig(
        name="high_shoot_reward",
        description="Higher shoot reward: +0.05 (was +0.02)",
        config_overrides={"SI_REWARD_SHOOT": 0.05}
    ),
    ExperimentConfig(
        name="high_stay_penalty",
        description="Higher stay penalty: -0.05 (was -0.02)",
        config_overrides={"SI_REWARD_STAY": -0.05}
    ),
    ExperimentConfig(
        name="strong_step_penalty",
        description="Stronger step penalty: -0.02 (was -0.01)",
        config_overrides={"SI_REWARD_STEP": -0.02}
    ),
    ExperimentConfig(
        name="low_death_penalty",
        description="Lower death penalty: -1.0 (was -2.5)",
        config_overrides={"SI_REWARD_PLAYER_DEATH": -1.0}
    ),
    ExperimentConfig(
        name="aggressive_combo",
        description="Combo: high shoot +0.05, high stay penalty -0.05",
        config_overrides={"SI_REWARD_SHOOT": 0.05, "SI_REWARD_STAY": -0.05}
    ),
    ExperimentConfig(
        name="original_arch",
        description="Original architecture [512, 256, 128]",
        config_overrides={"HIDDEN_LAYERS": [512, 256, 128]}
    ),
    ExperimentConfig(
        name="high_lr",
        description="Higher learning rate: 0.0003 (was 0.0001)",
        config_overrides={"LEARNING_RATE": 0.0003}
    ),
]


def create_experiment_script(
    experiment: ExperimentConfig,
    episodes: int,
    envs_per_experiment: int,
    output_dir: str
) -> str:
    """Create a Python script for a single experiment."""
    
    script = f'''#!/usr/bin/env python3
"""Auto-generated experiment: {experiment.name}"""
import os
import sys
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Add project root to path
sys.path.insert(0, '{os.getcwd()}')

import json
import time
from datetime import datetime

from config import Config
from src.game import get_game
from src.ai.agent import Agent
from src.ai.evaluator import Evaluator

# Create config with overrides
config = Config()
overrides = {json.dumps(experiment.config_overrides)}

for key, value in overrides.items():
    if hasattr(config, key):
        setattr(config, key, value)
        print(f"Override: {{key}} = {{value}}")

# Force Space Invaders game
config.GAME_NAME = 'space_invaders'

# Reduce envs for parallel experiments
config.NUM_ENVS = {envs_per_experiment}
config.EVAL_EVERY = 500  # Eval every 500 episodes

# Setup paths
experiment_name = "{experiment.name}"
output_dir = "{output_dir}"
model_dir = os.path.join(output_dir, experiment_name, "models")
log_dir = os.path.join(output_dir, experiment_name, "logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Initialize game and agent
GameClass = get_game(config.GAME_NAME)
temp_game = GameClass(config, headless=True)
state_size = temp_game.state_size
action_size = temp_game.action_size

agent = Agent(state_size, action_size, config)

# Setup evaluator
eval_game = GameClass(config, headless=True)
evaluator = Evaluator(eval_game, agent, config, log_dir=log_dir)

# Results tracking
results = {{
    "experiment": experiment_name,
    "description": "{experiment.description}",
    "overrides": overrides,
    "episodes": {episodes},
    "start_time": datetime.now().isoformat(),
    "eval_history": [],
    "final_eval": None
}}

print(f"\\n{'='*60}")
print(f"EXPERIMENT: {{experiment_name}}")
print(f"{{'{experiment.description}'}}")
print(f"{'='*60}\\n")

# Import vectorized environment
from src.game.space_invaders import VecSpaceInvaders

# Training loop
vec_env = VecSpaceInvaders({envs_per_experiment}, config, headless=True)
states = vec_env.reset()

episode = 0
total_steps = 0
episode_rewards = []
best_eval_score = 0

start_time = time.time()

while episode < {episodes}:
    # Select actions
    actions, explored, exploited = agent.select_actions_batch(states, training=True)
    
    # Step environment
    next_states, rewards, dones, infos = vec_env.step(actions)
    
    # Store experiences
    for i in range(len(states)):
        agent.memory.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    # Learn
    if len(agent.memory) >= config.BATCH_SIZE:
        for _ in range(config.GRADIENT_STEPS):
            agent.learn()
    
    total_steps += len(states)
    
    # Handle done episodes
    for i, done in enumerate(dones):
        if done:
            episode += 1
            score = infos[i].get('score', 0)
            episode_rewards.append(score)
            
            # Log every 100 episodes
            if episode % 100 == 0:
                avg = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
                elapsed = time.time() - start_time
                steps_per_sec = total_steps / elapsed
                print(f"[{{experiment_name}}] Ep {{episode:4d}} | Avg: {{avg:.0f}} | ε: {{agent.epsilon:.3f}} | {{steps_per_sec:.0f}} steps/s")
            
            # Periodic evaluation
            if episode % 500 == 0 and episode > 0:
                eval_results = evaluator.evaluate(num_episodes=30, max_steps=5000, episode_num=episode)
                eval_score = eval_results.mean_score
                results["eval_history"].append({{
                    "episode": episode,
                    "eval_score": eval_score,
                    "max_level": eval_results.max_level
                }})
                print(f"[{{experiment_name}}] EVAL @ {{episode}}: {{eval_score:.0f}} avg, max level {{eval_results.max_level}}")
                
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    # Save best model
                    agent.save(os.path.join(model_dir, f"{{experiment_name}}_best.pth"))
            
            # Reset done env
            vec_env.reset_single(i)
    
    states = next_states
    
    # Decay epsilon
    agent.epsilon = max(config.EPSILON_END, agent.epsilon * config.EPSILON_DECAY)

# Final evaluation
print(f"\\n[{{experiment_name}}] Running final evaluation...")
final_eval = evaluator.evaluate(num_episodes=50, max_steps=5000, episode_num=episode)
results["final_eval"] = {{
    "mean_score": final_eval.mean_score,
    "max_score": final_eval.max_score,
    "max_level": final_eval.max_level,
    "win_rate": final_eval.win_rate
}}

results["end_time"] = datetime.now().isoformat()
results["total_time_seconds"] = time.time() - start_time

# Save results
results_path = os.path.join(output_dir, experiment_name, "results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\\n{'='*60}")
print(f"EXPERIMENT COMPLETE: {{experiment_name}}")
print(f"Final Eval: {{final_eval.mean_score:.0f}} avg, max level {{final_eval.max_level}}")
print(f"Results saved to: {{results_path}}")
print(f"{'='*60}\\n")
'''
    return script


def run_experiments(
    experiments: List[ExperimentConfig],
    episodes: int = 2000,
    envs_per_experiment: int = 8,
    max_parallel: int = 4
) -> None:
    """Run experiments in parallel batches."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiments/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PARALLEL EXPERIMENT RUNNER")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Experiments: {len(experiments)}")
    print(f"Episodes per experiment: {episodes}")
    print(f"Environments per experiment: {envs_per_experiment}")
    print(f"Max parallel: {max_parallel}")
    print(f"{'='*70}\n")
    
    # Save experiment manifest
    manifest = {
        "timestamp": timestamp,
        "experiments": [e.to_dict() for e in experiments],
        "episodes": episodes,
        "envs_per_experiment": envs_per_experiment
    }
    with open(os.path.join(output_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create and run experiment scripts
    processes: List[subprocess.Popen] = []
    scripts: List[str] = []
    
    for exp in experiments:
        script_content = create_experiment_script(exp, episodes, envs_per_experiment, output_dir)
        script_path = os.path.join(output_dir, exp.name, f"{exp.name}_run.py")
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        scripts.append(script_path)
    
    print(f"Created {len(scripts)} experiment scripts\n")
    print("Starting experiments...\n")
    
    # Run in batches
    for i in range(0, len(scripts), max_parallel):
        batch = scripts[i:i + max_parallel]
        batch_procs = []
        
        print(f"--- Batch {i // max_parallel + 1} ({len(batch)} experiments) ---")
        
        for script_path in batch:
            exp_name = os.path.basename(os.path.dirname(script_path))
            log_path = os.path.join(os.path.dirname(script_path), "output.log")
            
            with open(log_path, 'w') as log_file:
                proc = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd()
                )
                batch_procs.append((exp_name, proc, log_path))
                print(f"  Started: {exp_name} (PID: {proc.pid})")
        
        # Wait for batch to complete
        print(f"\nWaiting for batch to complete...")
        for exp_name, proc, log_path in batch_procs:
            proc.wait()
            status = "✓" if proc.returncode == 0 else "✗"
            print(f"  {status} {exp_name} (exit code: {proc.returncode})")
        
        print()
    
    # Summarize results
    print(f"\n{'='*70}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    results_summary = []
    for exp in experiments:
        results_path = os.path.join(output_dir, exp.name, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
                if results.get("final_eval"):
                    results_summary.append({
                        "name": exp.name,
                        "description": exp.description,
                        "final_score": results["final_eval"]["mean_score"],
                        "max_level": results["final_eval"]["max_level"],
                        "win_rate": results["final_eval"]["win_rate"]
                    })
    
    # Sort by final score
    results_summary.sort(key=lambda x: x["final_score"], reverse=True)
    
    print(f"{'Rank':<5} {'Experiment':<25} {'Score':<10} {'Max Lvl':<10} {'Description'}")
    print("-" * 80)
    for i, r in enumerate(results_summary, 1):
        print(f"{i:<5} {r['name']:<25} {r['final_score']:<10.0f} {r['max_level']:<10} {r['description'][:30]}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Full results saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel hyperparameter experiments")
    parser.add_argument("--episodes", type=int, default=2000, help="Episodes per experiment")
    parser.add_argument("--envs", type=int, default=8, help="Environments per experiment")
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel experiments")
    parser.add_argument("--experiments", type=str, default="round1", 
                        help="Experiment set to run (round1)")
    
    args = parser.parse_args()
    
    if args.experiments == "round1":
        experiments = EXPERIMENTS_ROUND_1
    else:
        print(f"Unknown experiment set: {args.experiments}")
        sys.exit(1)
    
    run_experiments(
        experiments=experiments,
        episodes=args.episodes,
        envs_per_experiment=args.envs,
        max_parallel=args.parallel
    )

