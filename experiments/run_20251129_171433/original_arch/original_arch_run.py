#!/usr/bin/env python3
"""Auto-generated experiment: original_arch"""
import os
import sys
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Add project root to path
sys.path.insert(0, '/Users/justin/Documents/Github/NN-Game1')

import json
import time
from datetime import datetime

from config import Config
from src.game import get_game
from src.ai.agent import Agent
from src.ai.evaluator import Evaluator

# Create config with overrides
config = Config()
overrides = {"HIDDEN_LAYERS": [512, 256, 128]}

for key, value in overrides.items():
    if hasattr(config, key):
        setattr(config, key, value)
        print(f"Override: {key} = {value}")

# Reduce envs for parallel experiments
config.NUM_ENVS = 8
config.EVAL_EVERY = 500  # Eval every 500 episodes

# Setup paths
experiment_name = "original_arch"
output_dir = "experiments/run_20251129_171433"
model_dir = os.path.join(output_dir, experiment_name, "models")
log_dir = os.path.join(output_dir, experiment_name, "logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Initialize game and agent
game_class = get_game(config.GAME_NAME, config, headless=True)
if hasattr(game_class, 'state_size'):
    state_size = game_class.state_size
    action_size = game_class.action_size
else:
    # It's a class, instantiate it
    temp_game = game_class
    state_size = temp_game.state_size
    action_size = temp_game.action_size
    game_class = temp_game

agent = Agent(state_size, action_size, config)

# Setup evaluator
eval_game = get_game(config.GAME_NAME, config, headless=True)
evaluator = Evaluator(eval_game, agent, config, log_dir=log_dir)

# Results tracking
results = {
    "experiment": experiment_name,
    "description": "Original architecture [512, 256, 128]",
    "overrides": overrides,
    "episodes": 1500,
    "start_time": datetime.now().isoformat(),
    "eval_history": [],
    "final_eval": None
}

print(f"\n============================================================")
print(f"EXPERIMENT: {experiment_name}")
print(f"{'Original architecture [512, 256, 128]'}")
print(f"============================================================\n")

# Import vectorized environment
from src.game.space_invaders import VecSpaceInvaders

# Training loop
vec_env = VecSpaceInvaders(8, config, headless=True)
states = vec_env.reset()

episode = 0
total_steps = 0
episode_rewards = []
best_eval_score = 0

start_time = time.time()

while episode < 1500:
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
                print(f"[{experiment_name}] Ep {episode:4d} | Avg: {avg:.0f} | ε: {agent.epsilon:.3f} | {steps_per_sec:.0f} steps/s")
            
            # Periodic evaluation
            if episode % 500 == 0 and episode > 0:
                eval_results = evaluator.evaluate(num_episodes=30, max_steps=5000, episode_num=episode)
                eval_score = eval_results.mean_score
                results["eval_history"].append({
                    "episode": episode,
                    "eval_score": eval_score,
                    "max_level": eval_results.max_level
                })
                print(f"[{experiment_name}] EVAL @ {episode}: {eval_score:.0f} avg, max level {eval_results.max_level}")
                
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    # Save best model
                    agent.save(os.path.join(model_dir, f"{experiment_name}_best.pth"))
            
            # Reset done env
            vec_env.reset_single(i)
    
    states = next_states
    
    # Decay epsilon
    agent.epsilon = max(config.EPSILON_END, agent.epsilon * config.EPSILON_DECAY)

# Final evaluation
print(f"\n[{experiment_name}] Running final evaluation...")
final_eval = evaluator.evaluate(num_episodes=50, max_steps=5000, episode_num=episode)
results["final_eval"] = {
    "mean_score": final_eval.mean_score,
    "max_score": final_eval.max_score,
    "max_level": final_eval.max_level,
    "win_rate": final_eval.win_rate
}

results["end_time"] = datetime.now().isoformat()
results["total_time_seconds"] = time.time() - start_time

# Save results
results_path = os.path.join(output_dir, experiment_name, "results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n============================================================")
print(f"EXPERIMENT COMPLETE: {experiment_name}")
print(f"Final Eval: {final_eval.mean_score:.0f} avg, max level {final_eval.max_level}")
print(f"Results saved to: {results_path}")
print(f"============================================================\n")
