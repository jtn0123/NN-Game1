#!/usr/bin/env python3
"""
Checkpoint Evaluation Script
============================
Runs deterministic evaluation (ε=0) on saved checkpoints to measure true performance.

Usage:
    python evaluate_checkpoints.py                    # Evaluate all checkpoints
    python evaluate_checkpoints.py ep29300 ep25300   # Evaluate specific checkpoints
    python evaluate_checkpoints.py --episodes 50     # Run 50 eval episodes each
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.game.space_invaders import SpaceInvaders
from src.ai.agent import Agent


def evaluate_checkpoint(checkpoint_path: str, num_episodes: int = 100, seed: int = 42) -> dict:
    """
    Run deterministic evaluation on a checkpoint.
    
    Args:
        checkpoint_path: Path to .pth file
        num_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with evaluation statistics
    """
    # Setup
    config = Config()
    config.GAME_NAME = 'space_invaders'
    
    # Set seeds for reproducibility
    np.random.seed(seed)
    
    # Create game and agent
    game = SpaceInvaders(config)
    agent = Agent(
        state_size=game.state_size,
        action_size=game.action_size,
        config=config
    )
    
    # Load checkpoint
    try:
        agent.load(checkpoint_path)
    except Exception as e:
        return {'error': str(e), 'checkpoint': checkpoint_path}
    
    # Force epsilon to 0 for deterministic evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    # Run evaluation episodes
    scores = []
    levels_reached = []
    steps_per_episode = []
    wins = 0
    
    for ep in range(num_episodes):
        state = game.reset()
        done = False
        steps = 0
        
        while not done and steps < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state, training=False)
            state, _, done, info = game.step(action)
            steps += 1
        
        score = info.get('score', 0)
        level = info.get('level', 1)
        won = info.get('won', False)
        
        scores.append(score)
        levels_reached.append(level)
        steps_per_episode.append(steps)
        if won:
            wins += 1
        
        # Progress indicator
        if (ep + 1) % 10 == 0:
            print(f"  Evaluated {ep + 1}/{num_episodes} episodes...", end='\r')
    
    print()  # Clear progress line
    
    # Calculate statistics
    scores_arr = np.array(scores)
    levels_arr = np.array(levels_reached)
    
    results = {
        'checkpoint': os.path.basename(checkpoint_path),
        'episodes': num_episodes,
        'seed': seed,
        'scores': {
            'mean': float(np.mean(scores_arr)),
            'std': float(np.std(scores_arr)),
            'median': float(np.median(scores_arr)),
            'min': int(np.min(scores_arr)),
            'max': int(np.max(scores_arr)),
            'q25': float(np.percentile(scores_arr, 25)),
            'q75': float(np.percentile(scores_arr, 75)),
        },
        'levels': {
            'mean': float(np.mean(levels_arr)),
            'max': int(np.max(levels_arr)),
            'distribution': {str(i): int((levels_arr == i).sum()) for i in range(1, 11)}
        },
        'win_rate': wins / num_episodes,
        'wins': wins,
        'avg_steps': float(np.mean(steps_per_episode)),
    }
    
    return results


def format_results(results: dict) -> str:
    """Format results for console output."""
    if 'error' in results:
        return f"❌ {results['checkpoint']}: {results['error']}"
    
    s = results['scores']
    return (
        f"📊 {results['checkpoint']}\n"
        f"   Score: {s['mean']:.1f} ± {s['std']:.1f} (median: {s['median']:.1f})\n"
        f"   Range: {s['min']} - {s['max']} (IQR: {s['q25']:.0f}-{s['q75']:.0f})\n"
        f"   Max Level: {results['levels']['max']} | Avg Level: {results['levels']['mean']:.2f}\n"
        f"   Win Rate: {results['win_rate']*100:.1f}% ({results['wins']}/{results['episodes']})\n"
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate Space Invaders checkpoints')
    parser.add_argument('checkpoints', nargs='*', help='Specific checkpoints to evaluate (e.g., ep29300)')
    parser.add_argument('--episodes', '-n', type=int, default=100, help='Episodes per checkpoint')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    args = parser.parse_args()
    
    # Find checkpoints
    model_dir = 'models/space_invaders'
    
    if args.checkpoints:
        # Specific checkpoints requested
        checkpoint_files = []
        for cp in args.checkpoints:
            if not cp.endswith('.pth'):
                cp = f'space_invaders_{cp}.pth'
            path = os.path.join(model_dir, cp)
            if os.path.exists(path):
                checkpoint_files.append(path)
            else:
                print(f"⚠️  Checkpoint not found: {path}")
    else:
        # Find all checkpoints
        checkpoint_files = sorted(glob.glob(os.path.join(model_dir, 'space_invaders_ep*.pth')))
        # Also check for best/interrupted
        for name in ['space_invaders_best.pth', 'space_invaders_interrupted.pth']:
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                checkpoint_files.append(path)
    
    if not checkpoint_files:
        print("❌ No checkpoints found!")
        return 1
    
    print("=" * 70)
    print("🎮 CHECKPOINT EVALUATION - Space Invaders")
    print("=" * 70)
    print(f"   Episodes per checkpoint: {args.episodes}")
    print(f"   Random seed: {args.seed}")
    print(f"   Checkpoints to evaluate: {len(checkpoint_files)}")
    print("=" * 70)
    
    all_results = []
    
    for i, cp_path in enumerate(checkpoint_files):
        print(f"\n[{i+1}/{len(checkpoint_files)}] Evaluating {os.path.basename(cp_path)}...")
        results = evaluate_checkpoint(cp_path, args.episodes, args.seed)
        all_results.append(results)
        print(format_results(results))
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("📈 SUMMARY COMPARISON")
    print("=" * 70)
    
    valid_results = [r for r in all_results if 'error' not in r]
    if len(valid_results) >= 2:
        # Sort by checkpoint name (episode number)
        valid_results.sort(key=lambda x: x['checkpoint'])
        
        print(f"\n{'Checkpoint':<35} {'Mean':>8} {'Median':>8} {'Max':>6} {'Win%':>6}")
        print("-" * 70)
        for r in valid_results:
            s = r['scores']
            print(f"{r['checkpoint']:<35} {s['mean']:>8.1f} {s['median']:>8.1f} {s['max']:>6} {r['win_rate']*100:>5.1f}%")
        
        # Statistical comparison: first vs last
        first = valid_results[0]['scores']
        last = valid_results[-1]['scores']
        improvement = last['mean'] - first['mean']
        print(f"\n📊 Improvement (first→last): {improvement:+.1f} points ({improvement/first['mean']*100:+.1f}%)")
    
    # Save results if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'episodes_per_checkpoint': args.episodes,
                'seed': args.seed,
            },
            'results': all_results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
