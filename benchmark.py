#!/usr/bin/env python3
"""
Performance benchmark for DQN training.

Usage:
    python benchmark.py                    # Run standard benchmarks
    python benchmark.py --quick            # Quick 5-second test
    python benchmark.py --full             # Full benchmark suite
    python benchmark.py --config cpu,mps   # Test specific configs
    python benchmark.py --save results.json # Save results to file

Example output:
    Config: CPU B=128 LE=8 GS=2
    Steps: 15,234 | Time: 3.00s | Steps/sec: 5,078
    Gradient updates: 1,904 | Grad/sec: 635
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    config_name: str
    device: str
    batch_size: int
    learn_every: int
    gradient_steps: int
    total_steps: int
    total_time: float
    steps_per_sec: float
    gradient_updates: int
    gradients_per_sec: float
    episodes_completed: int


def run_benchmark(
    config_name: str,
    device: str,
    batch_size: int,
    learn_every: int,
    gradient_steps: int,
    duration: float = 3.0,
    warmup: float = 0.5,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    # Import here to avoid pygame init on module load
    from config import Config
    from src.game.breakout import Breakout
    from src.ai.agent import Agent
    
    # Setup config
    config = Config()
    config.BATCH_SIZE = batch_size
    config.LEARN_EVERY = learn_every
    config.GRADIENT_STEPS = gradient_steps
    config.USE_TORCH_COMPILE = False  # Skip for benchmarking consistency
    config.USE_MIXED_PRECISION = False
    
    if device == 'cpu':
        config.FORCE_CPU = True
    else:
        config.FORCE_CPU = False
        # Check if requested device is available
        if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print(f"  ⚠️  MPS not available, skipping")
            return None
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"  ⚠️  CUDA not available, skipping")
            return None
    
    # Create game and agent
    game = Breakout(config, headless=True)
    agent = Agent(game.state_size, game.action_size, config)
    
    # Fill replay buffer (required for learning)
    print(f"  Filling replay buffer...", end=" ", flush=True)
    state = game.reset()
    for _ in range(config.BATCH_SIZE * 2):
        action = agent.select_action(state)
        next_state, reward, done, _ = game.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state if not done else game.reset()
    print("done")
    
    # Warmup phase
    print(f"  Warming up ({warmup}s)...", end=" ", flush=True)
    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < warmup:
        action = agent.select_action(state)
        next_state, reward, done, _ = game.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        state = next_state if not done else game.reset()
    print("done")
    
    # Reset counters after warmup
    agent.steps = 0
    agent._learn_step = 0
    episodes = 0
    
    # Benchmark phase
    print(f"  Benchmarking ({duration}s)...", end=" ", flush=True)
    total_steps = 0
    start_time = time.perf_counter()
    
    while time.perf_counter() - start_time < duration:
        action = agent.select_action(state)
        next_state, reward, done, _ = game.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        total_steps += 1
        
        if done:
            state = game.reset()
            episodes += 1
        else:
            state = next_state
    
    elapsed = time.perf_counter() - start_time
    print("done")
    
    # Calculate metrics
    steps_per_sec = total_steps / elapsed
    gradient_updates = agent.steps  # Total gradient steps performed
    gradients_per_sec = gradient_updates / elapsed
    
    return BenchmarkResult(
        config_name=config_name,
        device=str(config.DEVICE),
        batch_size=batch_size,
        learn_every=learn_every,
        gradient_steps=gradient_steps,
        total_steps=total_steps,
        total_time=round(elapsed, 2),
        steps_per_sec=round(steps_per_sec, 1),
        gradient_updates=gradient_updates,
        gradients_per_sec=round(gradients_per_sec, 1),
        episodes_completed=episodes,
    )


# Predefined configurations to test
CONFIGS = {
    # Recommended turbo settings
    'turbo': ('cpu', 128, 8, 2),
    
    # CPU variations
    'cpu_b64': ('cpu', 64, 4, 1),
    'cpu_b128': ('cpu', 128, 4, 1),
    'cpu_b256': ('cpu', 256, 4, 1),
    'cpu_b128_le8': ('cpu', 128, 8, 2),
    'cpu_b128_le16': ('cpu', 128, 16, 4),
    
    # MPS variations (Apple Silicon GPU)
    'mps_b128': ('mps', 128, 4, 1),
    'mps_b256': ('mps', 256, 4, 1),
    'mps_b512': ('mps', 512, 4, 1),
    
    # CUDA variations (NVIDIA GPU)
    'cuda_b256': ('cuda', 256, 4, 1),
    'cuda_b512': ('cuda', 512, 4, 1),
}

# Quick test configs
QUICK_CONFIGS = ['turbo', 'cpu_b128', 'mps_b256']

# Standard test configs
STANDARD_CONFIGS = ['turbo', 'cpu_b128', 'cpu_b256', 'mps_b256']

# Full test configs
FULL_CONFIGS = list(CONFIGS.keys())


def print_result(result: BenchmarkResult) -> None:
    """Pretty print a benchmark result."""
    print(f"\n  📊 {result.config_name}")
    print(f"     Device: {result.device} | Batch: {result.batch_size} | LE: {result.learn_every} | GS: {result.gradient_steps}")
    print(f"     Steps: {result.total_steps:,} in {result.total_time}s → {result.steps_per_sec:,.0f} steps/sec")
    print(f"     Gradients: {result.gradient_updates:,} → {result.gradients_per_sec:,.0f} grad/sec")


def main():
    parser = argparse.ArgumentParser(description='DQN Training Performance Benchmark')
    parser.add_argument('--quick', action='store_true', help='Quick 5-second benchmarks')
    parser.add_argument('--full', action='store_true', help='Full benchmark suite (all configs)')
    parser.add_argument('--duration', type=float, default=3.0, help='Benchmark duration per config (seconds)')
    parser.add_argument('--config', type=str, help='Comma-separated config names to test')
    parser.add_argument('--save', type=str, help='Save results to JSON file')
    parser.add_argument('--list', action='store_true', help='List available configs')
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable benchmark configurations:")
        for name, (device, batch, le, gs) in CONFIGS.items():
            print(f"  {name}: device={device}, batch={batch}, learn_every={le}, grad_steps={gs}")
        return
    
    # Determine which configs to run
    if args.config:
        config_names = [c.strip() for c in args.config.split(',')]
        for name in config_names:
            if name not in CONFIGS:
                print(f"Unknown config: {name}")
                print(f"Available: {', '.join(CONFIGS.keys())}")
                return
    elif args.full:
        config_names = FULL_CONFIGS
    elif args.quick:
        config_names = QUICK_CONFIGS
    else:
        config_names = STANDARD_CONFIGS
    
    duration = 2.0 if args.quick else args.duration
    
    print("=" * 60)
    print("DQN Training Performance Benchmark")
    print("=" * 60)
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"\nConfigs to test: {', '.join(config_names)}")
    print(f"Duration per config: {duration}s")
    
    results = []
    
    for config_name in config_names:
        device, batch_size, learn_every, gradient_steps = CONFIGS[config_name]
        print(f"\n{'─' * 50}")
        print(f"Running: {config_name}")
        
        try:
            result = run_benchmark(
                config_name=config_name,
                device=device,
                batch_size=batch_size,
                learn_every=learn_every,
                gradient_steps=gradient_steps,
                duration=duration,
            )
            if result:
                results.append(result)
                print_result(result)
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    if results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print("=" * 60)
        
        # Sort by steps/sec
        results.sort(key=lambda r: r.steps_per_sec, reverse=True)
        
        print(f"\n{'Config':<20} {'Device':<8} {'Steps/sec':>12} {'Grad/sec':>12}")
        print("-" * 54)
        for r in results:
            print(f"{r.config_name:<20} {r.device:<8} {r.steps_per_sec:>12,.0f} {r.gradients_per_sec:>12,.0f}")
        
        best = results[0]
        print(f"\n🏆 Fastest: {best.config_name} ({best.steps_per_sec:,.0f} steps/sec)")
    
    # Save results
    if args.save and results:
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'results': [asdict(r) for r in results],
        }
        with open(args.save, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to {args.save}")


if __name__ == '__main__':
    main()

