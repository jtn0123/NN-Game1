#!/usr/bin/env python3
"""
Performance benchmark for DQN training.

Usage:
    python benchmark.py                    # Run standard benchmarks
    python benchmark.py --quick            # Quick 3-second test
    python benchmark.py --full             # Full benchmark suite
    python benchmark.py --realistic        # Test with realistic 50k buffer
    python benchmark.py --config cpu,mps   # Test specific configs
    python benchmark.py --save results.json # Save results to file

Example output:
    Config: CPU B=128 LE=8 GS=2 (buffer: 50,000)
    Steps: 15,234 | Time: 3.00s | Steps/sec: 5,078
    Gradient updates: 1,904 | Grad/sec: 635

Notes:
    - Buffer size significantly affects performance due to sampling algorithms
    - Small buffers (<3k): np.random.choice is faster
    - Large buffers (>3k): random.sample is ~30x faster
    - Use --realistic for production-like benchmarks
"""

import argparse
import json
import time
import platform
import sys
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Tuple

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
    buffer_size: int  # NEW: Track buffer size
    total_steps: int
    total_time: float
    steps_per_sec: float
    gradient_updates: int
    gradients_per_sec: float
    episodes_completed: int
    # NEW: Additional metrics for analysis
    avg_episode_length: float = 0.0
    memory_mb: float = 0.0


@dataclass  
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    device: str
    batch_size: int
    learn_every: int
    gradient_steps: int
    buffer_fill: int = 0  # 0 = minimal (batch_size * 2), >0 = fill to this size


# Predefined configurations
CONFIGS = {
    # === QUICK TESTS (small buffer) ===
    # These test raw computation speed with minimal buffer overhead
    'turbo': BenchmarkConfig('turbo', 'cpu', 128, 8, 2),
    'cpu_b64': BenchmarkConfig('cpu_b64', 'cpu', 64, 4, 1),
    'cpu_b128': BenchmarkConfig('cpu_b128', 'cpu', 128, 4, 1),
    'cpu_b256': BenchmarkConfig('cpu_b256', 'cpu', 256, 4, 1),
    'cpu_b128_le8': BenchmarkConfig('cpu_b128_le8', 'cpu', 128, 8, 2),
    'cpu_b128_le16': BenchmarkConfig('cpu_b128_le16', 'cpu', 128, 16, 4),
    
    # MPS variations (Apple Silicon GPU)
    'mps_b128': BenchmarkConfig('mps_b128', 'mps', 128, 4, 1),
    'mps_b256': BenchmarkConfig('mps_b256', 'mps', 256, 4, 1),
    'mps_b512': BenchmarkConfig('mps_b512', 'mps', 512, 4, 1),
    
    # CUDA variations (NVIDIA GPU)
    'cuda_b256': BenchmarkConfig('cuda_b256', 'cuda', 256, 4, 1),
    'cuda_b512': BenchmarkConfig('cuda_b512', 'cuda', 512, 4, 1),
    
    # === REALISTIC TESTS (large buffer) ===
    # These simulate actual training with full replay buffer
    'realistic_cpu': BenchmarkConfig('realistic_cpu', 'cpu', 128, 4, 1, buffer_fill=50000),
    'realistic_turbo': BenchmarkConfig('realistic_turbo', 'cpu', 128, 8, 2, buffer_fill=50000),
    'realistic_mps': BenchmarkConfig('realistic_mps', 'mps', 256, 4, 1, buffer_fill=50000),
    
    # === SCALING TESTS (vary buffer size) ===
    # These test how performance scales with buffer size
    'scale_1k': BenchmarkConfig('scale_1k', 'cpu', 128, 4, 1, buffer_fill=1000),
    'scale_5k': BenchmarkConfig('scale_5k', 'cpu', 128, 4, 1, buffer_fill=5000),
    'scale_10k': BenchmarkConfig('scale_10k', 'cpu', 128, 4, 1, buffer_fill=10000),
    'scale_50k': BenchmarkConfig('scale_50k', 'cpu', 128, 4, 1, buffer_fill=50000),
    'scale_100k': BenchmarkConfig('scale_100k', 'cpu', 128, 4, 1, buffer_fill=100000),
}

# Config groups
QUICK_CONFIGS = ['turbo', 'cpu_b128', 'mps_b256']
STANDARD_CONFIGS = ['turbo', 'cpu_b128', 'cpu_b256', 'mps_b256']
REALISTIC_CONFIGS = ['realistic_cpu', 'realistic_turbo', 'realistic_mps']
SCALING_CONFIGS = ['scale_1k', 'scale_5k', 'scale_10k', 'scale_50k', 'scale_100k']
FULL_CONFIGS = list(CONFIGS.keys())


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024 / 1024  # Convert to MB (macOS reports in bytes)
    except ImportError:
        return 0.0


def run_benchmark(
    cfg: BenchmarkConfig,
    duration: float = 3.0,
    warmup: float = 0.5,
    verbose: bool = True,
) -> Optional[BenchmarkResult]:
    """
    Run a single benchmark configuration.
    
    Args:
        cfg: Benchmark configuration
        duration: How long to run the benchmark (seconds)
        warmup: Warmup time before measuring (seconds)
        verbose: Print progress messages
        
    Returns:
        BenchmarkResult or None if device unavailable
    """
    # Import here to avoid pygame init on module load
    from config import Config
    from src.game.breakout import Breakout
    from src.ai.agent import Agent
    
    # Setup config
    config = Config()
    config.BATCH_SIZE = cfg.batch_size
    config.LEARN_EVERY = cfg.learn_every
    config.GRADIENT_STEPS = cfg.gradient_steps
    config.USE_TORCH_COMPILE = False  # Skip for benchmarking consistency
    config.USE_MIXED_PRECISION = False
    
    if cfg.device == 'cpu':
        config.FORCE_CPU = True
    else:
        config.FORCE_CPU = False
        # Check if requested device is available
        if cfg.device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            if verbose:
                print(f"  ⚠️  MPS not available, skipping")
            return None
        if cfg.device == 'cuda' and not torch.cuda.is_available():
            if verbose:
                print(f"  ⚠️  CUDA not available, skipping")
            return None
    
    # Create game and agent
    game = Breakout(config, headless=True)
    agent = Agent(game.state_size, game.action_size, config)
    
    # Determine buffer fill size
    min_buffer = cfg.batch_size * 2
    target_buffer = max(min_buffer, cfg.buffer_fill)
    
    # Fill replay buffer
    if verbose:
        if target_buffer > min_buffer:
            print(f"  Filling buffer to {target_buffer:,}...", end=" ", flush=True)
        else:
            print(f"  Filling buffer ({min_buffer})...", end=" ", flush=True)
    
    state = game.reset()
    for i in range(target_buffer):
        action = agent.select_action(state)
        next_state, reward, done, _ = game.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state if not done else game.reset()
        
        # Progress indicator for large buffers
        if verbose and target_buffer >= 10000 and (i + 1) % 10000 == 0:
            print(f"{(i+1)//1000}k...", end=" ", flush=True)
    
    if verbose:
        print("done")
    
    buffer_size = len(agent.memory)
    
    # Warmup phase (also continues filling buffer)
    if verbose:
        print(f"  Warming up ({warmup}s)...", end=" ", flush=True)
    warmup_start = time.perf_counter()
    while time.perf_counter() - warmup_start < warmup:
        action = agent.select_action(state)
        next_state, reward, done, _ = game.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        state = next_state if not done else game.reset()
    if verbose:
        print("done")
    
    # Update buffer size after warmup
    buffer_size = len(agent.memory)
    
    # Reset counters after warmup
    agent.steps = 0
    agent._learn_step = 0
    episodes = 0
    total_episode_steps = 0
    
    # Get memory before benchmark
    mem_before = get_memory_usage_mb()
    
    # Benchmark phase
    if verbose:
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
            total_episode_steps += total_steps
        else:
            state = next_state
    
    elapsed = time.perf_counter() - start_time
    if verbose:
        print("done")
    
    # Calculate metrics
    steps_per_sec = total_steps / elapsed
    gradient_updates = agent.steps
    gradients_per_sec = gradient_updates / elapsed
    avg_episode_length = total_episode_steps / episodes if episodes > 0 else total_steps
    mem_after = get_memory_usage_mb()
    
    return BenchmarkResult(
        config_name=cfg.name,
        device=str(config.DEVICE),
        batch_size=cfg.batch_size,
        learn_every=cfg.learn_every,
        gradient_steps=cfg.gradient_steps,
        buffer_size=buffer_size,
        total_steps=total_steps,
        total_time=round(elapsed, 2),
        steps_per_sec=round(steps_per_sec, 1),
        gradient_updates=gradient_updates,
        gradients_per_sec=round(gradients_per_sec, 1),
        episodes_completed=episodes,
        avg_episode_length=round(avg_episode_length, 1),
        memory_mb=round(mem_after, 1),
    )


def print_result(result: BenchmarkResult, show_details: bool = True) -> None:
    """Pretty print a benchmark result."""
    print(f"\n  📊 {result.config_name}")
    print(f"     Device: {result.device} | Batch: {result.batch_size} | LE: {result.learn_every} | GS: {result.gradient_steps}")
    print(f"     Buffer: {result.buffer_size:,} experiences")
    print(f"     Steps: {result.total_steps:,} in {result.total_time}s → {result.steps_per_sec:,.0f} steps/sec")
    print(f"     Gradients: {result.gradient_updates:,} → {result.gradients_per_sec:,.0f} grad/sec")
    if show_details:
        print(f"     Episodes: {result.episodes_completed} | Avg length: {result.avg_episode_length:.0f} steps")
        if result.memory_mb > 0:
            print(f"     Memory: {result.memory_mb:.0f} MB")


def print_scaling_analysis(results: List[BenchmarkResult]) -> None:
    """Print analysis of buffer size scaling."""
    scaling_results = [r for r in results if r.config_name.startswith('scale_')]
    if len(scaling_results) < 2:
        return
    
    print(f"\n{'=' * 60}")
    print("BUFFER SCALING ANALYSIS")
    print("=" * 60)
    print("\nHow sampling performance changes with buffer size:")
    print(f"{'Buffer Size':<15} {'Steps/sec':>12} {'Grad/sec':>12} {'Δ vs 1k':>12}")
    print("-" * 51)
    
    baseline = None
    for r in sorted(scaling_results, key=lambda x: x.buffer_size):
        if baseline is None:
            baseline = r.gradients_per_sec
            delta = "baseline"
        else:
            pct_change = ((r.gradients_per_sec - baseline) / baseline) * 100
            delta = f"{pct_change:+.1f}%"
        print(f"{r.buffer_size:>12,}   {r.steps_per_sec:>12,.0f} {r.gradients_per_sec:>12,.0f} {delta:>12}")
    
    print("\n💡 Note: Large buffers (>3k) use optimized random.sample() for ~30x faster sampling")


def get_system_info() -> dict:
    """Collect system information for benchmark context."""
    info = {
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'platform': platform.platform(),
        'processor': platform.processor() or 'unknown',
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_device'] = torch.cuda.get_device_name(0)
    
    return info


def main():
    parser = argparse.ArgumentParser(
        description='DQN Training Performance Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                  # Standard benchmark
  python benchmark.py --quick          # Quick 3-second test
  python benchmark.py --realistic      # Test with 50k buffer (production-like)
  python benchmark.py --scaling        # Test buffer size impact
  python benchmark.py --full           # Run everything
  python benchmark.py --config turbo,realistic_cpu
        """
    )
    parser.add_argument('--quick', action='store_true', help='Quick 3-second benchmarks with small buffer')
    parser.add_argument('--realistic', action='store_true', help='Test with realistic 50k buffer')
    parser.add_argument('--scaling', action='store_true', help='Test buffer size scaling (1k to 100k)')
    parser.add_argument('--full', action='store_true', help='Full benchmark suite (all configs)')
    parser.add_argument('--duration', type=float, default=5.0, help='Benchmark duration per config (seconds)')
    parser.add_argument('--config', type=str, help='Comma-separated config names to test')
    parser.add_argument('--save', type=str, help='Save results to JSON file')
    parser.add_argument('--list', action='store_true', help='List available configs')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable benchmark configurations:")
        print("\n  Quick tests (small buffer):")
        for name in ['turbo', 'cpu_b64', 'cpu_b128', 'cpu_b256', 'cpu_b128_le8', 'cpu_b128_le16']:
            cfg = CONFIGS[name]
            print(f"    {name}: device={cfg.device}, batch={cfg.batch_size}, le={cfg.learn_every}, gs={cfg.gradient_steps}")
        
        print("\n  GPU tests:")
        for name in ['mps_b128', 'mps_b256', 'mps_b512', 'cuda_b256', 'cuda_b512']:
            cfg = CONFIGS[name]
            print(f"    {name}: device={cfg.device}, batch={cfg.batch_size}, le={cfg.learn_every}, gs={cfg.gradient_steps}")
        
        print("\n  Realistic tests (50k buffer):")
        for name in REALISTIC_CONFIGS:
            cfg = CONFIGS[name]
            print(f"    {name}: device={cfg.device}, batch={cfg.batch_size}, buffer={cfg.buffer_fill:,}")
        
        print("\n  Scaling tests (vary buffer size):")
        for name in SCALING_CONFIGS:
            cfg = CONFIGS[name]
            print(f"    {name}: buffer={cfg.buffer_fill:,}")
        return
    
    # Determine which configs to run
    if args.config:
        config_names = [c.strip() for c in args.config.split(',')]
        for name in config_names:
            if name not in CONFIGS:
                print(f"Unknown config: {name}")
                print(f"Use --list to see available configs")
                return
    elif args.full:
        config_names = FULL_CONFIGS
    elif args.scaling:
        config_names = SCALING_CONFIGS
    elif args.realistic:
        config_names = REALISTIC_CONFIGS
    elif args.quick:
        config_names = QUICK_CONFIGS
    else:
        config_names = STANDARD_CONFIGS
    
    # Adjust duration
    if args.quick and args.duration == 5.0:  # User didn't override
        duration = 3.0
    else:
        duration = args.duration
    
    # Print header
    sys_info = get_system_info()
    print("=" * 60)
    print("DQN Training Performance Benchmark")
    print("=" * 60)
    print(f"\nSystem: {sys_info['platform']}")
    print(f"Python: {sys_info['python_version']} | PyTorch: {sys_info['pytorch_version']}")
    print(f"CUDA: {'✓ ' + sys_info.get('cuda_device', '') if sys_info['cuda_available'] else '✗'}")
    print(f"MPS: {'✓' if sys_info['mps_available'] else '✗'}")
    print(f"\nConfigs to test: {', '.join(config_names)}")
    print(f"Duration per config: {duration}s")
    
    results = []
    
    for config_name in config_names:
        cfg = CONFIGS[config_name]
        print(f"\n{'─' * 50}")
        print(f"Running: {config_name}")
        if cfg.buffer_fill > 0:
            print(f"  (Pre-filling buffer to {cfg.buffer_fill:,} experiences)")
        
        try:
            result = run_benchmark(
                cfg=cfg,
                duration=duration,
                verbose=not args.quiet,
            )
            if result:
                results.append(result)
                if not args.quiet:
                    print_result(result, show_details=not args.quick)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print("=" * 60)
        
        # Sort by steps/sec
        results.sort(key=lambda r: r.steps_per_sec, reverse=True)
        
        print(f"\n{'Config':<20} {'Device':<8} {'Buffer':>10} {'Steps/sec':>12} {'Grad/sec':>12}")
        print("-" * 64)
        for r in results:
            buf_str = f"{r.buffer_size//1000}k" if r.buffer_size >= 1000 else str(r.buffer_size)
            print(f"{r.config_name:<20} {r.device:<8} {buf_str:>10} {r.steps_per_sec:>12,.0f} {r.gradients_per_sec:>12,.0f}")
        
        best = results[0]
        print(f"\n🏆 Fastest: {best.config_name} ({best.steps_per_sec:,.0f} steps/sec)")
        
        # Print scaling analysis if applicable
        print_scaling_analysis(results)
    
    # Save results
    if args.save and results:
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': sys_info,
            'duration_per_config': duration,
            'results': [asdict(r) for r in results],
        }
        with open(args.save, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to {args.save}")


if __name__ == '__main__':
    main()
