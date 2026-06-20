"""Headless training runtime for high-throughput local training."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Optional

import numpy as np

from config import Config
from src.ai.agent import Agent
from src.ai.evaluator import Evaluator
from src.ai.trainer import calculate_progress_count
from src.app.game_factory import create_training_environment
from src.app.headless_dashboard import HeadlessDashboardMixin
from src.app.model_service import ModelService as AppModelService
from src.app.training_runtime import (
    build_nn_snapshot,
    emit_nn_snapshot_to_dashboard,
    is_new_best_eval,
    is_new_best_score,
    read_eval_best_baseline,
    read_eval_best_record,
    should_emit_episode_metrics,
    write_eval_best_baseline,
)
from src.game import BaseGame, BaseVecGame, list_games

WEB_AVAILABLE: bool
WebDashboard: Optional[type[Any]]
try:
    from src.web import WebDashboard as _WebDashboard

    WebDashboard = _WebDashboard
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    WebDashboard = None

__all__ = ["HeadlessTrainer", "build_nn_snapshot", "emit_nn_snapshot_to_dashboard"]


class HeadlessTrainer(HeadlessDashboardMixin):
    """
    Lightweight headless trainer that skips pygame entirely.

    This provides maximum training throughput by:
        - No pygame initialization
        - No visualization overhead
        - Optimized training loop with configurable learning frequency
        - Progress reporting via terminal
        - Optional web dashboard for remote monitoring

    Usage:
        python main.py --headless --turbo --episodes 5000
        python main.py --headless --turbo --web --port 5001  # With web dashboard
    """

    def __init__(
        self,
        config: Config,
        args: argparse.Namespace,
        existing_dashboard: Optional[Any] = None,
    ):
        """
        Initialize headless trainer.

        Args:
            config: Configuration object
            args: Command line arguments
            existing_dashboard: Optional existing WebDashboard to reuse (for launcher mode)
        """
        self.config = config
        self.args = args
        self._existing_dashboard = existing_dashboard
        self.running = True
        self.model_service = AppModelService(config)

        # Apply CLI overrides to config
        if args.lr:
            config.LEARNING_RATE = args.lr
        if args.episodes:
            config.MAX_EPISODES = args.episodes
        if args.learn_every:
            config.LEARN_EVERY = args.learn_every
        if args.gradient_steps:
            config.GRADIENT_STEPS = args.gradient_steps
        if args.batch_size:
            config.BATCH_SIZE = args.batch_size
        if args.torch_compile:
            config.USE_TORCH_COMPILE = True

        # Apply turbo preset (overrides individual settings)
        # Optimized for M4 CPU based on benchmarks
        if args.turbo:
            config.LEARN_EVERY = 8
            config.BATCH_SIZE = 128
            config.GRADIENT_STEPS = 2
            config.USE_TORCH_COMPILE = False  # No benefit for small models on CPU
            config.FORCE_CPU = True  # CPU is faster for this (small MLP) model size
            print("🚀 Turbo mode: CPU, B=128, LE=8, GS=2 (~5000 steps/sec on M4)")

        # The convolutional net is a different, conv-heavy workload — the M4 GPU is
        # ~25% faster on it (the B=128 learn step ~30% faster), so don't force CPU.
        import torch as _torch

        gpu_available = _torch.cuda.is_available() or _torch.backends.mps.is_available()
        if getattr(config, "USE_CNN_STATE", False) and gpu_available:
            config.FORCE_CPU = False
            # NOTE: a larger batch is NOT a free throughput win here. With the fixed
            # turbo learn schedule it just doubles the replay ratio (2x learn compute
            # per env step), which lowered wall-clock throughput in practice. It's a
            # gradient-quality / sample-efficiency dial, not a speedup — left at the
            # turbo default. The real throughput win is the vectorized get_state().
            print(f"⚙️  CNN: using {config.DEVICE} (conv is faster on the GPU than CPU)")

        # Vectorized environment support
        self.num_envs = getattr(args, "vec_envs", 1)

        # Get game environment from registry
        try:
            game_environment = create_training_environment(
                config.GAME_NAME,
                config,
                num_envs=self.num_envs,
                headless=True,
            )
        except ValueError:
            print(f"❌ Unknown game: {config.GAME_NAME}")
            print(f"   Available games: {', '.join(list_games())}")
            sys.exit(1)

        self.vec_env: Optional[BaseVecGame] = game_environment.vec_env
        self.game: BaseGame = game_environment.game
        GameClass = game_environment.game_class
        if self.num_envs > 1 and self.vec_env is None:
            print(f"⚠️ Vectorized environments not yet supported for {config.GAME_NAME}")
            print("   Falling back to single environment")
        self.num_envs = game_environment.num_envs
        if self.vec_env is not None:
            print(f"🎮 Vectorized: {self.num_envs} parallel environments")

        # Create AI agent
        self.agent = Agent(
            state_size=self.game.state_size,
            action_size=self.game.action_size,
            config=config,
        )

        # Load model if specified (headless mode - just restore agent state)
        # Note: Compatibility check happens in _resolve_model_path, so we skip explicit load here
        # and let _resolve_model_path handle it to avoid duplicate loading

        # Create game-specific model directory
        os.makedirs(config.GAME_MODEL_DIR, exist_ok=True)

        # Tracking for loaded model info
        self.best_score = 0
        self.current_episode = 0
        self.scores: list[int] = []
        self.wins: list[bool] = []
        self.levels: list[int] = []  # Track level reached per episode
        self.q_values: list[float] = []  # Track Q-values for chart persistence
        self.losses: list[float] = []  # Track losses for chart persistence
        self.epsilons: list[float] = []  # Track epsilon for chart persistence
        self.rewards: list[float] = []  # Track rewards for chart persistence
        self.progresses: list[float] = []  # Track completion-progress (Crystal Caves)
        self.end_reasons: list[str] = []  # CA-03: why each episode ended
        self.progress_parts: list[dict] = []  # CA-03: Phi components at death
        self.total_steps = 0
        self.training_start_time = time.time()

        # Extended metrics tracking (previously missing from headless mode)
        self.exploration_actions = 0
        self.exploitation_actions = 0
        self.target_updates = 0
        self.last_target_update_step = 0

        # Web dashboard (initialize early so _load_model can use it)
        self.web_dashboard: Optional[Any] = None

        # Auto-load most recent model if no explicit model specified
        initial_model_path = self._resolve_model_path(
            args.model,
            state_size=self.game.state_size,
            action_size=self.game.action_size,
        )
        if initial_model_path:
            self._load_model(initial_model_path)

        # Setup web dashboard if enabled
        self.paused = False
        if self._existing_dashboard is not None:
            # Reuse existing dashboard from launcher mode (already running on the port)
            self.web_dashboard = self._existing_dashboard
            self._setup_web_callbacks()
            # Dashboard is already started, just send system info
            self._send_system_info()
            self._log_startup_info()
            if initial_model_path:
                self.web_dashboard.log(
                    f"📂 Auto-loaded: {os.path.basename(initial_model_path)}", "success"
                )
                # Sync history to dashboard NOW that dashboard is ready
                self._sync_history_to_dashboard_after_load(initial_model_path)
        elif hasattr(args, "web") and args.web and WEB_AVAILABLE and WebDashboard is not None:
            self.web_dashboard = WebDashboard(
                config, port=args.port, host=getattr(args, "host", "127.0.0.1")
            )
            self._setup_web_callbacks()
            self.web_dashboard.start()

            # Show URL prominently
            print("\n" + "=" * 60)
            print(f"🌐 WEB DASHBOARD: {self.web_dashboard.dashboard_url()}")
            print("=" * 60 + "\n")

            self._send_system_info()
            self._log_startup_info()
            if initial_model_path:
                self.web_dashboard.log(
                    f"📂 Auto-loaded: {os.path.basename(initial_model_path)}", "success"
                )
                # Sync history to dashboard NOW that dashboard is ready
                self._sync_history_to_dashboard_after_load(initial_model_path)

        # Initialize evaluator for deterministic performance tracking
        # Uses a separate single-game instance (not vectorized) for consistent eval
        self.evaluator: Optional[Evaluator] = None
        self._exploration_boost_active: bool = False
        self._exploration_boost_end_episode: int = 0
        if config.EVAL_EVERY > 0:
            eval_game = GameClass(config, headless=True)  # type: ignore[call-arg]
            self.evaluator = Evaluator(
                game=eval_game,
                agent=self.agent,
                config=config,
                log_dir=os.path.join(config.LOG_DIR, "eval"),
                plateau_threshold=config.EVAL_PLATEAU_THRESHOLD,
            )
            eval_best_baseline = read_eval_best_baseline(
                config.GAME_MODEL_DIR,
                config.GAME_NAME,
            )
            if eval_best_baseline is not None:
                self.evaluator.best_eval_score = eval_best_baseline
                if self.web_dashboard:
                    eval_best_record = read_eval_best_record(
                        config.GAME_MODEL_DIR,
                        config.GAME_NAME,
                    )
                    baseline_episode = (
                        int(eval_best_record.get("episode", 0))
                        if eval_best_record is not None
                        else 0
                    )
                    self.web_dashboard.publisher.record_eval_baseline(
                        episode=baseline_episode,
                        mean_score=eval_best_baseline,
                        num_games=int(getattr(config, "EVAL_EPISODES", 0) or 0),
                    )
                    self.web_dashboard.log(
                        f"🎯 Restored held-out best: mean {eval_best_baseline:.0f}",
                        "info",
                    )

    def train(self) -> None:
        """Run headless training loop with optimized throughput."""
        # Dispatch to vectorized training if using multiple environments
        if self.num_envs > 1:
            self.train_vectorized()
            return

        config = self.config

        # Calculate starting episode (may be resuming from loaded model)
        start_episode = self.current_episode

        print("\n" + "=" * 70)
        print("🚀 HEADLESS TRAINING - Maximum Performance Mode")
        if self.web_dashboard:
            print("🌐 Web dashboard enabled")
        print("=" * 70)
        eps_str = "∞ (Unlimited)" if config.MAX_EPISODES == 0 else str(config.MAX_EPISODES)
        print(f"   Episodes:        {start_episode} → {eps_str}")
        print(f"   Device:          {config.DEVICE}")
        print(f"   Batch size:      {config.BATCH_SIZE}")
        print(f"   Learn every:     {config.LEARN_EVERY} steps")
        print(f"   Gradient steps:  {config.GRADIENT_STEPS}")
        print(f"   torch.compile:   {config.USE_TORCH_COMPILE}")
        if self.best_score > 0:
            print(f"   Resumed best:    {self.best_score}")
        print("=" * 70 + "\n")

        # Training timing
        self.training_start_time = time.time()
        last_report_time = self.training_start_time
        steps_since_report = 0
        last_logged_episode = start_episode - 1  # Track last logged episode to prevent duplicates

        # MAX_EPISODES == 0 means unlimited (train until manually stopped)
        episode = start_episode
        while self.running and (config.MAX_EPISODES == 0 or episode < config.MAX_EPISODES):
            self.current_episode = episode

            # Handle pause (only if web dashboard is active)
            while self.running and self.paused:
                time.sleep(0.1)
            if not self.running:
                break

            state = self.game.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            info = {"score": 0, "won": False}

            while not done and episode_steps < config.MAX_STEPS_PER_EPISODE:
                # Handle pause during episode
                while self.paused:
                    time.sleep(0.1)

                # Select action
                action = self.agent.select_action(state, training=True)

                # Track exploration vs exploitation
                if self.agent._last_action_explored:
                    self.exploration_actions += 1
                else:
                    self.exploitation_actions += 1

                # Execute action
                next_state, reward, done, info = self.game.step(action)

                # Store experience
                self.agent.remember(state, action, reward, next_state, done)

                # Learn (agent handles LEARN_EVERY and GRADIENT_STEPS internally)
                self.agent.learn()

                # Track target network updates
                if (
                    self.agent.steps % config.TARGET_UPDATE == 0
                    and self.agent.steps != self.last_target_update_step
                ):
                    self.target_updates += 1
                    self.last_target_update_step = self.agent.steps
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"🎯 Target network updated (#{self.target_updates})",
                            "metric",
                            {
                                "step": self.agent.steps,
                                "update_number": self.target_updates,
                            },
                        )

                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                steps_since_report += 1

            # Episode complete
            self.agent.decay_epsilon(episode)
            self.agent.step_scheduler()  # Step learning rate scheduler
            self._apply_lr_decay(start_episode, episode)
            self.scores.append(info["score"])

            # Track wins (all bricks cleared)
            won = bool(info.get("won", False))
            self.wins.append(won)

            bricks_broken = calculate_progress_count(info, config)

            # Update web dashboard metrics (throttled to every 5 episodes for performance)
            # Always emit on: first 10 episodes, new best score, or every 5th episode
            is_new_best = is_new_best_score(info["score"], getattr(self, "best_score", 0))
            dashboard = self.web_dashboard
            if dashboard is not None and should_emit_episode_metrics(episode, is_new_best):
                avg_loss = self.agent.get_average_loss(100)

                # Calculate average Q-value for current state (was missing from headless)
                q_values = self.agent.get_q_values(state)
                avg_q_value = float(np.mean(q_values))

                dashboard.emit_metrics(
                    episode=episode,
                    score=info["score"],
                    epsilon=self.agent.epsilon,
                    loss=avg_loss,
                    total_steps=self.total_steps,
                    won=won,
                    reward=episode_reward,
                    memory_size=len(self.agent.memory),
                    avg_q_value=avg_q_value,
                    exploration_actions=self.exploration_actions,
                    exploitation_actions=self.exploitation_actions,
                    target_updates=self.target_updates,
                    bricks_broken=bricks_broken,
                    episode_length=episode_steps,
                    game_name=config.GAME_NAME,
                    cc_info=info if config.GAME_NAME == "crystal_caves" else None,
                )
                # Update performance settings in dashboard state
                dashboard.publisher.state.learn_every = config.LEARN_EVERY
                dashboard.publisher.state.gradient_steps = config.GRADIENT_STEPS
                dashboard.publisher.state.batch_size = config.BATCH_SIZE
                dashboard.publisher.state.cc_difficulty = getattr(
                    config, "CRYSTAL_CAVES_DIFFICULTY", ""
                )

                # Emit NN visualization data (throttled by server to ~10 FPS)
                self._emit_nn_visualization(state, action)

            # Progress reporting (terminal) - only log when episodes complete
            current_time = time.time()
            elapsed_since_report = current_time - last_report_time

            # Handle fresh start: if current_episode was reset (e.g., by _start_fresh)
            if self.current_episode < episode:
                episode = self.current_episode
                last_logged_episode = -1  # Reset so logging can resume
                start_episode = 0
                last_report_time = current_time
                steps_since_report = 0

            # Log every LOG_EVERY episodes OR if REPORT_INTERVAL_SECONDS has passed since last log
            # Only log if this is a new episode (prevent duplicate logs)
            should_log_by_episode = (episode - last_logged_episode) >= config.LOG_EVERY
            should_log_by_time = elapsed_since_report >= config.REPORT_INTERVAL_SECONDS
            is_new_episode = episode > last_logged_episode

            if is_new_episode and (should_log_by_episode or should_log_by_time):
                elapsed_total = current_time - self.training_start_time
                steps_per_sec = (
                    steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                )
                eps_per_hour = (
                    (episode - start_episode) / elapsed_total * 3600 if elapsed_total > 0 else 0
                )
                avg_score = np.mean(self.scores[-100:]) if self.scores else 0

                progress_msg = (
                    f"Ep {episode:5d} | "
                    f"Score: {info['score']:4d} | "
                    f"Avg: {avg_score:6.1f} | "
                    f"ε: {self.agent.epsilon:.3f} | "
                    f"⚡ {steps_per_sec:,.0f} steps/s | "
                    f"📊 {eps_per_hour:,.0f} ep/hr"
                )

                print(progress_msg)

                # Also log to web dashboard console
                if self.web_dashboard:
                    self.web_dashboard.log(progress_msg, "metric")

                last_logged_episode = episode
                last_report_time = current_time
                steps_since_report = 0

            # Save checkpoints
            if is_new_best_score(info["score"], self.best_score):
                self.best_score = info["score"]
                self._save_model(
                    f"{self.config.GAME_NAME}_best.pth", save_reason="best", quiet=True
                )
                if self.web_dashboard:
                    self.web_dashboard.log(f"🏆 New best score: {self.best_score}", "success")

            if episode % config.SAVE_EVERY == 0 and episode > 0:
                self._save_model(
                    f"{self.config.GAME_NAME}_ep{episode}.pth",
                    save_reason="periodic",
                    save_replay_buffer=False,  # Periodic saves are lightweight
                )
                self._cleanup_old_periodic_saves(keep_last=5)

            # Increment episode counter (was implicit in for loop, now explicit for while loop)
            episode += 1

        # Final save
        self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="final")

        # Summary
        total_time = time.time() - self.training_start_time
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"   Total episodes:   {self.current_episode - start_episode}")
        print(f"   Total steps:      {self.total_steps:,}")
        print(f"   Total time:       {total_time/60:.1f} minutes")
        print(f"   Avg steps/sec:    {self.total_steps/total_time:,.0f}")
        print(f"   Best score:       {self.best_score}")
        print(f"   Final avg score:  {np.mean(self.scores[-100:]):.1f}")
        recent_wins = self.wins[-100:]
        win_rate = sum(recent_wins) / len(recent_wins) if len(recent_wins) > 0 else 0
        print(f"   Win rate (100):   {win_rate*100:.1f}%")
        print("=" * 70)

        if self.web_dashboard:
            self.web_dashboard.log("✅ Training complete!", "success")

    def train_vectorized(self) -> None:
        """
        Run headless training with vectorized environments for parallel execution.

        This method runs N games simultaneously, collecting N experiences per step
        and performing batched action selection for improved throughput.
        """
        # This method is only called when num_envs > 1, so vec_env is always set
        assert self.vec_env is not None, "train_vectorized requires vec_env to be initialized"

        config = self.config
        num_envs = self.num_envs

        # Calculate starting episode (may be resuming from loaded model)
        start_episode = self.current_episode

        print("\n" + "=" * 70)
        print("🚀 VECTORIZED HEADLESS TRAINING - Maximum Performance Mode")
        if self.web_dashboard:
            print("🌐 Web dashboard enabled")
        print("=" * 70)
        print(f"   Environments:    {num_envs} parallel games")
        eps_str = "∞ (Unlimited)" if config.MAX_EPISODES == 0 else str(config.MAX_EPISODES)
        print(f"   Episodes:        {start_episode} → {eps_str}")
        print(f"   Device:          {config.DEVICE}")
        print(f"   Batch size:      {config.BATCH_SIZE}")
        print(f"   Learn every:     {config.LEARN_EVERY} steps")
        print(f"   Gradient steps:  {config.GRADIENT_STEPS}")
        print(f"   torch.compile:   {config.USE_TORCH_COMPILE}")
        if self.best_score > 0:
            print(f"   Resumed best:    {self.best_score}")
        print("=" * 70 + "\n")

        # Training timing
        self.training_start_time = time.time()
        last_report_time = self.training_start_time
        steps_since_report = 0

        # Per-environment episode tracking
        env_episode_rewards = np.zeros(
            num_envs, dtype=np.float64
        )  # Use float64 to prevent precision loss
        env_episode_steps = np.zeros(num_envs, dtype=np.int64)
        episodes_completed = 0

        # Initialize all environments
        states = (
            self.vec_env.reset().copy()
        )  # Shape: (num_envs, state_size) - copy to avoid aliasing

        # Track last completed episode info for reporting
        last_score = 0
        last_logged_episode = start_episode - 1  # Track last logged episode to prevent duplicates

        # MAX_EPISODES == 0 means unlimited (train until manually stopped)
        while self.running and (
            config.MAX_EPISODES == 0 or self.current_episode < config.MAX_EPISODES
        ):
            # Handle pause (only if web dashboard is active)
            while self.running and self.paused:
                time.sleep(0.1)
            if not self.running:
                break

            # Batch action selection for all environments
            actions, num_explored, num_exploited = self.agent.select_actions_batch(
                states, training=True
            )
            self.exploration_actions += num_explored
            self.exploitation_actions += num_exploited

            # Step all environments simultaneously
            next_states, rewards, dones, infos = self.vec_env.step_no_copy(actions)

            # Store experiences from all environments
            self.agent.remember_batch(states, actions, rewards, next_states, dones)

            # Learn (agent handles LEARN_EVERY and GRADIENT_STEPS internally)
            self.agent.learn()

            # Update per-environment tracking
            env_episode_rewards += rewards
            env_episode_steps += 1
            self.total_steps += num_envs
            steps_since_report += num_envs

            # Handle completed episodes
            for i, done in enumerate(dones):
                if done:
                    # Episode completed for environment i
                    score = infos[i].get("score", 0)
                    self.scores.append(score)

                    won = infos[i].get("won", False)
                    self.wins.append(won)

                    level = infos[i].get("level", 1)
                    self.levels.append(level)

                    self.progresses.append(float(infos[i].get("progress", 0.0)))
                    self.end_reasons.append(str(infos[i].get("end_reason", "")))
                    parts = infos[i].get("progress_parts")
                    if isinstance(parts, dict):
                        self.progress_parts.append(parts)

                    # Track metrics for persistence (used by save)
                    avg_loss = self.agent.get_average_loss(100)
                    q_values_arr = self.agent.get_q_values(states[i])
                    avg_q_value = float(np.mean(q_values_arr))
                    self.q_values.append(avg_q_value)
                    self.losses.append(avg_loss)
                    self.epsilons.append(self.agent.epsilon)
                    self.rewards.append(float(env_episode_rewards[i]))

                    is_new_best = is_new_best_score(score, self.best_score)

                    # Track best score
                    if is_new_best:
                        self.best_score = score
                        self._save_model(
                            f"{self.config.GAME_NAME}_best.pth",
                            save_reason="best",
                            quiet=True,
                        )
                        if self.web_dashboard:
                            self.web_dashboard.log(
                                f"🏆 New best score: {self.best_score}", "success"
                            )

                    # Track target updates (for persistence, independent of dashboard)
                    if self.agent.steps > self.last_target_update_step + config.TARGET_UPDATE:
                        self.target_updates += 1
                        self.last_target_update_step = self.agent.steps

                    # Update web dashboard metrics (throttled to every 5 episodes for performance)
                    # Always emit on: first 10 episodes, new best score, or every 5th episode
                    dashboard = self.web_dashboard
                    if dashboard is not None and (
                        should_emit_episode_metrics(self.current_episode, is_new_best)
                    ):
                        bricks_broken = calculate_progress_count(infos[i], config)

                        dashboard.emit_metrics(
                            episode=self.current_episode,
                            score=score,
                            epsilon=self.agent.epsilon,
                            loss=avg_loss,
                            total_steps=self.total_steps,
                            won=won,
                            reward=float(env_episode_rewards[i]),
                            memory_size=len(self.agent.memory),
                            avg_q_value=avg_q_value,
                            exploration_actions=self.exploration_actions,
                            exploitation_actions=self.exploitation_actions,
                            target_updates=self.target_updates,
                            bricks_broken=bricks_broken,
                            episode_length=int(env_episode_steps[i]),
                            game_name=config.GAME_NAME,
                            cc_info=(infos[i] if config.GAME_NAME == "crystal_caves" else None),
                        )
                        # Update performance settings in dashboard state
                        dashboard.publisher.state.learn_every = config.LEARN_EVERY
                        dashboard.publisher.state.gradient_steps = config.GRADIENT_STEPS
                        dashboard.publisher.state.batch_size = config.BATCH_SIZE
                        dashboard.publisher.state.cc_difficulty = getattr(
                            config, "CRYSTAL_CAVES_DIFFICULTY", ""
                        )

                        # Emit NN visualization data (throttled by server to ~10 FPS)
                        # Convert numpy int64 to Python int for JSON serialization
                        self._emit_nn_visualization(states[i], int(actions[i]))

                    # Store for reporting
                    last_score = score

                    # Reset per-environment tracking
                    env_episode_rewards[i] = 0.0
                    env_episode_steps[i] = 0

                    # Increment episode counter
                    episodes_completed += 1
                    self.current_episode += 1

                    # Save checkpoints (no replay buffer for periodic saves - saves disk space)
                    if self.current_episode % config.SAVE_EVERY == 0 and self.current_episode > 0:
                        self._save_model(
                            f"{self.config.GAME_NAME}_ep{self.current_episode}.pth",
                            save_reason="periodic",
                            save_replay_buffer=False,  # Periodic saves are lightweight
                        )
                        self._cleanup_old_periodic_saves(keep_last=5)

                    # Run deterministic evaluation periodically
                    if (
                        self.evaluator is not None
                        and config.EVAL_EVERY > 0
                        and self.current_episode % config.EVAL_EVERY == 0
                        and self.current_episode > 0
                    ):
                        eval_results = self.evaluator.evaluate(
                            num_episodes=config.EVAL_EPISODES,
                            max_steps=config.EVAL_MAX_STEPS,
                            episode_num=self.current_episode,
                        )
                        self.evaluator.log_results(eval_results)

                        if is_new_best_eval(self.evaluator, eval_results.mean_score):
                            eval_best_filename = f"{self.config.GAME_NAME}_eval_best.pth"
                            self._save_model(
                                eval_best_filename,
                                save_reason="eval_best",
                                quiet=True,
                                save_replay_buffer=False,
                            )
                            write_eval_best_baseline(
                                self.config.GAME_MODEL_DIR,
                                self.config.GAME_NAME,
                                episode=self.current_episode,
                                mean_score=eval_results.mean_score,
                                checkpoint=eval_best_filename,
                            )
                            if self.web_dashboard:
                                self.web_dashboard.log(
                                    f"🎯 New held-out eval best: {eval_results.mean_score:.0f}",
                                    "success",
                                )

                        # Surface the held-out eval on the dashboard (the trustworthy
                        # generalisation number, distinct from the training win rate).
                        if self.web_dashboard:
                            self.web_dashboard.publisher.record_eval(
                                episode=self.current_episode,
                                mean_score=eval_results.mean_score,
                                std_score=eval_results.std_score,
                                median_score=eval_results.median_score,
                                win_rate=eval_results.win_rate,
                                num_games=eval_results.num_games,
                            )

                        # Early-stop: end the stage once eval has plateaued, instead
                        # of training the live policy past its peak into collapse
                        # (the best checkpoint already holds the peak). Uses its own
                        # patience and pre-empts the exploration boost below.
                        early_patience = getattr(config, "EARLY_STOP_PATIENCE", 4)
                        if (
                            getattr(config, "EARLY_STOP_ON_PLATEAU", False)
                            and self.evaluator.evals_since_improvement >= early_patience
                        ):
                            print(
                                f"\n⏹️  Early stop: eval plateaued "
                                f"({self.evaluator.evals_since_improvement} evals without "
                                f"improvement). Best checkpoint holds the peak.\n"
                            )
                            if self.web_dashboard:
                                self.web_dashboard.log("⏹️ Early stop: eval plateaued", "warning")
                            self.running = False

                        # Auto-exploration boost: when plateau detected, increase epsilon
                        elif self.evaluator.is_plateau() and not self._exploration_boost_active:
                            self._exploration_boost_active = True
                            self._exploration_boost_end_episode = (
                                self.current_episode + config.EVAL_PLATEAU_BOOST_EPISODES
                            )
                            old_epsilon = self.agent.epsilon
                            self.agent.epsilon = config.EVAL_PLATEAU_EPSILON_BOOST
                            print(
                                f"\n🚀 PLATEAU DETECTED! Boosting exploration: "
                                f"ε {old_epsilon:.3f} → {self.agent.epsilon:.3f} "
                                f"for {config.EVAL_PLATEAU_BOOST_EPISODES} episodes\n"
                            )
                            if self.web_dashboard:
                                self.web_dashboard.log(
                                    f"🚀 Exploration boost activated! ε → {self.agent.epsilon:.2f}",
                                    "warning",
                                )

                        # Log to web dashboard if available
                        if self.web_dashboard:
                            plateau_str = (
                                " ⚠️ PLATEAU DETECTED" if self.evaluator.is_plateau() else ""
                            )
                            self.web_dashboard.log(
                                f"📊 EVAL: {eval_results.mean_score:.0f} avg, "
                                f"max level {eval_results.max_level}, "
                                f"{eval_results.win_rate*100:.0f}% wins{plateau_str}",
                                ("info" if not self.evaluator.is_plateau() else "warning"),
                            )

            # Decay epsilon once per step if any episodes completed
            # (NOT per environment - that would decay too fast with many parallel envs)
            if np.any(dones):
                # Check if exploration boost period has ended
                if (
                    self._exploration_boost_active
                    and self.current_episode >= self._exploration_boost_end_episode
                ):
                    self._exploration_boost_active = False
                    # Reset epsilon to minimum and let it decay normally
                    self.agent.epsilon = config.EPSILON_END
                    print(
                        f"\n✓ Exploration boost ended. Resuming normal ε={self.agent.epsilon:.3f}\n"
                    )
                    if self.web_dashboard:
                        self.web_dashboard.log(
                            f"✓ Exploration boost ended, ε → {self.agent.epsilon:.3f}",
                            "info",
                        )
                    # Reset plateau counter so we can detect new plateaus
                    if self.evaluator:
                        self.evaluator.evals_since_improvement = 0

                # Only decay epsilon if not in boost mode
                if not self._exploration_boost_active:
                    self.agent.decay_epsilon(self.current_episode)
                self.agent.step_scheduler()  # Step learning rate scheduler
                self._apply_lr_decay(start_episode, self.current_episode)

            # Update states for next iteration (vector envs auto-reset completed games)
            states = next_states.copy()

            # Progress reporting (terminal) - only log when new episodes complete
            # Check if we should log: either LOG_EVERY episodes completed OR time interval passed
            current_time = time.time()
            elapsed_since_report = current_time - last_report_time

            # Handle fresh start: if current_episode < last_logged_episode, a reset occurred
            if last_logged_episode > self.current_episode:
                last_logged_episode = -1  # Reset so logging can resume
                last_report_time = current_time
                steps_since_report = 0
                episodes_completed = 0  # Reset episodes count for accurate ep/hr

            should_log_by_episode = (self.current_episode - last_logged_episode) >= config.LOG_EVERY
            should_log_by_time = elapsed_since_report >= config.REPORT_INTERVAL_SECONDS

            # Only log if we have new episodes AND (LOG_EVERY condition OR time interval)
            if self.current_episode > last_logged_episode and (
                should_log_by_episode or should_log_by_time
            ):
                steps_per_sec = (
                    steps_since_report / elapsed_since_report if elapsed_since_report > 0 else 0
                )
                avg_score = np.mean(self.scores[-100:]) if self.scores else 0
                avg_loss = self.agent.get_average_loss(100)
                avg_q = np.mean(self.q_values[-100:]) if self.q_values else 0.0

                # Completion-progress (Crystal Caves): rolling mean and best-so-far
                # of info["progress"]. This is the signal that should climb before
                # win-rate does, so surface it directly in the training log.
                progress_str = ""
                if self.progresses:
                    avg_prog = float(np.mean(self.progresses[-100:]))
                    best_prog = float(np.max(self.progresses))
                    progress_str = f"Φ completion: {avg_prog:.3f} (best {best_prog:.3f}) | "

                lr_str = ""
                if getattr(self.config, "LR_DECAY", False):
                    lr_str = f"lr: {self.agent.get_learning_rate():.1e} | "

                progress_msg = (
                    f"Ep {self.current_episode:5d} | "
                    f"Score: {last_score:4d} | "
                    f"Avg: {avg_score:6.1f} | "
                    f"{progress_str}"
                    f"Loss: {avg_loss:.4f} | "
                    f"Q: {avg_q:.2f} | "
                    f"{lr_str}"
                    f"ε: {self.agent.epsilon:.3f} | "
                    f"⚡ {steps_per_sec:,.0f} steps/s"
                )

                print(progress_msg)

                # Also log to web dashboard console
                if self.web_dashboard:
                    self.web_dashboard.log(progress_msg, "metric")

                last_logged_episode = self.current_episode
                last_report_time = current_time
                steps_since_report = 0

        # Final save
        self._save_model(f"{self.config.GAME_NAME}_final.pth", save_reason="final")

        # Summary
        total_time = time.time() - self.training_start_time
        print("\n" + "=" * 70)
        print("✅ VECTORIZED TRAINING COMPLETE!")
        print("=" * 70)
        print(f"   Parallel envs:    {num_envs}")
        print(f"   Total episodes:   {self.current_episode - start_episode}")
        print(f"   Total steps:      {self.total_steps:,}")
        print(f"   Total time:       {total_time/60:.1f} minutes")
        print(f"   Avg steps/sec:    {self.total_steps/total_time:,.0f}")
        print(f"   Best score:       {self.best_score}")
        print(f"   Final avg score:  {np.mean(self.scores[-100:]):.1f}")
        recent_wins = self.wins[-100:]
        win_rate = sum(recent_wins) / len(recent_wins) if len(recent_wins) > 0 else 0
        print(f"   Win rate (100):   {win_rate*100:.1f}%")
        if self.progresses:
            print(f"   Final avg prog:   {np.mean(self.progresses[-100:]):.3f}")
            print(f"   Best progress:    {np.max(self.progresses):.3f}")
        self._print_progress_breakdown()
        print("=" * 70)

        if self.web_dashboard:
            self.web_dashboard.log("✅ Vectorized training complete!", "success")

    def _apply_lr_decay(self, start_episode: int, current_episode: int) -> None:
        """Cosine-decay the learning rate from LEARNING_RATE to LR_MIN over this
        run's episodes (start_episode..MAX_EPISODES). Early LR matches the old
        constant rate; late LR approaches zero, freezing the policy near its peak
        so the live win rate stops oscillating. No-op unless LR_DECAY is set and
        the run has a finite episode target."""
        if not getattr(self.config, "LR_DECAY", False):
            return
        target = self.config.MAX_EPISODES
        if target <= 0:
            return  # unlimited run -> no horizon to decay over
        lr0 = self.config.LEARNING_RATE
        lr_min = getattr(self.config, "LR_MIN", 1e-5)
        span = max(1, target - start_episode)
        frac = min(1.0, max(0.0, (current_episode - start_episode) / span))
        lr = lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * frac))
        self.agent.set_learning_rate(lr)

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        """Human-friendly time-remaining, e.g. '3m', '1h12m', '45s'."""
        s = int(max(0, seconds))
        if s >= 3600:
            return f"{s // 3600}h{(s % 3600) // 60:02d}m"
        if s >= 60:
            return f"{s // 60}m"
        return f"{s}s"

    def _print_progress_breakdown(self, window: int = 100) -> None:
        """CA-03: print where the agent stalls — the end-reason mix and the mean
        completion-progress components over the last `window` episodes. This
        pinpoints the gate (crystals vs switch vs depth) the agent gets stuck on.
        """
        if not self.end_reasons and not self.progress_parts:
            return

        recent_reasons = self.end_reasons[-window:]
        if recent_reasons:
            counts: dict[str, int] = {}
            for reason in recent_reasons:
                counts[reason] = counts.get(reason, 0) + 1
            total = len(recent_reasons)
            mix = ", ".join(
                f"{name} {count / total * 100:.0f}%"
                for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
            )
            print(f"   End reasons:      {mix}")

        recent_parts = self.progress_parts[-window:]
        if recent_parts:
            keys = ("crystal_frac", "switch_done", "depth_frac", "won")
            means = {
                key: float(np.mean([float(p.get(key, 0.0)) for p in recent_parts])) for key in keys
            }
            print(
                "   Phi@death parts:  "
                f"crystals {means['crystal_frac']:.2f} | "
                f"switch {means['switch_done']:.2f} | "
                f"depth {means['depth_frac']:.2f} | "
                f"won {means['won']:.2f}"
            )
