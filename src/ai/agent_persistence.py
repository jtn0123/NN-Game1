"""Checkpoint save, load, and inspection helpers for DQN agents."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.utils.checkpoint_loader import load_checkpoint

from .agent_metadata import SaveMetadata, TrainingHistory


class AgentPersistenceMixin:
    def save(
        self: Any,
        filepath: str,
        save_reason: str = "manual",
        episode: int = 0,
        best_score: int = 0,
        avg_score_last_100: float = 0.0,
        win_rate: float = 0.0,
        max_level: int = 1,
        training_start_time: Optional[float] = None,
        training_history: Optional["TrainingHistory"] = None,
        save_replay_buffer: bool = False,
        quiet: bool = False,
    ) -> Optional[SaveMetadata]:
        """
        Save agent state to file with rich metadata.

        Args:
            filepath: Path to save file
            save_reason: Why this save is happening ('best', 'periodic', 'manual', 'final', 'interrupted')
            episode: Current episode number
            best_score: Best score achieved so far
            avg_score_last_100: Average score over last 100 episodes
            win_rate: Win rate over last 100 episodes
            training_start_time: Unix timestamp when training started (for calculating total time)
            training_history: Training history for dashboard restoration (scores, rewards, etc.)
            save_replay_buffer: If True, save the replay buffer for cross-session persistence
            quiet: If True, suppress most output

        Returns:
            SaveMetadata object if save succeeded, None on failure
        """
        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Calculate training time
        total_time = 0.0
        if training_start_time:
            total_time = time.time() - training_start_time

        # Build metadata
        metadata = SaveMetadata(
            timestamp=datetime.now().isoformat(),
            save_reason=save_reason,
            total_training_time_seconds=total_time,
            episode=episode,
            total_steps=self.steps,
            epsilon=self.epsilon,
            best_score=best_score,
            avg_score_last_100=avg_score_last_100,
            avg_loss=self.get_average_loss(100),
            win_rate=win_rate,
            memory_buffer_size=len(self.memory),
            learning_rate=self.config.LEARNING_RATE,
            gamma=self.config.GAMMA,
            batch_size=self.config.BATCH_SIZE,
            hidden_layers=list(self.config.HIDDEN_LAYERS),
            epsilon_start=self.config.EPSILON_START,
            epsilon_end=self.config.EPSILON_END,
            epsilon_decay=self.config.EPSILON_DECAY,
            use_dueling=self.config.USE_DUELING,
        )

        # Get state dicts, stripping _orig_mod. prefix if model is compiled
        # This ensures saved models are portable regardless of torch.compile status
        policy_state = self.policy_net.state_dict()
        target_state = self.target_net.state_dict()

        if self._compiled:
            # Strip _orig_mod. prefix for portability
            policy_state = {k.replace("_orig_mod.", ""): v for k, v in policy_state.items()}
            target_state = {k.replace("_orig_mod.", ""): v for k, v in target_state.items()}

        checkpoint = {
            "policy_net_state_dict": policy_state,
            "target_net_state_dict": target_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "_learn_step": self._learn_step,
            "_next_target_update": self._next_target_update,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "metadata": metadata.to_dict(),
        }

        # Save training history if provided (for dashboard restoration)
        if training_history is not None:
            checkpoint["training_history"] = training_history.to_dict()

        # Optionally save replay buffer for cross-session persistence
        if save_replay_buffer and len(self.memory) > 0:
            checkpoint["replay_buffer"] = self.memory.save_to_dict()
            if not quiet:
                buffer_size_mb = (len(self.memory) * self.state_size * 8 * 2) / (
                    1024 * 1024
                )  # Rough estimate
                print(
                    f"💾 Saving replay buffer ({len(self.memory):,} experiences, ~{buffer_size_mb:.1f}MB)"
                )

        try:
            torch.save(checkpoint, filepath)

            # Verify the save by checking file exists and size
            if not os.path.exists(filepath):
                print(f"❌ Save verification FAILED: {filepath} not found after save")
                return None

            file_size = os.path.getsize(filepath)
            if file_size < 1000:  # Less than 1KB is suspicious
                print(f"⚠️ Warning: Saved file seems too small ({file_size} bytes)")

            # Format output
            if not quiet:
                size_mb = file_size / (1024 * 1024)
                reason_emoji = {
                    "best": "🏆",
                    "periodic": "📅",
                    "manual": "💾",
                    "final": "✅",
                    "interrupted": "⛔",
                }.get(save_reason, "💾")

                print(f"\n{reason_emoji} Model Saved: {os.path.basename(filepath)}")
                print(f"   Episode: {episode:,} | Steps: {self.steps:,} | ε: {self.epsilon:.4f}")
                print(
                    f"   Best Score: {best_score} | Avg(100): {avg_score_last_100:.1f} | Win Rate: {win_rate*100:.1f}% | Max Lv: {max_level}"
                )
                print(f"   Size: {size_mb:.2f} MB | Reason: {save_reason}")
                if total_time > 0:
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    print(f"   Training Time: {hours}h {minutes}m")

            return metadata

        except Exception as e:
            print(f"❌ Save FAILED: {e}")
            return None

    def _adapt_state_dict_for_compile(
        self: Any, state_dict: Dict[str, Any], target_module
    ) -> Dict[str, Any]:
        """
        Adapt state dict keys between compiled and non-compiled models.

        torch.compile() wraps the model and prefixes keys with '_orig_mod.'
        This method handles loading models regardless of compile status.

        Args:
            state_dict: The state dict to adapt
            target_module: The module to load into (may be compiled or not)

        Returns:
            Adapted state dict with correct key prefixes
        """
        # Check if saved state dict has _orig_mod prefix
        saved_has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())

        # Check if target module expects _orig_mod prefix (is compiled)
        target_expects_prefix = self._compiled

        if saved_has_prefix == target_expects_prefix:
            # No adaptation needed
            return state_dict

        adapted = {}
        if saved_has_prefix and not target_expects_prefix:
            # Remove _orig_mod. prefix
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
                adapted[new_key] = v
        elif not saved_has_prefix and target_expects_prefix:
            # Add _orig_mod. prefix
            for k, v in state_dict.items():
                adapted[f"_orig_mod.{k}"] = v

        return adapted

    def _trusted_checkpoint_dirs(self: Any) -> List[str]:
        """Return local directories that may contain app-created checkpoints."""
        dirs = [self.config.MODEL_DIR]
        game_model_dir = getattr(self.config, "GAME_MODEL_DIR", None)
        if game_model_dir:
            dirs.append(game_model_dir)
        return dirs

    def load(
        self: Any, filepath: str, quiet: bool = False
    ) -> Tuple[Optional[SaveMetadata], Optional["TrainingHistory"]]:
        """
        Load agent state from file with detailed resume summary.

        Args:
            filepath: Path to checkpoint file
            quiet: If True, suppress output

        Returns:
            Tuple of (SaveMetadata, TrainingHistory) - either may be None for old saves
        """
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return None, None

        try:
            checkpoint = load_checkpoint(
                filepath,
                map_location=self.device,
                trusted_dirs=self._trusted_checkpoint_dirs(),
                allow_unsafe_fallback=True,
            )
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None, None

        # Check for architecture mismatch
        saved_state_size = checkpoint.get("state_size", self.state_size)
        saved_action_size = checkpoint.get("action_size", self.action_size)

        # If architecture doesn't match, cannot load this model
        if saved_state_size != self.state_size or saved_action_size != self.action_size:
            if not quiet:
                if saved_state_size != self.state_size:
                    print(
                        f"⚠️  Model incompatible: State size mismatch (saved: {saved_state_size}, current: {self.state_size})"
                    )
                if saved_action_size != self.action_size:
                    print(
                        f"⚠️  Model incompatible: Action size mismatch (saved: {saved_action_size}, current: {self.action_size})"
                    )
                print(f"❌ Cannot load model - architecture mismatch. Starting fresh training.")
            return None, None

        # Adapt state dicts for torch.compile() compatibility
        policy_state = self._adapt_state_dict_for_compile(
            checkpoint["policy_net_state_dict"], self.policy_net
        )
        target_state = self._adapt_state_dict_for_compile(
            checkpoint["target_net_state_dict"], self.target_net
        )

        # Load network weights
        self.policy_net.load_state_dict(policy_state)
        self.target_net.load_state_dict(target_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self._learn_step = checkpoint.get("_learn_step", 0)  # Backwards compatible
        # Calculate next target update based on current steps (backwards compatible)
        self._next_target_update = checkpoint.get(
            "_next_target_update", self.steps + self.config.TARGET_UPDATE
        )

        # Load metadata if available
        metadata = None
        if "metadata" in checkpoint:
            try:
                metadata = SaveMetadata.from_dict(checkpoint["metadata"])
            except Exception:
                pass  # Old format without full metadata

        # Load training history if available (for dashboard restoration)
        training_history = None
        if "training_history" in checkpoint:
            try:
                training_history = TrainingHistory.from_dict(checkpoint["training_history"])
            except Exception:
                pass  # Old format without training history

        # Load replay buffer if available (for cross-session persistence)
        replay_buffer_loaded = False
        if "replay_buffer" in checkpoint:
            try:
                replay_buffer_loaded = self.memory.load_from_dict(checkpoint["replay_buffer"])
            except Exception as e:
                if not quiet:
                    print(f"⚠️ Could not restore replay buffer: {e}")

        if not quiet:
            file_size = os.path.getsize(filepath)
            size_mb = file_size / (1024 * 1024)

            print(f"\n{'='*60}")
            print(f"📂 Resuming Training")
            print(f"{'='*60}")
            print(f"   Model: {os.path.basename(filepath)} ({size_mb:.2f} MB)")

            if metadata:
                # Parse timestamp for human-readable format
                try:
                    save_time = datetime.fromisoformat(metadata.timestamp)
                    time_ago = datetime.now() - save_time
                    if time_ago.days > 0:
                        time_str = f"{time_ago.days}d ago"
                    elif time_ago.seconds > 3600:
                        time_str = f"{time_ago.seconds // 3600}h ago"
                    else:
                        time_str = f"{time_ago.seconds // 60}m ago"
                    print(f"   Saved: {save_time.strftime('%b %d, %Y %I:%M %p')} ({time_str})")
                except Exception:
                    print(f"   Saved: {metadata.timestamp}")

                print(
                    f"\n   Episode: {metadata.episode:,} | Steps: {metadata.total_steps:,} | ε: {metadata.epsilon:.4f}"
                )
                print(
                    f"   Best Score: {metadata.best_score} | Avg(100): {metadata.avg_score_last_100:.1f}"
                )
                print(
                    f"   Win Rate: {metadata.win_rate*100:.1f}% | Avg Loss: {metadata.avg_loss:.4f}"
                )

                if metadata.total_training_time_seconds > 0:
                    hours = int(metadata.total_training_time_seconds // 3600)
                    minutes = int((metadata.total_training_time_seconds % 3600) // 60)
                    print(f"   Previous Training Time: {hours}h {minutes}m")

                print(
                    f"\n   Config: LR={metadata.learning_rate}, γ={metadata.gamma}, Batch={metadata.batch_size}"
                )
                print(f"   Architecture: {metadata.hidden_layers}")
            else:
                # Old format - show basic info
                print(f"\n   Steps: {self.steps:,} | Epsilon: {self.epsilon:.4f}")
                print(f"   (Legacy save - no detailed metadata)")

            # Report training history status
            if training_history and len(training_history.scores) > 0:
                print(f"   Training History: {len(training_history.scores)} episodes restored")
            else:
                print(f"   Training History: Not available (older save format)")

            # Report replay buffer status
            if replay_buffer_loaded:
                print(f"   Replay Buffer: {len(self.memory):,} experiences restored")
            else:
                print(f"   Replay Buffer: Starting fresh (not saved or incompatible)")

            print(f"{'='*60}\n")

        return metadata, training_history

    @staticmethod
    def inspect_model(
        filepath: str,
        trusted_dirs: Optional[List[str]] = None,
        allow_unsafe_fallback: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Inspect a model file without loading it into an agent.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with model info, or None on error
        """
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None

        try:
            checkpoint = load_checkpoint(
                filepath,
                map_location="cpu",
                trusted_dirs=trusted_dirs,
                allow_unsafe_fallback=allow_unsafe_fallback,
            )
        except Exception as e:
            print(f"❌ Failed to read model: {e}")
            return None

        file_size = os.path.getsize(filepath)
        file_mtime = os.path.getmtime(filepath)

        info = {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "file_modified": datetime.fromtimestamp(file_mtime).isoformat(),
            "steps": checkpoint.get("steps", "unknown"),
            "epsilon": checkpoint.get("epsilon", "unknown"),
            "state_size": checkpoint.get("state_size", "unknown"),
            "action_size": checkpoint.get("action_size", "unknown"),
            "has_metadata": "metadata" in checkpoint,
            "metadata": checkpoint.get("metadata", None),
        }

        return info

    @staticmethod
    def list_models(model_dir: str = "models") -> List[Dict[str, Any]]:
        """
        List all model files in a directory with their metadata.

        Args:
            model_dir: Directory to scan for .pth files

        Returns:
            List of model info dictionaries, sorted by modified time (newest first)
        """
        models: List[Dict[str, Any]] = []

        if not os.path.exists(model_dir):
            return models

        for filename in os.listdir(model_dir):
            if filename.endswith(".pth"):
                filepath = os.path.join(model_dir, filename)
                info = AgentPersistenceMixin.inspect_model(
                    filepath,
                    trusted_dirs=[model_dir],
                    allow_unsafe_fallback=True,
                )
                if info:
                    models.append(info)

        # Sort by file modified time, newest first
        models.sort(key=lambda x: x["file_modified"], reverse=True)
        return models
