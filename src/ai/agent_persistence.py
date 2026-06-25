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
        """Save agent state to file with rich metadata."""
        self._ensure_checkpoint_dir(filepath)
        total_time = self._training_time_seconds(training_start_time)
        metadata = self._build_save_metadata(
            save_reason=save_reason,
            total_time=total_time,
            episode=episode,
            best_score=best_score,
            avg_score_last_100=avg_score_last_100,
            win_rate=win_rate,
        )
        checkpoint = self._build_checkpoint_payload(
            metadata=metadata,
            training_history=training_history,
        )
        self._attach_replay_buffer(
            checkpoint,
            save_replay_buffer=save_replay_buffer,
            quiet=quiet,
        )

        return self._write_checkpoint_file(
            checkpoint=checkpoint,
            filepath=filepath,
            metadata=metadata,
            quiet=quiet,
            save_reason=save_reason,
            episode=episode,
            best_score=best_score,
            avg_score_last_100=avg_score_last_100,
            win_rate=win_rate,
            max_level=max_level,
            total_time=total_time,
        )

    def _write_checkpoint_file(
        self: Any,
        *,
        checkpoint: Dict[str, Any],
        filepath: str,
        metadata: SaveMetadata,
        quiet: bool,
        save_reason: str,
        episode: int,
        best_score: int,
        avg_score_last_100: float,
        win_rate: float,
        max_level: int,
        total_time: float,
    ) -> Optional[SaveMetadata]:
        try:
            torch.save(checkpoint, filepath)
            file_size = self._verify_saved_checkpoint(filepath)
            if file_size is None:
                return None

            if not quiet:
                self._print_save_summary(
                    filepath=filepath,
                    file_size=file_size,
                    save_reason=save_reason,
                    episode=episode,
                    best_score=best_score,
                    avg_score_last_100=avg_score_last_100,
                    win_rate=win_rate,
                    max_level=max_level,
                    total_time=total_time,
                )

            return metadata

        except Exception as e:
            print(f"❌ Save FAILED: {e}")
            return None

    def _ensure_checkpoint_dir(self: Any, filepath: str) -> None:
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def _training_time_seconds(self: Any, training_start_time: Optional[float]) -> float:
        if not training_start_time:
            return 0.0
        return time.time() - training_start_time

    def _build_save_metadata(
        self: Any,
        *,
        save_reason: str,
        total_time: float,
        episode: int,
        best_score: int,
        avg_score_last_100: float,
        win_rate: float,
    ) -> SaveMetadata:
        return SaveMetadata(
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

    def _portable_state_dict(self: Any, module: Any) -> Dict[str, Any]:
        state_dict = module.state_dict()
        if not self._compiled:
            return state_dict
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    def _build_checkpoint_payload(
        self: Any,
        *,
        metadata: SaveMetadata,
        training_history: Optional["TrainingHistory"],
    ) -> Dict[str, Any]:
        checkpoint = {
            "policy_net_state_dict": self._portable_state_dict(self.policy_net),
            "target_net_state_dict": self._portable_state_dict(self.target_net),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "_learn_step": self._learn_step,
            "_next_target_update": self._next_target_update,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "metadata": metadata.to_dict(),
        }
        if training_history is not None:
            checkpoint["training_history"] = training_history.to_dict()
        return checkpoint

    def _attach_replay_buffer(
        self: Any,
        checkpoint: Dict[str, Any],
        *,
        save_replay_buffer: bool,
        quiet: bool,
    ) -> None:
        if not save_replay_buffer or len(self.memory) == 0:
            return

        checkpoint["replay_buffer"] = self.memory.save_to_dict()
        if not quiet:
            buffer_size_mb = self._replay_buffer_size_mb()
            print(
                f"💾 Saving replay buffer ({len(self.memory):,} experiences, ~{buffer_size_mb:.1f}MB)"
            )

    def _replay_buffer_size_mb(self: Any) -> float:
        return (len(self.memory) * self.state_size * 8 * 2) / (1024 * 1024)

    def _verify_saved_checkpoint(self: Any, filepath: str) -> Optional[int]:
        if not os.path.exists(filepath):
            print(f"❌ Save verification FAILED: {filepath} not found after save")
            return None

        file_size = os.path.getsize(filepath)
        if file_size < 1000:
            print(f"⚠️ Warning: Saved file seems too small ({file_size} bytes)")
        return file_size

    def _save_reason_emoji(self: Any, save_reason: str) -> str:
        return {
            "best": "🏆",
            "periodic": "📅",
            "manual": "💾",
            "final": "✅",
            "interrupted": "⛔",
        }.get(save_reason, "💾")

    def _print_save_summary(
        self: Any,
        *,
        filepath: str,
        file_size: int,
        save_reason: str,
        episode: int,
        best_score: int,
        avg_score_last_100: float,
        win_rate: float,
        max_level: int,
        total_time: float,
    ) -> None:
        size_mb = file_size / (1024 * 1024)
        reason_emoji = self._save_reason_emoji(save_reason)

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

    def _load_checkpoint_payload(
        self: Any,
        filepath: str,
        *,
        quiet: bool = False,
        error_prefix: str = "❌ Failed to load model",
    ) -> Optional[Dict[str, Any]]:
        try:
            return load_checkpoint(
                filepath,
                map_location=self.device,
                trusted_dirs=self._trusted_checkpoint_dirs(),
                allow_unsafe_fallback=True,
            )
        except Exception as e:
            if not quiet:
                print(f"{error_prefix}: {e}")
            return None

    def _checkpoint_architecture_sizes(self: Any, checkpoint: Dict[str, Any]) -> Tuple[Any, Any]:
        return (
            checkpoint.get("state_size", self.state_size),
            checkpoint.get("action_size", self.action_size),
        )

    def _checkpoint_architecture_matches(self: Any, checkpoint: Dict[str, Any]) -> bool:
        saved_state_size, saved_action_size = self._checkpoint_architecture_sizes(checkpoint)
        return saved_state_size == self.state_size and saved_action_size == self.action_size

    def _print_architecture_mismatch(self: Any, checkpoint: Dict[str, Any]) -> None:
        saved_state_size, saved_action_size = self._checkpoint_architecture_sizes(checkpoint)
        if saved_state_size != self.state_size:
            print(
                f"⚠️  Model incompatible: State size mismatch (saved: {saved_state_size}, current: {self.state_size})"
            )
        if saved_action_size != self.action_size:
            print(
                f"⚠️  Model incompatible: Action size mismatch (saved: {saved_action_size}, current: {self.action_size})"
            )
        print("❌ Cannot load model - architecture mismatch. Starting fresh training.")

    def _restore_checkpoint_network_weights(self: Any, checkpoint: Dict[str, Any]) -> None:
        policy_state = self._adapt_state_dict_for_compile(
            checkpoint["policy_net_state_dict"], self.policy_net
        )
        target_state = self._adapt_state_dict_for_compile(
            checkpoint["target_net_state_dict"], self.target_net
        )

        self.policy_net.load_state_dict(policy_state)
        self.target_net.load_state_dict(target_state)

    def _restore_core_checkpoint_state(self: Any, checkpoint: Dict[str, Any]) -> None:
        self._restore_checkpoint_network_weights(checkpoint)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self._learn_step = checkpoint.get("_learn_step", 0)
        self._next_target_update = checkpoint.get(
            "_next_target_update", self.steps + self.config.TARGET_UPDATE
        )

    def _load_checkpoint_metadata(self: Any, checkpoint: Dict[str, Any]) -> Optional[SaveMetadata]:
        if "metadata" not in checkpoint:
            return None
        try:
            return SaveMetadata.from_dict(checkpoint["metadata"])
        except Exception:
            return None

    def _load_training_history(
        self: Any, checkpoint: Dict[str, Any]
    ) -> Optional["TrainingHistory"]:
        if "training_history" not in checkpoint:
            return None
        try:
            return TrainingHistory.from_dict(checkpoint["training_history"])
        except Exception:
            return None

    def _load_replay_buffer(self: Any, checkpoint: Dict[str, Any], *, quiet: bool) -> bool:
        if "replay_buffer" not in checkpoint:
            return False
        try:
            return self.memory.load_from_dict(checkpoint["replay_buffer"])
        except Exception as e:
            if not quiet:
                print(f"⚠️ Could not restore replay buffer: {e}")
            return False

    def _format_saved_time(self: Any, metadata: SaveMetadata) -> str:
        save_time = datetime.fromisoformat(metadata.timestamp)
        time_ago = datetime.now() - save_time
        if time_ago.days > 0:
            time_str = f"{time_ago.days}d ago"
        elif time_ago.seconds > 3600:
            time_str = f"{time_ago.seconds // 3600}h ago"
        else:
            time_str = f"{time_ago.seconds // 60}m ago"
        return f"{save_time.strftime('%b %d, %Y %I:%M %p')} ({time_str})"

    def _print_resume_summary(
        self: Any,
        *,
        filepath: str,
        metadata: Optional[SaveMetadata],
        training_history: Optional["TrainingHistory"],
        replay_buffer_loaded: bool,
    ) -> None:
        file_size = os.path.getsize(filepath)
        size_mb = file_size / (1024 * 1024)

        print(f"\n{'='*60}")
        print("📂 Resuming Training")
        print(f"{'='*60}")
        print(f"   Model: {os.path.basename(filepath)} ({size_mb:.2f} MB)")

        if metadata:
            try:
                print(f"   Saved: {self._format_saved_time(metadata)}")
            except Exception:
                print(f"   Saved: {metadata.timestamp}")

            print(
                f"\n   Episode: {metadata.episode:,} | Steps: {metadata.total_steps:,} | ε: {metadata.epsilon:.4f}"
            )
            print(
                f"   Best Score: {metadata.best_score} | Avg(100): {metadata.avg_score_last_100:.1f}"
            )
            print(f"   Win Rate: {metadata.win_rate*100:.1f}% | Avg Loss: {metadata.avg_loss:.4f}")

            if metadata.total_training_time_seconds > 0:
                hours = int(metadata.total_training_time_seconds // 3600)
                minutes = int((metadata.total_training_time_seconds % 3600) // 60)
                print(f"   Previous Training Time: {hours}h {minutes}m")

            print(
                f"\n   Config: LR={metadata.learning_rate}, γ={metadata.gamma}, Batch={metadata.batch_size}"
            )
            print(f"   Architecture: {metadata.hidden_layers}")
        else:
            print(f"\n   Steps: {self.steps:,} | Epsilon: {self.epsilon:.4f}")
            print("   (Legacy save - no detailed metadata)")

        if training_history and len(training_history.scores) > 0:
            print(f"   Training History: {len(training_history.scores)} episodes restored")
        else:
            print("   Training History: Not available (older save format)")

        if replay_buffer_loaded:
            print(f"   Replay Buffer: {len(self.memory):,} experiences restored")
        else:
            print("   Replay Buffer: Starting fresh (not saved or incompatible)")

        print(f"{'='*60}\n")

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

        checkpoint = self._load_checkpoint_payload(filepath)
        if checkpoint is None:
            return None, None

        if not self._checkpoint_architecture_matches(checkpoint):
            if not quiet:
                self._print_architecture_mismatch(checkpoint)
            return None, None

        self._restore_core_checkpoint_state(checkpoint)
        metadata = self._load_checkpoint_metadata(checkpoint)
        training_history = self._load_training_history(checkpoint)
        replay_buffer_loaded = self._load_replay_buffer(checkpoint, quiet=quiet)

        if not quiet:
            self._print_resume_summary(
                filepath=filepath,
                metadata=metadata,
                training_history=training_history,
                replay_buffer_loaded=replay_buffer_loaded,
            )

        return metadata, training_history

    def load_weights_only(self: Any, filepath: str, quiet: bool = True) -> bool:
        """Load only the policy/target network weights from a checkpoint.

        Unlike :meth:`load`, this does NOT restore the optimizer, epsilon, step
        counters, metadata, or replay buffer — it swaps the network parameters in
        place while preserving the current training position. Used for "true
        rollback" to the eval-best checkpoint after early-stop so the live policy
        matches the kept-best one without rewinding the run.

        Returns True on success, False if the file is missing or incompatible.
        """
        if not os.path.exists(filepath):
            return False
        checkpoint = self._load_checkpoint_payload(
            filepath,
            quiet=quiet,
            error_prefix=(f"⚠️ Could not load weights from {os.path.basename(filepath)}"),
        )
        if checkpoint is None:
            return False

        if not self._checkpoint_architecture_matches(checkpoint):
            if not quiet:
                print("⚠️ Cannot load weights - architecture mismatch.")
            return False

        try:
            self._restore_checkpoint_network_weights(checkpoint)
        except (KeyError, RuntimeError) as e:
            if not quiet:
                print(f"⚠️ Could not load weights: {e}")
            return False

        return True

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
