"""CLI model inspection and listing commands."""

from __future__ import annotations

from src.ai.agent import Agent


def inspect_model(filepath: str) -> None:
    """Inspect a model file and display its metadata."""
    info = Agent.inspect_model(filepath)
    if not info:
        return

    print("\n" + "=" * 60)
    print(f"🔍 Model Inspection: {info['filename']}")
    print("=" * 60)
    print(f"   File Size: {info['file_size_mb']:.2f} MB")
    print(f"   Modified:  {info['file_modified']}")
    print(
        f"\n   Steps:     {info['steps']:,}"
        if isinstance(info["steps"], int)
        else f"\n   Steps:     {info['steps']}"
    )
    print(
        f"   Epsilon:   {info['epsilon']:.4f}"
        if isinstance(info["epsilon"], float)
        else f"   Epsilon:   {info['epsilon']}"
    )
    print(f"   State Size: {info['state_size']}")
    print(f"   Action Size: {info['action_size']}")

    if info["has_metadata"] and info["metadata"]:
        meta = info["metadata"]
        print("\n   📊 Training Metadata:")
        print("   ─────────────────────")
        print(f"   Save Reason:    {meta.get('save_reason', 'unknown')}")
        print(
            f"   Episode:        {meta.get('episode', 'unknown'):,}"
            if isinstance(meta.get("episode"), int)
            else f"   Episode:        {meta.get('episode', 'unknown')}"
        )
        print(f"   Best Score:     {meta.get('best_score', 'unknown')}")
        print(f"   Avg Score(100): {meta.get('avg_score_last_100', 0):.1f}")
        print(f"   Win Rate:       {meta.get('win_rate', 0)*100:.1f}%")
        print(f"   Avg Loss:       {meta.get('avg_loss', 0):.4f}")

        training_time = meta.get("total_training_time_seconds", 0)
        if training_time > 0:
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            print(f"   Training Time:  {hours}h {minutes}m")

        print("\n   ⚙️ Config Snapshot:")
        print("   ─────────────────────")
        print(f"   Learning Rate:  {meta.get('learning_rate', 'unknown')}")
        print(f"   Gamma:          {meta.get('gamma', 'unknown')}")
        print(f"   Batch Size:     {meta.get('batch_size', 'unknown')}")
        print(f"   Hidden Layers:  {meta.get('hidden_layers', 'unknown')}")
        print(f"   Dueling DQN:    {meta.get('use_dueling', 'unknown')}")
    else:
        print("\n   ⚠️ No detailed metadata (legacy save format)")

    print("=" * 60 + "\n")


def list_models(model_dir: str = "models") -> None:
    """List all model files in the models directory."""
    models = Agent.list_models(model_dir)

    if not models:
        print(f"\n❌ No model files found in '{model_dir}/'")
        return

    print("\n" + "=" * 80)
    print(f"📁 Saved Models in '{model_dir}/' ({len(models)} files)")
    print("=" * 80)
    print(f"{'Filename':<35} {'Episode':>8} {'Steps':>12} {'Best':>6} {'Epsilon':>8} {'Size':>8}")
    print("-" * 80)

    for model in models:
        filename = (
            model["filename"][:33] + ".." if len(model["filename"]) > 35 else model["filename"]
        )

        if model["has_metadata"] and model["metadata"]:
            meta = model["metadata"]
            episode = meta.get("episode", "?")
            steps = meta.get("total_steps", model.get("steps", "?"))
            best = meta.get("best_score", "?")
            epsilon = meta.get("epsilon", model.get("epsilon", "?"))
        else:
            episode = "?"
            steps = model.get("steps", "?")
            best = "?"
            epsilon = model.get("epsilon", "?")

        size_mb = f"{model['file_size_mb']:.1f}MB"
        ep_str = f"{episode:,}" if isinstance(episode, int) else str(episode)
        steps_str = f"{steps:,}" if isinstance(steps, int) else str(steps)
        best_str = str(best)
        eps_str = f"{epsilon:.3f}" if isinstance(epsilon, float) else str(epsilon)

        print(f"{filename:<35} {ep_str:>8} {steps_str:>12} {best_str:>6} {eps_str:>8} {size_mb:>8}")

    print("=" * 80)
    print("\nUse --inspect <path> to see detailed info about a specific model.\n")
