"""Network throughput micro-benchmark: CPU vs MPS (vs CUDA) crossover.

Purpose: replace folklore ("MPS is 8x slower") with real, current numbers for the
EXACT networks this project trains (614K-param dueling MLP, 798K flatten-CNN, 77K
global-pool CNN), so a device choice for the next architecture swing (RUN-14+) is
data-driven. Times a realistic train step: forward -> MSE(Q, target) -> backward ->
Adam.step(), which is what the real learner does.

THERMAL / CONTENTION HYGIENE (this is a laptop sharing one heat budget):
  * Each cell is TIME-BOXED to a few seconds of steady-state after a warmup, so the
    whole sweep is a short burst, not a sustained soak that would heat-soak a
    concurrently-running training job.
  * Each cell is measured as TWO back-to-back windows (early vs late). If the late
    window is materially slower, that is the thermal-throttle / CPU-contention
    signature and the cell is flagged -- so you can SEE throttling instead of
    silently baking it into the number.
  * Run the GPU leg (`--device mps`) while a CPU training job is live: it uses the
    otherwise-idle GPU and barely touches the CPU. Run the matched `--device cpu`
    leg only when the box is IDLE (e.g. after the run finishes) -- a CPU baseline
    measured under a saturated CPU is meaningless.

Usage:
    # GPU leg now (safe alongside a CPU training run; uses the idle GPU):
    python -m experiments.cc_status.bench_net --device mps  --out scratchpad/bench_mps.json
    # CPU leg later, on an IDLE box:
    python -m experiments.cc_status.bench_net --device cpu  --out scratchpad/bench_cpu.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import Config  # noqa: E402
from src.ai.network import DuelingDQN, SpatialDQN  # noqa: E402

# State geometry of the tutorial rich state (window 11x19 + gmap 6x11 + meta 20).
# Matches CrystalCaves defaults; hardcoded so the bench builds in <1s without the
# (slow) game/agent construction path.
STATE_SIZE = 295
STATE_LAYOUT = {"window": (11, 19), "gmap": (6, 11), "meta": 20}
ACTION_SIZE = 10


def _make_config(global_pool: bool) -> Config:
    c = Config()
    c.STATE_LAYOUT = dict(STATE_LAYOUT)
    c.CRYSTAL_CAVES_CNN_GLOBAL_POOL = global_pool
    return c


def build_model(kind: str) -> torch.nn.Module:
    if kind == "mlp":
        return DuelingDQN(state_size=STATE_SIZE, action_size=ACTION_SIZE, config=_make_config(False))
    if kind == "cnn":
        return SpatialDQN(state_size=STATE_SIZE, action_size=ACTION_SIZE, config=_make_config(False))
    if kind == "cnn_gap":
        return SpatialDQN(state_size=STATE_SIZE, action_size=ACTION_SIZE, config=_make_config(True))
    raise ValueError(f"unknown model kind: {kind}")


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _timed_window(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    device: str,
    seconds: float,
) -> tuple[int, float]:
    """Run train steps for ~`seconds` wall-time; return (n_steps, elapsed)."""
    _sync(device)
    start = time.perf_counter()
    steps = 0
    while time.perf_counter() - start < seconds:
        opt.zero_grad(set_to_none=True)
        q = model(x)
        loss = F.mse_loss(q, target)
        loss.backward()
        opt.step()
        steps += 1
    _sync(device)
    return steps, time.perf_counter() - start


def bench_cell(
    kind: str, batch: int, device: str, *, warmup: float, seconds: float
) -> dict[str, Any]:
    model = build_model(kind).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(batch, STATE_SIZE, device=device)
    target = torch.randn(batch, ACTION_SIZE, device=device)

    # Warmup (lazy kernel compile / clock spin-up), discarded.
    _timed_window(model, opt, x, target, device, warmup)
    # Two back-to-back windows: early vs late exposes throttling/contention.
    s1, t1 = _timed_window(model, opt, x, target, device, seconds)
    s2, t2 = _timed_window(model, opt, x, target, device, seconds)
    early = s1 / t1
    late = s2 / t2
    drift = (late - early) / early if early else 0.0
    return {
        "model": kind,
        "params": n_params,
        "batch": batch,
        "device": device,
        "steps_per_s_early": round(early, 1),
        "steps_per_s_late": round(late, 1),
        "ms_per_step": round(1000.0 / late, 3) if late else None,
        "throttle_drift_pct": round(100.0 * drift, 1),
        "throttled": drift < -0.10,  # late >10% slower than early
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--models", default="mlp,cnn,cnn_gap")
    p.add_argument("--batches", default="128,256,512,1024")
    p.add_argument("--warmup", type=float, default=1.5, help="warmup seconds per cell")
    p.add_argument("--seconds", type=float, default=3.0, help="timed seconds per window (x2)")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=leave default)")
    p.add_argument("--out", default="")
    args = p.parse_args()

    if args.device == "mps" and not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
        print("MPS not available on this build/host.", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available on this host.", file=sys.stderr)
        return 2
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    rows: list[dict[str, Any]] = []
    print(f"device={args.device} threads={torch.get_num_threads()} "
          f"warmup={args.warmup}s window={args.seconds}sx2\n")
    header = f"{'model':9s} {'params':>9s} {'batch':>5s} {'steps/s(early)':>14s} {'steps/s(late)':>13s} {'ms/step':>8s} {'drift%':>7s} flag"
    print(header)
    print("-" * len(header))
    for kind in models:
        for batch in batches:
            r = bench_cell(kind, batch, args.device, warmup=args.warmup, seconds=args.seconds)
            rows.append(r)
            flag = "THROTTLE?" if r["throttled"] else ""
            print(f"{r['model']:9s} {r['params']:>9,d} {r['batch']:>5d} "
                  f"{r['steps_per_s_early']:>14.1f} {r['steps_per_s_late']:>13.1f} "
                  f"{r['ms_per_step']:>8.3f} {r['throttle_drift_pct']:>7.1f} {flag}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"device": args.device, "rows": rows}, indent=2))
        print(f"\nwrote {out}")
    if any(r["throttled"] for r in rows):
        print("\n⚠️  Some cells slowed >10% across windows — likely thermal throttle or CPU "
              "contention. Re-run idle / cooled, or trust the EARLY column as the headroom number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
