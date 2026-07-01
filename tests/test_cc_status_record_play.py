import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from experiments.cc_status.record_play import record_policy_play  # noqa: E402


def test_record_policy_play_writes_a_valid_labeled_gif(tmp_path):
    # A NOOP agent stalls; the recorder must render the greedy rollout to a real, openable
    # GIF labeled by end reason. Validates the headless render -> Pillow GIF pipeline end to
    # end (the piece that lets us actually WATCH the agent play).
    from PIL import Image

    cfg = Config()
    cfg.CRYSTAL_CAVES_DIFFICULTY = "normal"
    cfg.EVAL_MAX_STEPS = 800  # > the 720 stall threshold; keeps the test short
    agent = SimpleNamespace(select_action=lambda state, training=False: 0)  # NOOP

    saved = record_policy_play(
        agent,
        cfg,
        games=2,
        out_dir=tmp_path,
        max_gifs=2,
        capture_games=2,
        max_frames=40,
    )

    assert saved, "expected at least one recorded gif"
    for entry in saved:
        assert entry["end_reason"] == "stalled"  # the motionless agent stalls
        path = entry["path"]
        assert path.endswith(".gif")
        assert entry["end_reason"] in path  # filename is labeled by outcome
        with Image.open(path) as img:
            assert img.format == "GIF"
            assert img.n_frames >= 2  # an animation, not a single frame
            assert img.width > 0 and img.height > 0
