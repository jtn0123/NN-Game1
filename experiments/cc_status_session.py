# ruff: noqa: F401,F403,F405,I001
"""Compatibility wrapper for Crystal Caves status-session experiments.

The implementation lives in experiments.cc_status.* so each module stays small.
This file remains the public import path and command entrypoint:
    python experiments/cc_status_session.py <mode> ...
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

if len(sys.argv) > 1 and sys.argv[1] == "list-recipes":
    from experiments.cc_status.recipes import format_recipe_list  # noqa: E402

    print(format_recipe_list())
    raise SystemExit(0)

if len(sys.argv) > 1 and sys.argv[1] == "run-recipe":
    from experiments.cc_status.recipes import expand_recipe_argv  # noqa: E402

    try:
        sys.argv = list(expand_recipe_argv(sys.argv))
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

if len(sys.argv) > 1 and sys.argv[1] == "compare-artifact":
    from experiments.cc_status.promotion import promotion_main  # noqa: E402

    raise SystemExit(promotion_main(sys.argv[2:]))

if len(sys.argv) > 1 and sys.argv[1] == "metric-audit":
    from experiments.cc_status.metric_audit import metric_audit_main  # noqa: E402

    raise SystemExit(metric_audit_main(sys.argv[2:]))

if len(sys.argv) > 1 and sys.argv[1] == "paired-ab":
    from experiments.cc_status.paired_ab import paired_ab_main  # noqa: E402

    raise SystemExit(paired_ab_main(sys.argv[2:]))

from experiments.cc_status.config_helpers import *  # noqa: F401,F403,E402
from experiments.cc_status.artifacts import *  # noqa: F401,F403,E402
from experiments.cc_status.contact_head import *  # noqa: F401,F403,E402
from experiments.cc_status.contact_label_audit import *  # noqa: F401,F403,E402
from experiments.cc_status.cli_label_modes import *  # noqa: F401,F403,E402
from experiments.cc_status.corrections import *  # noqa: F401,F403,E402
from experiments.cc_status.demo_collect import *  # noqa: F401,F403,E402
from experiments.cc_status.demo_planners import *  # noqa: F401,F403,E402
from experiments.cc_status.evals import *  # noqa: F401,F403,E402
from experiments.cc_status.io_utils import *  # noqa: F401,F403,E402
from experiments.cc_status.metric_audit import *  # noqa: F401,F403,E402
from experiments.cc_status.paired_ab import *  # noqa: F401,F403,E402
from experiments.cc_status.promotion import *  # noqa: F401,F403,E402
from experiments.cc_status.recipes import *  # noqa: F401,F403,E402
from experiments.cc_status.reports import *  # noqa: F401,F403,E402
from experiments.cc_status.scorecard import *  # noqa: F401,F403,E402
from experiments.cc_status.runs_baseline import *  # noqa: F401,F403,E402
from experiments.cc_status.runs_demo import *  # noqa: F401,F403,E402
from experiments.cc_status.runs_mixed import *  # noqa: F401,F403,E402
from experiments.cc_status.runs_route import *  # noqa: F401,F403,E402
from experiments.cc_status.runs_transfer import *  # noqa: F401,F403,E402
from experiments.cc_status.snapshots import *  # noqa: F401,F403,E402
from experiments.cc_status.training import *  # noqa: F401,F403,E402
from experiments.cc_status.vec_envs import *  # noqa: F401,F403,E402
from experiments.cc_status.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
