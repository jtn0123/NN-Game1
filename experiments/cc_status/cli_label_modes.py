"""CLI dispatch for offline contact-label dataset modes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .contact_label_audit import run_contact_label_audit, run_contact_label_filter
from .correction_calibration import parse_correction_dataset_paths


def run_label_dataset_mode(
    parser: argparse.ArgumentParser,
    opts: argparse.Namespace,
    out_dir: Path,
    payload: dict[str, Any],
) -> bool:
    if opts.mode == "contact-label-audit":
        correction_dataset_paths = parse_correction_dataset_paths(opts.correction_datasets)
        if not correction_dataset_paths:
            parser.error("contact-label-audit requires --correction-datasets")
        payload["runs"].append(
            run_contact_label_audit(
                out_dir,
                correction_dataset_paths=correction_dataset_paths,
                state_round_decimals=opts.contact_label_state_decimals,
                adjacent_step_window=opts.contact_label_adjacent_step_window,
                top_groups=opts.contact_label_top_groups,
                label=opts.label or "contact_label_audit",
            )
        )
        return True
    if opts.mode == "contact-label-filter":
        correction_dataset_paths = parse_correction_dataset_paths(opts.correction_datasets)
        if not correction_dataset_paths:
            parser.error("contact-label-filter requires --correction-datasets")
        payload["runs"].append(
            run_contact_label_filter(
                out_dir,
                correction_dataset_paths=correction_dataset_paths,
                semantic_majority_threshold=opts.contact_label_filter_majority_threshold,
                adjacent_step_window=opts.contact_label_adjacent_step_window,
                label=opts.label or "contact_label_filter",
            )
        )
        return True
    return False
