"""
03b — Event Distributions
=========================
Generates two sets of visualisations over the full timeline corpus:

  Figure 1  — Per-event-type histogram
      X-axis: total events of that type per match
      Y-axis: frequency (count of matches)
      One subplot per event type, saved as reports/event_count_histograms.png

  Figure 2  — Cumulative events over game time
      X-axis: minute (frame index)
      Y-axis: cumulative event count
      One subplot per event type (+ one "ALL EVENTS" combined)
      Each match is a semi-transparent grey line.
      Bold coloured line = mean. Shaded band = mean ± 1 std.
      Saved as reports/event_cumulative_timeseries.png

Usage
-----
    conda activate lol_shap_env
    python src/03b_event_distributions.py

Notes
-----
  Processes up to MAX_FILES timelines (default = all).
  Progress is printed every 1000 files.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir, PROJECT_ROOT

# ── Config ────────────────────────────────────────────────────────────────────
TIMELINE_DIR = PROJECT_ROOT / "data" / "raw" / "timelines"
REPORT_DIR   = ensure_dir(PROJECT_ROOT / "reports")
MAX_FILES    = None          # set to e.g. 5000 to limit for quick testing
RANDOM_SEED  = 42
GREY_ALPHA   = 0.03          # opacity for individual match lines
MAX_GREY_LINES = 3000        # cap how many grey lines we draw (visual limit)

# Which event types to include in individual-type plots.
# We ignore very noisy / irrelevant events.
SKIP_EVENT_TYPES = {"PAUSE_END", "OBJECTIVE_BOUNTY_PRESTART", "OBJECTIVE_BOUNTY_FINISH", "GAME_END"}

# DPI / figure sizes
FIG_DPI = 150
HIST_FIG_SIZE  = (22, 14)
TS_FIG_SIZE    = (26, 18)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_json(path: Path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


def iter_timeline_files(directory: Path, max_files: int | None, seed: int):
    files = sorted(directory.glob("timeline_*.json"))
    if max_files and len(files) > max_files:
        rng = random.Random(seed)
        files = rng.sample(files, max_files)
    return files


# ── Data Collection ───────────────────────────────────────────────────────────

def collect_event_data(files: list[Path]) -> tuple[
    dict[str, list[int]],        # event_type -> list of total counts per match
    dict[str, list[list[int]]],  # event_type -> list of per-minute cumulative counts
    int,                         # max frames seen
]:
    """
    Returns:
      per_match_counts[event_type] = [count_match1, count_match2, ...]
      cumulative_series[event_type] = [[0,0,1,2,...], [0,1,1,...], ...]  (per match)
      max_frames: the longest game length in frames
    """
    per_match_counts: dict[str, list[int]] = defaultdict(list)
    cumulative_series: dict[str, list[list[int]]] = defaultdict(list)
    max_frames = 0

    n = len(files)
    for i, path in enumerate(files):
        if i % 1000 == 0:
            print(f"  Processing file {i}/{n} …")

        data = load_json(path)
        if not data:
            continue
        frames = data.get("info", {}).get("frames", [])
        if not frames:
            continue
        max_frames = max(max_frames, len(frames))

        # Count events per frame per type
        frame_event_counts: list[dict[str, int]] = []
        for frame in frames:
            counts: dict[str, int] = defaultdict(int)
            for event in frame.get("events", []):
                etype = event.get("type", "UNKNOWN")
                if etype not in SKIP_EVENT_TYPES:
                    counts[etype] += 1
            frame_event_counts.append(counts)

        # Gather all event types seen in this match
        all_types_in_match = set()
        for fc in frame_event_counts:
            all_types_in_match.update(fc.keys())

        # Build cumulative series per type for this match
        for etype in all_types_in_match:
            cumsum = 0
            series = []
            for fc in frame_event_counts:
                cumsum += fc.get(etype, 0)
                series.append(cumsum)
            per_match_counts[etype].append(cumsum)     # total for histogram
            cumulative_series[etype].append(series)    # time series for plot

    return per_match_counts, cumulative_series, max_frames


# ── Figure 1: Histograms ──────────────────────────────────────────────────────

def plot_event_histograms(per_match_counts: dict[str, list[int]], out_path: Path) -> None:
    # Sort event types by median count descending for consistent layout
    types = sorted(per_match_counts.keys(), key=lambda t: -np.median(per_match_counts[t]))

    ncols = 4
    nrows = -(-len(types) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=HIST_FIG_SIZE)
    axes = axes.flatten()

    fig.suptitle(
        f"Distribution of Event Counts per Match  (n={len(next(iter(per_match_counts.values())))} matches)",
        fontsize=15, fontweight="bold", y=1.01
    )

    palette = plt.cm.tab20.colors

    for idx, etype in enumerate(types):
        ax = axes[idx]
        counts = per_match_counts[etype]
        color = palette[idx % len(palette)]
        ax.hist(counts, bins=40, color=color, edgecolor="none", alpha=0.85)
        ax.set_title(etype.replace("_", " ").title(), fontsize=8, fontweight="bold")
        ax.set_xlabel("Events per match", fontsize=7)
        ax.set_ylabel("# matches", fontsize=7)
        ax.tick_params(labelsize=6)
        med = np.median(counts)
        ax.axvline(med, color="black", linewidth=1.2, linestyle="--", label=f"median={med:.0f}")
        ax.legend(fontsize=6)

    # Hide unused axes
    for ax in axes[len(types):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_path.name}")


# ── Figure 2: Cumulative Time-Series ─────────────────────────────────────────

def pad_series(series: list[int], length: int) -> list[int]:
    """Extend a series to `length` by repeating its last value."""
    if len(series) >= length:
        return series[:length]
    return series + [series[-1]] * (length - len(series))


def plot_cumulative_timeseries(
    cumulative_series: dict[str, list[list[int]]],
    max_frames: int,
    out_path: Path,
) -> None:
    types = sorted(cumulative_series.keys(), key=lambda t: -np.median([s[-1] for s in cumulative_series[t]]))

    ncols = 4
    nrows = -(-len(types) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=TS_FIG_SIZE)
    axes = axes.flatten()

    fig.suptitle(
        "Cumulative Events Over Game Time — Grey=Individual Match, Bold=Mean ± 1 Std",
        fontsize=14, fontweight="bold", y=1.01
    )

    palette = plt.cm.tab10.colors
    minutes = np.arange(max_frames)

    for idx, etype in enumerate(types):
        ax = axes[idx]
        color = palette[idx % len(palette)]

        all_series = cumulative_series[etype]

        # Pad all series to max_frames
        padded = np.array([pad_series(s, max_frames) for s in all_series], dtype=float)

        # Grey individual lines (capped at MAX_GREY_LINES for visual clarity)
        sample_idx = list(range(len(padded)))
        if len(sample_idx) > MAX_GREY_LINES:
            rng = random.Random(RANDOM_SEED)
            sample_idx = rng.sample(sample_idx, MAX_GREY_LINES)

        for j in sample_idx:
            ax.plot(minutes[:len(all_series[j])], all_series[j],
                    color="grey", alpha=GREY_ALPHA, linewidth=0.4)

        # Mean and std
        mean_ = padded.mean(axis=0)
        std_  = padded.std(axis=0)

        ax.plot(minutes, mean_, color=color, linewidth=2.2, label="Mean")
        ax.fill_between(minutes, mean_ - std_, mean_ + std_,
                        color=color, alpha=0.25, label="±1 Std")

        ax.set_title(etype.replace("_", " ").title(), fontsize=8, fontweight="bold")
        ax.set_xlabel("Minute (frame)", fontsize=7)
        ax.set_ylabel("Cumulative count", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for ax in axes[len(types):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    files = iter_timeline_files(TIMELINE_DIR, MAX_FILES, RANDOM_SEED)
    print(f"Processing {len(files)} timeline files …")

    per_match_counts, cumulative_series, max_frames = collect_event_data(files)

    print(f"\nFound {len(per_match_counts)} event types. Max game length: {max_frames} frames.")
    print("Generating histogram figure …")
    plot_event_histograms(per_match_counts, REPORT_DIR / "event_count_histograms.png")

    print("Generating cumulative time-series figure …")
    plot_cumulative_timeseries(cumulative_series, max_frames, REPORT_DIR / "event_cumulative_timeseries.png")

    print("\n✅ Done. Reports saved to reports/")


if __name__ == "__main__":
    main()
