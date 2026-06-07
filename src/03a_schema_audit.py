"""
03a — Schema Audit
==================
Blindly inspects a sample of match, player (mastery), and timeline JSON files
and produces a structured text report summarising:
  • What top-level sections exist
  • All field names, their Python types, and example values
  • Numeric fields: min / mean / max over the sample
  • Categorical fields: unique values (up to 20)
  • Nested structures: recursively expanded

Output: reports/schema_audit.txt

Usage
-----
    conda activate lol_shap_env
    python src/03a_schema_audit.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils import ensure_dir, PROJECT_ROOT

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_N    = 200           # number of files to randomly sample per type
MATCH_DIR   = PROJECT_ROOT / "data" / "raw" / "matches"
PLAYER_DIR  = PROJECT_ROOT / "data" / "raw" / "players"
TIMELINE_DIR= PROJECT_ROOT / "data" / "raw" / "timelines"
REPORT_DIR  = ensure_dir(PROJECT_ROOT / "reports")
REPORT_PATH = REPORT_DIR / "schema_audit.txt"

random.seed(42)


# ── Core Utilities ────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception as e:
        return None


def sample_files(directory: Path, pattern: str, n: int) -> list[Path]:
    files = list(directory.glob(pattern))
    if len(files) <= n:
        return files
    return random.sample(files, n)


# ── Field Statistics Accumulator ──────────────────────────────────────────────

class FieldStats:
    """Accumulates type and value information across many records."""

    def __init__(self):
        self.count = 0
        self.types: set[str] = set()
        self.numerics: list[float] = []
        self.categoricals: set[str] = set()
        self.has_nested = False

    def update(self, val: Any):
        self.count += 1
        t = type(val).__name__
        self.types.add(t)
        if isinstance(val, (int, float)):
            self.numerics.append(float(val))
        elif isinstance(val, bool):
            # bools are ints in Python – override
            self.categoricals.add(str(val))
        elif isinstance(val, str):
            self.categoricals.add(val)
        elif isinstance(val, (dict, list)):
            self.has_nested = True

    def summary(self) -> str:
        parts = [f"types={{{','.join(sorted(self.types))}}}  n={self.count}"]
        if self.numerics:
            mn, mx, avg = min(self.numerics), max(self.numerics), sum(self.numerics) / len(self.numerics)
            parts.append(f"range=[{mn:.2g}, {mx:.2g}]  mean={avg:.2g}")
        elif self.categoricals:
            cats = sorted(self.categoricals)[:20]
            ellipsis = "…" if len(self.categoricals) > 20 else ""
            parts.append(f"values=[{', '.join(cats)}{ellipsis}]")
        if self.has_nested:
            parts.append("(contains nested dict/list)")
        return "  |  ".join(parts)


def flatten_record(obj: Any, stats: dict[str, FieldStats], prefix: str = "") -> None:
    """Recursively walk obj and accumulate field stats."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if key not in stats:
                stats[key] = FieldStats()
            stats[key].update(v)
            if isinstance(v, dict):
                flatten_record(v, stats, prefix=key)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # Only flatten the first item of homogeneous lists
                flatten_record(v[0], stats, prefix=f"{key}[0]")
    elif isinstance(obj, list):
        for item in obj:
            flatten_record(item, stats, prefix=prefix)


# ── Section Writers ───────────────────────────────────────────────────────────

def write_section(out, title: str, stats: dict[str, FieldStats]) -> None:
    out.write(f"\n{'='*80}\n")
    out.write(f"  {title}\n")
    out.write(f"{'='*80}\n")
    for field in sorted(stats.keys()):
        out.write(f"\n  [{field}]\n    {stats[field].summary()}\n")


# ── Per-Section Collectors ────────────────────────────────────────────────────

def audit_matches(files: list[Path]) -> dict[str, FieldStats]:
    """Collect stats from match['info']['participants'][i] across all files."""
    stats: dict[str, FieldStats] = {}
    for path in files:
        data = load_json(path)
        if not data:
            continue
        for p in data.get("info", {}).get("participants", []):
            flatten_record(p, stats)
    return stats


def audit_match_meta(files: list[Path]) -> dict[str, FieldStats]:
    """Stats from the match-level info block (not participants)."""
    stats: dict[str, FieldStats] = {}
    for path in files:
        data = load_json(path)
        if not data:
            continue
        info = {k: v for k, v in data.get("info", {}).items() if k != "participants"}
        flatten_record(info, stats)
    return stats


def audit_timeline_frames(files: list[Path]) -> dict[str, FieldStats]:
    """Stats from participantFrames at each minute, aggregated across files."""
    stats: dict[str, FieldStats] = {}
    for path in files:
        data = load_json(path)
        if not data:
            continue
        frames = data.get("info", {}).get("frames", [])
        for frame in frames:
            for pid, pf in frame.get("participantFrames", {}).items():
                flatten_record(pf, stats)
    return stats


def audit_players(files: list[Path]) -> dict[str, FieldStats]:
    """Stats from player mastery lists."""
    stats: dict[str, FieldStats] = {}
    for path in files:
        data = load_json(path)
        if not isinstance(data, list):
            continue
        for entry in data:
            flatten_record(entry, stats)
    return stats


def audit_game_meta(match_files: list[Path]) -> dict:
    """High-level game-level summary stats."""
    durations, n_participants, game_versions = [], [], set()
    for path in match_files:
        data = load_json(path)
        if not data:
            continue
        info = data.get("info", {})
        durations.append(info.get("gameDuration", 0))
        n_participants.append(len(info.get("participants", [])))
        game_versions.add(info.get("gameVersion", "?"))
    return {
        "game_count": len(durations),
        "duration_min_s": min(durations) if durations else 0,
        "duration_max_s": max(durations) if durations else 0,
        "duration_mean_s": sum(durations) / len(durations) if durations else 0,
        "participants_per_game_values": sorted(set(n_participants)),
        "game_versions_seen": sorted(game_versions)[:10],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Sampling {SAMPLE_N} files from each category...")

    match_files   = sample_files(MATCH_DIR,    "match_*.json",    SAMPLE_N)
    player_files  = sample_files(PLAYER_DIR,   "player_*.json",   SAMPLE_N)
    timeline_files= sample_files(TIMELINE_DIR, "timeline_*.json", SAMPLE_N)

    print(f"  match files    : {len(match_files)}")
    print(f"  player files   : {len(player_files)}")
    print(f"  timeline files : {len(timeline_files)}")

    print("Auditing match participant fields...")
    match_part_stats = audit_matches(match_files)
    print("Auditing match meta fields...")
    match_meta_stats = audit_match_meta(match_files)
    print("Auditing timeline per-minute participant frames...")
    timeline_stats = audit_timeline_frames(timeline_files)
    print("Auditing player mastery fields...")
    player_stats = audit_players(player_files)
    print("Computing game-level summary...")
    game_meta = audit_game_meta(match_files)

    print(f"Writing report to {REPORT_PATH} ...")

    with open(REPORT_PATH, "w", encoding="utf-8") as out:
        out.write("LoL Win-Contribution Pipeline — Schema Audit Report\n")
        out.write(f"Sampled {SAMPLE_N} files per category.\n")
        out.write("\n")

        # ── Game-level summary ─────────────────────────────────────────────
        out.write("="*80 + "\n")
        out.write("  GAME-LEVEL SUMMARY\n")
        out.write("="*80 + "\n")
        for k, v in game_meta.items():
            out.write(f"  {k}: {v}\n")

        # ── Match info (game-wide) ─────────────────────────────────────────
        write_section(out, "MATCH INFO (game-level metadata — info.* excluding participants)", match_meta_stats)

        # ── Match Participants ─────────────────────────────────────────────
        write_section(out, "MATCH PARTICIPANT FIELDS (info.participants[i].*)", match_part_stats)

        # ── Timeline ──────────────────────────────────────────────────────
        write_section(out, "TIMELINE — PER-MINUTE PARTICIPANT FRAME FIELDS", timeline_stats)

        # ── Players ───────────────────────────────────────────────────────
        write_section(out, "PLAYER MASTERY FIELDS (per champion entry in mastery list)", player_stats)

        # ── Recommendations ───────────────────────────────────────────────
        out.write("\n" + "="*80 + "\n")
        out.write("  RECOMMENDATIONS FOR FEATURE ENGINEERING\n")
        out.write("="*80 + "\n")
        out.write("""
MATCH-LEVEL FEATURES (from info.participants)
---------------------------------------------
These produce one feature vector per player, per match. They are end-of-game
stats so they are useful as labels / validation but NOT as model inputs
(they'd leak the outcome). Exceptions that are safe as inputs:
  • championId / championName — what champion was played.
  • teamId (100 blue / 200 red) — team assignment.
  • individualPosition / teamPosition — role (TOP, JUNGLE, MID, BOTTOM, UTILITY).
  • summoner1Id / summoner2Id — summoner spells (can indicate playstyle).
  • perks.* — rune choices.
  • info.gameDuration — game length (available only for training, not at inference
    time before the game ends).

TIMELINE FEATURES (from participantFrames at minute T)
------------------------------------------------------
These are the core minute-by-minute model inputs. Each frame gives:
  Gold  : totalGold, currentGold, goldPerSecond
  XP    : xp, level
  CS    : minionsKilled, jungleMinionsKilled
  Stats : championStats.* (AD, AP, armor, MR, MoveSpeed, HP, etc.)
  Damage: damageStats.* (magic/physical/true dealt/taken to/from champs)
  Map   : position.x, position.y (can compute distance from objectives)
  Misc  : timeEnemySpentControlled (CC impact)

  DERIVED FEATURES to compute:
  • Gold differential vs. lane opponent (Blue_top_gold - Red_top_gold)
  • XP differential
  • CS differential
  • Damage differential
  • Net gold lead per role
  • Current Gold / Total Gold ratio (resource efficiency)

PLAYER MASTERY FEATURES (from mastery list)
-------------------------------------------
Champion Mastery V4 provides per-champion:
  • championId
  • championPoints — proxy for experience on champion (higher = more experienced)
  • championPointsSinceLastLevel
  • championPointsUntilNextLevel
  • lastPlayTime — Unix ms timestamp of last game on that champ
  • championLevel — mastery level (1-21 in current system)
  • tokensEarned
  
  NOTE: winrate and games-played-on-champion are NOT natively provided.
        As a smurf indicator, championPoints is the best proxy available.
  
  Useful engineered features:
  • played_champion_mastery_points — mastery on the specific champion
    they played in this match (join mastery list by championId).
  • played_champion_mastery_level — mastery level on that champion.
  • days_since_last_played — staleness of champion experience.
  • is_champion_in_top_3_mastery — boolean, are they playing a main?

NOTES ON JOINING DATA
----------------------
Match and Timeline share the same matchId.
Player PUUID appears in both match.metadata.participants (list, ordered)
and match.info.participants[i].puuid (object). They are the same list order.
The participantId in the timeline (1-10) maps to match.info.participants
by index (participantId 1 = participants[0], etc.).
""")

    print(f"\n✅ Report written to {REPORT_PATH}")
    print("   Open reports/schema_audit.txt in any text editor to review.")


if __name__ == "__main__":
    main()
