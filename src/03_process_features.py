"""
03_process_features.py
======================
Feature engineering pipeline. Reads raw match + timeline + player JSONs and
produces a single Parquet file with one row per (game_id, minute).

Column groups
-------------
  Metadata      : game_id, minute, blue_win, game_duration_min
  Pre-match      : {slot}_{stat} — champion, summoner spells, runes, mastery  (80 cols)
  Timeline       : {slot}_{stat} — gold, xp, cs, stats, damage, position      (290 cols)
  Differentials  : diff_{role}_{stat}, team-level totals                        (30 cols)
  Delta/rate     : {slot}_gold_delta_1m/3m, xp/cs/dmg_delta_1m,               (~77 cols)
                   gold_per_min, cs_per_min per slot; team gold-delta diffs

  Total: ~480 feature columns + 4 metadata columns.

Usage
-----
    conda activate lol_shap_env
    python src/03_process_features.py              # full run
    python src/03_process_features.py --limit 500  # test on 500 games

Output
------
    data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCH_DIR    = PROJECT_ROOT / "data" / "raw" / "matches"
TIMELINE_DIR = PROJECT_ROOT / "data" / "raw" / "timelines"
PLAYERS_DIR  = PROJECT_ROOT / "data" / "raw" / "players"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── Constants ─────────────────────────────────────────────────────────────────

# Fixed 10-slot ordering: blue team first (teamId=100), then red (teamId=200).
# Within each team: TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY.
ROLES = ["top", "jungle", "middle", "bottom", "utility"]
TEAM_ROLE_TO_SLOT: dict[tuple[int, str], str] = {
    (100, "TOP"):     "blue_top",
    (100, "JUNGLE"):  "blue_jungle",
    (100, "MIDDLE"):  "blue_middle",
    (100, "BOTTOM"):  "blue_bottom",
    (100, "UTILITY"): "blue_utility",
    (200, "TOP"):     "red_top",
    (200, "JUNGLE"):  "red_jungle",
    (200, "MIDDLE"):  "red_middle",
    (200, "BOTTOM"):  "red_bottom",
    (200, "UTILITY"): "red_utility",
}
SLOT_NAMES = [
    "blue_top", "blue_jungle", "blue_middle", "blue_bottom", "blue_utility",
    "red_top",  "red_jungle",  "red_middle",  "red_bottom",  "red_utility",
]

VALID_POSITIONS = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}

# Region encoding: extracted from game_id prefix (e.g. "NA1_XXXXX" -> 0).
# Unknown regions are assigned -1 and kept in the data for transparency.
REGION_INT_MAP: dict[str, int] = {"na1": 0, "euw1": 1, "kr": 2}

# Map size for position normalisation (SR map is ~14870 × 14870 units).
MAP_SIZE = 14_870.0

# Games shorter than this (seconds) are likely remakes — skip them.
MIN_GAME_DURATION_S = 900  # 15 minutes

# Number of game-rows (not game count) to buffer before writing a Parquet chunk.
# Each row ≈ 400 columns × 4 bytes ≈ 1.6 KB  →  50 000 rows ≈ 80 MB per chunk.
ROW_CHUNK_SIZE = 50_000

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "03_process_features.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Mastery cache ─────────────────────────────────────────────────────────────
# Lazy-loaded dict: puuid → {championId → mastery_entry} | None (file absent).
# Challenger/GM players appear in many games, so the cache hit-rate is high.
# Cap at MAX_CACHE_PLAYERS to avoid OOM on large datasets (>100K games).
_mastery_cache: dict[str, Optional[dict[int, dict]]] = {}
MAX_CACHE_PLAYERS = 8_000   # ~65 KB raw × 8K × 4x Python overhead ≈ 2 GB RAM


def _get_mastery_entry(puuid: str, champion_id: int) -> Optional[dict]:
    """Return the mastery entry for *puuid* on *champion_id*, or None."""
    if puuid not in _mastery_cache:
        # Stop caching once we hit the cap — just read and discard
        if len(_mastery_cache) >= MAX_CACHE_PLAYERS:
            path = PLAYERS_DIR / f"player_{puuid}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                champ_map = {e["championId"]: e for e in data}
                return champ_map.get(champion_id)
            return None

        path = PLAYERS_DIR / f"player_{puuid}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            _mastery_cache[puuid] = {e["championId"]: e for e in data}
        else:
            _mastery_cache[puuid] = None  # file absent — mark so we don't retry

    champ_map = _mastery_cache[puuid]
    if champ_map is None:
        return None
    return champ_map.get(champion_id)


def _mastery_features(puuid: str, champion_id: int, game_creation_ms: int) -> tuple[int, int, float]:
    """Return (mastery_points, mastery_level, days_since_last_played)."""
    entry = _get_mastery_entry(puuid, champion_id)
    if entry is None:
        return 0, 0, 0.0

    points = int(entry.get("championPoints", 0))
    level  = int(entry.get("championLevel",  0))

    last_play_ms = entry.get("lastPlayTime", game_creation_ms)
    # Positive = the player had a gap before this game; clip to [0, 3650] days.
    days = max(0.0, (game_creation_ms - last_play_ms) / 86_400_000.0)
    days = min(days, 3650.0)

    return points, level, days


# ── Role/slot mapping ─────────────────────────────────────────────────────────

def _build_role_map(participants: list[dict]) -> Optional[dict[int, str]]:
    """
    Map participantId (1-10) → slot name (e.g. "blue_top").

    Returns None if the game has missing / duplicate / invalid positions.
    """
    if len(participants) != 10:
        return None

    role_map: dict[int, str] = {}
    seen_slots: set[str] = set()

    for p in participants:
        pid     = p.get("participantId")
        team_id = p.get("teamId")
        pos     = p.get("teamPosition", "")

        if pos not in VALID_POSITIONS:
            return None  # blank or undetected role

        slot = TEAM_ROLE_TO_SLOT.get((team_id, pos))
        if slot is None or slot in seen_slots:
            return None  # duplicate or unknown (team_id, position) combo

        seen_slots.add(slot)
        role_map[pid] = slot

    return role_map if len(role_map) == 10 else None


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_prematch(
    participants: list[dict],
    role_map: dict[int, str],
    game_creation_ms: int,
) -> dict:
    """Return a flat dict of pre-match features for all 10 slots."""
    row: dict = {}

    for p in participants:
        slot = role_map[p["participantId"]]

        # -- Champion & summoner spells --
        row[f"{slot}_champion_id"]   = p.get("championId", 0)
        row[f"{slot}_summoner1_id"]  = p.get("summoner1Id", 0)
        row[f"{slot}_summoner2_id"]  = p.get("summoner2Id", 0)

        # -- Rune tree & keystone --
        styles = p.get("perks", {}).get("styles", [])
        if styles:
            primary = styles[0]
            row[f"{slot}_primary_tree"] = primary.get("style", 0)
            sels = primary.get("selections", [])
            row[f"{slot}_keystone"] = sels[0].get("perk", 0) if sels else 0
        else:
            row[f"{slot}_primary_tree"] = 0
            row[f"{slot}_keystone"]     = 0

        # -- Champion mastery --
        pts, lvl, days = _mastery_features(
            p.get("puuid", ""), p.get("championId", 0), game_creation_ms
        )
        row[f"{slot}_mastery_points"]          = pts
        row[f"{slot}_mastery_level"]           = lvl
        row[f"{slot}_days_since_last_played"]  = days

    return row


def _extract_frame(pframes: dict, role_map: dict[int, str]) -> Optional[dict]:
    """
    Extract per-player features from one participantFrames snapshot.

    Returns None if the frame is incomplete (missing participants).
    """
    if len(pframes) != 10:
        return None

    row: dict = {}

    for pid_str, pf in pframes.items():
        pid  = int(pid_str)
        slot = role_map.get(pid)
        if slot is None:
            return None  # unexpected participantId

        cs  = pf.get("championStats", {})
        ds  = pf.get("damageStats",   {})
        pos = pf.get("position",      {})

        health     = cs.get("health",    0)
        health_max = cs.get("healthMax", 1) or 1  # avoid division by zero

        row.update({
            # Economy
            f"{slot}_total_gold":         pf.get("totalGold",      0),
            f"{slot}_current_gold":       pf.get("currentGold",    0),
            f"{slot}_gold_per_second":    pf.get("goldPerSecond",  0),
            # Progression
            f"{slot}_xp":                 pf.get("xp",    0),
            f"{slot}_level":              pf.get("level", 1),
            # Farm
            f"{slot}_minions_killed":         pf.get("minionsKilled",       0),
            f"{slot}_jungle_minions_killed":  pf.get("jungleMinionsKilled", 0),
            f"{slot}_cs_total":               (pf.get("minionsKilled", 0)
                                               + pf.get("jungleMinionsKilled", 0)),
            # CC inflicted on enemies
            f"{slot}_time_enemy_cc":      pf.get("timeEnemySpentControlled", 0),
            # Champion stats
            f"{slot}_ability_haste":      cs.get("abilityHaste",      0),
            f"{slot}_ability_power":      cs.get("abilityPower",      0),
            f"{slot}_armor":              cs.get("armor",             0),
            f"{slot}_attack_damage":      cs.get("attackDamage",      0),
            f"{slot}_attack_speed":       cs.get("attackSpeed",       0),
            f"{slot}_health":             health,
            f"{slot}_health_max":         health_max,
            f"{slot}_hp_ratio":           round(health / health_max, 4),
            f"{slot}_magic_resist":       cs.get("magicResist",       0),
            f"{slot}_movement_speed":     cs.get("movementSpeed",     0),
            f"{slot}_magic_pen":          cs.get("magicPen",          0),
            f"{slot}_armor_pen_pct":      cs.get("armorPenPercent",   0),
            f"{slot}_omnivamp":           cs.get("omnivamp",          0),
            # Cumulative damage stats
            f"{slot}_dmg_to_champs":      ds.get("totalDamageDoneToChampions",    0),
            f"{slot}_phys_dmg_to_champs": ds.get("physicalDamageDoneToChampions", 0),
            f"{slot}_magic_dmg_to_champs":ds.get("magicDamageDoneToChampions",    0),
            f"{slot}_true_dmg_to_champs": ds.get("trueDamageDoneToChampions",     0),
            f"{slot}_dmg_taken":          ds.get("totalDamageTaken",              0),
            # Map position (normalised 0-1)
            f"{slot}_pos_x":              round(pos.get("x", 0) / MAP_SIZE, 4),
            f"{slot}_pos_y":              round(pos.get("y", 0) / MAP_SIZE, 4),
        })

    return row


def _compute_differentials(frame_row: dict) -> dict:
    """Compute lane matchup differentials and team-level totals."""
    diffs: dict = {}

    for role in ROLES:
        b_gold = frame_row.get(f"blue_{role}_total_gold", 0)
        r_gold = frame_row.get(f"red_{role}_total_gold",  0)
        b_xp   = frame_row.get(f"blue_{role}_xp",         0)
        r_xp   = frame_row.get(f"red_{role}_xp",          0)
        b_cs   = frame_row.get(f"blue_{role}_cs_total",   0)
        r_cs   = frame_row.get(f"red_{role}_cs_total",    0)
        b_lvl  = frame_row.get(f"blue_{role}_level",      0)
        r_lvl  = frame_row.get(f"red_{role}_level",       0)
        b_dmg  = frame_row.get(f"blue_{role}_dmg_to_champs", 0)
        r_dmg  = frame_row.get(f"red_{role}_dmg_to_champs",  0)

        diffs[f"diff_{role}_gold"]  = b_gold - r_gold
        diffs[f"diff_{role}_xp"]    = b_xp   - r_xp
        diffs[f"diff_{role}_cs"]    = b_cs   - r_cs
        diffs[f"diff_{role}_level"] = b_lvl  - r_lvl
        diffs[f"diff_{role}_dmg"]   = b_dmg  - r_dmg

    # Team totals and global differentials
    b_total_gold = sum(frame_row.get(f"blue_{r}_total_gold", 0) for r in ROLES)
    r_total_gold = sum(frame_row.get(f"red_{r}_total_gold",  0) for r in ROLES)
    b_total_xp   = sum(frame_row.get(f"blue_{r}_xp",         0) for r in ROLES)
    r_total_xp   = sum(frame_row.get(f"red_{r}_xp",          0) for r in ROLES)
    b_total_cs   = sum(frame_row.get(f"blue_{r}_cs_total",   0) for r in ROLES)
    r_total_cs   = sum(frame_row.get(f"red_{r}_cs_total",    0) for r in ROLES)

    diffs["blue_total_gold"]  = b_total_gold
    diffs["red_total_gold"]   = r_total_gold
    diffs["gold_diff_total"]  = b_total_gold - r_total_gold
    diffs["xp_diff_total"]    = b_total_xp   - r_total_xp
    diffs["cs_diff_total"]    = b_total_cs   - r_total_cs

    return diffs


# ── Delta / rate features ─────────────────────────────────────────────────────

def _add_delta_features(rows: list[dict]) -> None:
    """
    Add per-slot momentum (delta) and rate features in-place to a game's rows.

    Requires rows to be in ascending minute order (as produced by _process_game).

    Per slot (×10 slots):
      {slot}_gold_delta_1m   — gold earned in the last 1 min  (vs prior minute)
      {slot}_gold_delta_3m   — gold earned in the last 3 min
      {slot}_xp_delta_1m     — XP gained in the last 1 min
      {slot}_cs_delta_1m     — CS gained in the last 1 min
      {slot}_dmg_delta_1m    — damage to champs dealt in last 1 min
      {slot}_gold_per_min    — total_gold / current_minute  (overall rate)
      {slot}_cs_per_min      — cs_total   / current_minute

    Team differentials (new):
      diff_{role}_gold_delta_1m  (×5 roles)
      gold_diff_delta_1m         — team-total gold-delta diff, 1 min
      gold_diff_delta_3m         — team-total gold-delta diff, 3 min
    """
    # Build minute → row-index map for O(1) lookups
    min_to_idx: dict[int, int] = {r["minute"]: i for i, r in enumerate(rows)}

    for i, row in enumerate(rows):
        m     = row["minute"]
        prev1 = min_to_idx.get(m - 1)   # index of row at minute m-1 (or None)
        prev3 = min_to_idx.get(m - 3)   # index of row at minute m-3 (or None)

        for slot in SLOT_NAMES:
            g   = row[f"{slot}_total_gold"]
            xp  = row[f"{slot}_xp"]
            cs  = row[f"{slot}_cs_total"]
            dmg = row[f"{slot}_dmg_to_champs"]

            if prev1 is not None:
                p1 = rows[prev1]
                row[f"{slot}_gold_delta_1m"] = g   - p1[f"{slot}_total_gold"]
                row[f"{slot}_xp_delta_1m"]   = xp  - p1[f"{slot}_xp"]
                row[f"{slot}_cs_delta_1m"]   = cs  - p1[f"{slot}_cs_total"]
                row[f"{slot}_dmg_delta_1m"]  = dmg - p1[f"{slot}_dmg_to_champs"]
            else:
                # Minute 1: no prior row — delta equals the absolute value
                # (frame 0 was all-zeros and is excluded from our dataset)
                row[f"{slot}_gold_delta_1m"] = g
                row[f"{slot}_xp_delta_1m"]   = xp
                row[f"{slot}_cs_delta_1m"]   = cs
                row[f"{slot}_dmg_delta_1m"]  = dmg

            row[f"{slot}_gold_delta_3m"] = (
                g - rows[prev3][f"{slot}_total_gold"] if prev3 is not None else g
            )

            row[f"{slot}_gold_per_min"] = g  / m if m > 0 else 0.0
            row[f"{slot}_cs_per_min"]   = cs / m if m > 0 else 0.0

        # Team-level gold-delta differentials
        for role in ROLES:
            b_d1 = row[f"blue_{role}_gold_delta_1m"]
            r_d1 = row[f"red_{role}_gold_delta_1m"]
            row[f"diff_{role}_gold_delta_1m"] = b_d1 - r_d1

        b_total_d1 = sum(row[f"blue_{r}_gold_delta_1m"] for r in ROLES)
        r_total_d1 = sum(row[f"red_{r}_gold_delta_1m"]  for r in ROLES)
        b_total_d3 = sum(row[f"blue_{r}_gold_delta_3m"] for r in ROLES)
        r_total_d3 = sum(row[f"red_{r}_gold_delta_3m"]  for r in ROLES)
        row["gold_diff_delta_1m"] = b_total_d1 - r_total_d1
        row["gold_diff_delta_3m"] = b_total_d3 - r_total_d3


# ── Per-game processing ───────────────────────────────────────────────────────

def _process_game(
    match_path: Path,
    timeline_path: Path,
) -> tuple[Optional[list[dict]], str]:
    """
    Process one game.  Returns (rows, "ok") or (None, reason_skipped).
    """
    # ── Load match ──
    with open(match_path, encoding="utf-8") as f:
        match = json.load(f)

    info = match.get("info", {})

    # Filters
    if info.get("gameMode") != "CLASSIC":
        return None, "not_classic"
    if info.get("gameDuration", 0) < MIN_GAME_DURATION_S:
        return None, "too_short"

    participants = info.get("participants", [])
    if len(participants) != 10:
        return None, "wrong_participant_count"

    # Role mapping
    role_map = _build_role_map(participants)
    if role_map is None:
        return None, "invalid_roles"

    # Win label (blue team = teamId 100)
    blue_win: Optional[int] = None
    for team in info.get("teams", []):
        if team.get("teamId") == 100:
            blue_win = 1 if team.get("win") else 0
            break
    if blue_win is None:
        return None, "no_blue_team"

    game_id          = match.get("metadata", {}).get("matchId", match_path.stem)
    game_creation_ms = int(info.get("gameCreation", 0))
    game_duration_s  = info.get("gameDuration", 0)

    region_prefix = game_id.split("_")[0].lower()
    region_int    = REGION_INT_MAP.get(region_prefix, -1)

    # Pre-match features (static — same for every minute of this game)
    prematch = _extract_prematch(participants, role_map, game_creation_ms)

    # ── Load timeline ──
    with open(timeline_path, encoding="utf-8") as f:
        timeline = json.load(f)

    frames = timeline.get("info", {}).get("frames", [])
    if len(frames) < 2:
        return None, "insufficient_timeline"

    # ── Build one row per minute (skip frame 0 = pre-game all-zeros state) ──
    rows: list[dict] = []

    for frame_idx, frame in enumerate(frames):
        if frame_idx == 0:
            continue  # pre-game snapshot, not useful

        frame_features = _extract_frame(frame.get("participantFrames", {}), role_map)
        if frame_features is None:
            continue  # incomplete frame — skip

        diffs = _compute_differentials(frame_features)

        row: dict = {
            "game_id":          game_id,
            "minute":           frame_idx,
            "blue_win":         blue_win,
            "game_duration_min": round(game_duration_s / 60.0, 2),
            "region":           region_int,
        }
        row.update(prematch)
        row.update(frame_features)
        row.update(diffs)
        rows.append(row)

    if rows:
        _add_delta_features(rows)

    return (rows, "ok") if rows else (None, "empty_timeline")


# ── Parquet writer helper ─────────────────────────────────────────────────────

def _flush_chunk(
    chunk: list[dict],
    writer_box: list,
    output_path: Path,
) -> None:
    """Convert *chunk* to a DataFrame and append it to the Parquet file."""
    df = pd.DataFrame(chunk)

    # Downcast types to save memory and disk space.
    # float64 → float32, int64 → int32 where values fit.
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")

    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer_box[0] is None:
        writer_box[0] = pq.ParquetWriter(output_path, table.schema, compression="snappy")

    writer_box[0].write_table(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(limit: Optional[int] = None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "features.parquet"

    # ── Prompt before overwriting ──
    if output_path.exists():
        ans = input(f"\n'{output_path}' already exists. Overwrite? [y/N]: ").strip().lower()
        if ans != "y":
            log.info("Aborted — existing file preserved.")
            return
        output_path.unlink()

    # ── Collect match files ──
    match_files = sorted(MATCH_DIR.glob("match_*.json"))
    if limit:
        match_files = match_files[:limit]

    log.info(f"Processing {len(match_files):,} games -> {output_path}")

    skip_counts: dict[str, int] = {}
    total_rows   = 0
    total_games  = 0
    chunk: list[dict] = []
    writer_box   = [None]  # mutable box so _flush_chunk can assign the writer

    try:
        pbar = tqdm(match_files, desc="Games", unit="game", dynamic_ncols=True)
        for match_path in pbar:
            match_id      = match_path.stem.replace("match_", "")
            timeline_path = TIMELINE_DIR / f"timeline_{match_id}.json"

            if not timeline_path.exists():
                skip_counts["no_timeline"] = skip_counts.get("no_timeline", 0) + 1
                continue

            try:
                rows, reason = _process_game(match_path, timeline_path)
            except Exception as exc:
                log.debug(f"Parse error in {match_path.name}: {exc}")
                skip_counts["parse_error"] = skip_counts.get("parse_error", 0) + 1
                continue

            if rows is None:
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                continue

            chunk.extend(rows)
            total_rows  += len(rows)
            total_games += 1

            # Write chunk when buffer is large enough
            if len(chunk) >= ROW_CHUNK_SIZE:
                _flush_chunk(chunk, writer_box, output_path)
                chunk.clear()

            pbar.set_postfix(
                rows=f"{total_rows:,}",
                cache=f"{len(_mastery_cache):,}",
                skipped=sum(skip_counts.values()),
            )

        # Flush remaining rows
        if chunk:
            _flush_chunk(chunk, writer_box, output_path)

    finally:
        if writer_box[0] is not None:
            writer_box[0].close()

    # ── Summary ──
    size_mb = output_path.stat().st_size / 1e6 if output_path.exists() else 0
    log.info("=" * 60)
    log.info(f"Games processed : {total_games:,}")
    log.info(f"Total rows      : {total_rows:,}")
    log.info(f"Output size     : {size_mb:.1f} MB")
    log.info(f"Mastery cache   : {len(_mastery_cache):,} players loaded")
    if skip_counts:
        log.info("Skip reasons:")
        for reason, count in sorted(skip_counts.items(), key=lambda x: -x[1]):
            log.info(f"  {reason:30s}: {count:,}")
    log.info("=" * 60)
    log.info(f"Done. Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoL feature engineering pipeline")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N games (useful for testing)."
    )
    args = parser.parse_args()
    main(limit=args.limit)
