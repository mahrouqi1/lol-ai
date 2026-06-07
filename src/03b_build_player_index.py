"""
03b_build_player_index.py
=========================
Builds a per-player game history index from raw match / timeline JSONs.

Two outputs
-----------
1.  data/processed/player_game_summary.parquet  (always produced)
    One row per (game_id, puuid) — 10 rows per game.
    Columns:
        game_id, puuid, game_creation_ms, champion_id, team_position,
        team_id, player_won,
        total_gold, xp, cs_total, dmg_to_champs, dmg_taken, level,
        kills, deaths, assists, time_cc_others

    Used by 04c (game-summary player-context Transformer).

2.  data/processed/player_minute_sequences.parquet  (with --include-sequences)
    One row per (game_id, puuid, minute).
    Columns:
        game_id, puuid, minute,
        total_gold, xp, cs_total, dmg_to_champs, dmg_taken,
        level, time_enemy_cc_ms

    Used by 04d (minute-level player-context Transformer).

Usage
-----
    # Summary only (fast, ~130 K games in a few minutes)
    python src/03b_build_player_index.py

    # Also emit per-minute sequences (~35 M rows, needs ~4 GB RAM)
    python src/03b_build_player_index.py --include-sequences

    # Also index the context-game directories written by the updated 02
    python src/03b_build_player_index.py --include-context

    # Limit to N games (dev / smoke test)
    python src/03b_build_player_index.py --limit 5000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import json
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED    = PROJECT_ROOT / "data" / "processed"
LOG_DIR      = PROJECT_ROOT / "logs"

PROCESSED.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT   = PROCESSED / "player_game_summary.parquet"
SEQUENCES_OUT = PROCESSED / "player_minute_sequences.parquet"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "03b_build_player_index.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Participant fields we capture for the game-summary index
PART_FIELDS = {
    "goldEarned":                    "total_gold",
    "champExperience":               "xp",
    "totalDamageDealtToChampions":   "dmg_to_champs",
    "totalDamageTaken":              "dmg_taken",
    "champLevel":                    "level",
    "kills":                         "kills",
    "deaths":                        "deaths",
    "assists":                       "assists",
    "timeCCingOthers":               "time_cc_others",
}

N_WORKERS = max(1, (os.cpu_count() or 4) - 2)


# ── Worker function (runs in subprocess) ──────────────────────────────────────

def _process_match(args: tuple) -> tuple[list[dict], list[dict]]:
    """
    Parse one match JSON (+ its timeline if requested).
    Returns (summary_rows, sequence_rows).
    Called in a worker process — no logging here.
    """
    match_path, timeline_path, include_seq = args

    try:
        with open(match_path, encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return [], []

    info = m.get("info", {})
    game_id        = m.get("metadata", {}).get("matchId", match_path.stem.replace("match_", ""))
    game_creation  = info.get("gameCreation", 0)
    participants   = info.get("participants", [])
    teams_raw      = info.get("teams", [])

    if len(participants) != 10:
        return [], []

    # Build team_id -> win map
    team_win = {t["teamId"]: t["win"] for t in teams_raw}

    summary_rows = []
    pid_to_puuid: dict[int, str] = {}  # for timeline cross-ref

    for p in participants:
        puuid      = p.get("puuid", "")
        team_id    = p.get("teamId", 0)
        pid        = p.get("participantId", 0)
        position   = p.get("teamPosition") or p.get("individualPosition") or "UNKNOWN"
        champ_id   = p.get("championId", 0)
        player_won = bool(team_win.get(team_id, False))

        cs_total = (p.get("totalMinionsKilled", 0) or 0) + (p.get("neutralMinionsKilled", 0) or 0)

        row: dict = {
            "game_id":          game_id,
            "puuid":            puuid,
            "game_creation_ms": game_creation,
            "champion_id":      champ_id,
            "team_position":    position,
            "team_id":          team_id,
            "player_won":       player_won,
            "cs_total":         cs_total,
        }
        for src_field, dst_field in PART_FIELDS.items():
            if dst_field != "cs_total":  # already handled above
                row[dst_field] = p.get(src_field, 0) or 0

        summary_rows.append(row)
        pid_to_puuid[pid] = puuid

    # ── Per-minute sequences (optional) ───────────────────────────────────────
    seq_rows: list[dict] = []

    if include_seq and timeline_path is not None and timeline_path.exists():
        try:
            with open(timeline_path, encoding="utf-8") as f:
                tl = json.load(f)
        except Exception:
            tl = {}

        frames = tl.get("info", {}).get("frames", [])
        for frame in frames:
            ts_ms  = frame.get("timestamp", 0)
            minute = ts_ms // 60_000  # integer minute
            pframes = frame.get("participantFrames", {})

            for pid_str, pf in pframes.items():
                pid  = int(pid_str)
                puuid = pid_to_puuid.get(pid, "")
                if not puuid:
                    continue

                ds = pf.get("damageStats", {})
                cs = (pf.get("minionsKilled", 0) or 0) + (pf.get("jungleMinionsKilled", 0) or 0)

                seq_rows.append({
                    "game_id":          game_id,
                    "puuid":            puuid,
                    "minute":           minute,
                    "total_gold":       pf.get("totalGold", 0) or 0,
                    "xp":               pf.get("xp", 0) or 0,
                    "cs_total":         cs,
                    "dmg_to_champs":    ds.get("totalDamageDoneToChampions", 0) or 0,
                    "dmg_taken":        ds.get("totalDamageTaken", 0) or 0,
                    "level":            pf.get("level", 1) or 1,
                    "time_enemy_cc_ms": pf.get("timeEnemySpentControlled", 0) or 0,
                })

    return summary_rows, seq_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build player game-history index")
    p.add_argument("--include-sequences", action="store_true",
                   help="Also emit per-minute player sequences (large file)")
    p.add_argument("--include-context", action="store_true",
                   help="Include context_matches / context_timelines dirs")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N match files (0 = all)")
    p.add_argument("--workers", type=int, default=N_WORKERS,
                   help=f"Parallel worker processes (default: {N_WORKERS})")
    return p.parse_args()


def collect_match_paths(include_context: bool) -> list[Path]:
    dirs = [RAW_DIR / "matches"]
    if include_context:
        dirs.append(RAW_DIR / "context_matches")
    paths = []
    for d in dirs:
        if d.exists():
            paths.extend(sorted(d.glob("match_*.json")))
    return paths


def main() -> None:
    args = parse_args()
    t0   = time.time()

    match_paths = collect_match_paths(args.include_context)
    if args.limit:
        match_paths = match_paths[: args.limit]

    log.info("Match files to process: %d", len(match_paths))
    log.info("Include sequences: %s | workers: %d", args.include_sequences, args.workers)

    # Build timeline path for each match
    tl_dir         = RAW_DIR / "timelines"
    ctx_tl_dir     = RAW_DIR / "context_timelines"

    def _tl_path(mp: Path) -> Optional[Path]:
        tl_name = mp.name.replace("match_", "timeline_")
        # check both timeline dirs
        for td in (tl_dir, ctx_tl_dir):
            p = td / tl_name
            if p.exists():
                return p
        return None

    work_items = [
        (mp, _tl_path(mp), args.include_sequences)
        for mp in match_paths
    ]

    all_summary: list[dict] = []
    all_seqs:    list[dict] = []
    done = 0
    errors = 0

    log.info("Dispatching %d tasks to %d workers...", len(work_items), args.workers)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_match, item): item for item in work_items}
        for fut in as_completed(futures):
            try:
                s_rows, q_rows = fut.result()
                all_summary.extend(s_rows)
                all_seqs.extend(q_rows)
            except Exception as exc:
                errors += 1
                log.debug("Worker error: %s", exc)

            done += 1
            if done % 5_000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta  = (len(work_items) - done) / rate if rate > 0 else 0
                log.info("  Progress: %d / %d  (%.0f/s, ETA %.0f s)",
                         done, len(work_items), rate, eta)

    log.info("Finished processing. Errors: %d", errors)
    log.info("Summary rows: %d", len(all_summary))

    # ── Write summary parquet ─────────────────────────────────────────────────
    log.info("Writing %s ...", SUMMARY_OUT)
    df_summary = pd.DataFrame(all_summary)

    # Enforce types
    df_summary["game_creation_ms"] = df_summary["game_creation_ms"].astype("int64")
    df_summary["champion_id"]      = df_summary["champion_id"].astype("int16")
    df_summary["team_id"]          = df_summary["team_id"].astype("int16")
    df_summary["player_won"]       = df_summary["player_won"].astype(bool)
    for col in ["total_gold", "xp", "cs_total", "dmg_to_champs", "dmg_taken",
                "kills", "deaths", "assists", "time_cc_others"]:
        df_summary[col] = df_summary[col].astype("int32")
    df_summary["level"] = df_summary["level"].astype("int8")

    df_summary.to_parquet(SUMMARY_OUT, index=False, compression="snappy")
    log.info("Saved: %s  (%d rows x %d cols)",
             SUMMARY_OUT, len(df_summary), df_summary.shape[1])

    # ── Write sequences parquet ───────────────────────────────────────────────
    if args.include_sequences and all_seqs:
        log.info("Sequence rows: %d", len(all_seqs))
        log.info("Writing %s ...", SEQUENCES_OUT)
        df_seq = pd.DataFrame(all_seqs)
        df_seq["minute"] = df_seq["minute"].astype("int8")
        for col in ["total_gold", "xp", "cs_total", "dmg_to_champs", "dmg_taken",
                    "time_enemy_cc_ms"]:
            df_seq[col] = df_seq[col].astype("int32")
        df_seq["level"] = df_seq["level"].astype("int8")
        df_seq.to_parquet(SEQUENCES_OUT, index=False, compression="snappy")
        log.info("Saved: %s  (%d rows x %d cols)",
                 SEQUENCES_OUT, len(df_seq), df_seq.shape[1])
    elif args.include_sequences:
        log.warning("No sequence rows collected — check that timelines exist.")

    elapsed = time.time() - t0
    log.info("Done in %.1f s (%.1f min)", elapsed, elapsed / 60)

    # Summary stats
    n_games   = df_summary["game_id"].nunique()
    n_players = df_summary["puuid"].nunique()
    avg_games = len(df_summary) / max(n_players, 1)
    print("\n" + "=" * 55)
    print(f"  Unique games  : {n_games:>10,}")
    print(f"  Unique players: {n_players:>10,}")
    print(f"  Rows          : {len(df_summary):>10,}")
    print(f"  Avg games/player: {avg_games:.1f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
