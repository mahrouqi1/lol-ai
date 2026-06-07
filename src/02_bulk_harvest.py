"""
02 — Bulk Harvester
===================
Downloads League of Legends Match V5, Timeline, and Player Metadata (Mastery)
across multiple regions concurrently to bypass single-region rate limits.

Features
--------
- Multi-threaded: One thread per region (NA1, EUW1, KR).
- Resumable: Skips matches already on disk.
- Robust: Pauses and asks for a new API key on 403 rather than crashing.
- Polite: Respects 429 Retry-After headers and RiotWatcher's built-in limits.
- Context harvesting: Optionally fetches player match history for player-context
  models (03b / 04c / 04d).

Context harvesting modes
------------------------
--context-depth N
    After downloading each new game, fetch the N most recent prior games for
    every participant.  Context games are stored in separate directories:
        data/raw/context_matches/
        data/raw/context_timelines/
    These are NOT added to the main training set but are indexed by 03b so
    the player-context models can look up player histories.
    Default: 0 (disabled)

--context-pass
    Instead of harvesting new Challenger/GM games, iterate over ALL already-
    seen player PUUIDs and fetch their context history.  Useful to back-fill
    history for players already in your dataset without doing a full harvest.
    Implies --context-depth (use --context-depth to set depth, default 20).

Usage
-----
    conda activate lol_shap_env

    # Normal harvest (no context)
    python src/02_bulk_harvest.py

    # Harvest + fetch 20 context games per player
    python src/02_bulk_harvest.py --context-depth 20

    # Back-fill context for existing players only
    python src/02_bulk_harvest.py --context-pass --context-depth 20
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from riotwatcher import ApiError, LolWatcher

from utils import (
    ensure_dir,
    get_watcher,
    handle_api_error,
    load_api_key,
    PROJECT_ROOT,
)

# ── Configuration ─────────────────────────────────────────────────────────────

REGION_MAP = {
    "na1": "americas",
    "euw1": "europe",
    "kr": "asia",
}

MATCHES_PER_SUMMONER = 100

# Tiers. Apex tiers use dedicated league endpoints; standard tiers paginate via
# league.entries(region, queue, tier, division, page). ALL entries include puuid.
APEX_ENDPOINTS = {
    "CHALLENGER":  "challenger_by_queue",
    "GRANDMASTER": "grandmaster_by_queue",
    "MASTER":      "master_by_queue",
}
STANDARD_TIERS = ["DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"]
DIVISIONS = ["I", "II", "III", "IV"]
ALL_TIERS = list(APEX_ENDPOINTS.keys()) + STANDARD_TIERS
DEFAULT_PLAYERS_PER_TIER = 1000   # cap per tier per region (balance across elos)

# ── Directories ───────────────────────────────────────────────────────────────

RAW_DIR          = PROJECT_ROOT / "data" / "raw"
MATCH_DIR        = ensure_dir(RAW_DIR / "matches")
TIMELINE_DIR     = ensure_dir(RAW_DIR / "timelines")
PLAYER_DIR       = ensure_dir(RAW_DIR / "players")
CTX_MATCH_DIR    = ensure_dir(RAW_DIR / "context_matches")
CTX_TIMELINE_DIR = ensure_dir(RAW_DIR / "context_timelines")
GAME_TIER_FILE   = RAW_DIR / "game_source_tier.csv"   # game_id,tier (approx game elo)

# ── Logging ───────────────────────────────────────────────────────────────────

ensure_dir(PROJECT_ROOT / "logs")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "bulk_harvest.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Global State (Thread-Safe) ────────────────────────────────────────────────

_API_KEY_LOCK   = threading.Lock()
_SHARED_API_KEY: str = ""

_SEEN_MATCHES_LOCK = threading.Lock()
_SEEN_MATCHES: set[str] = set()

# Covers BOTH main matches and context matches
_SEEN_CTX_LOCK = threading.Lock()
_SEEN_CTX: set[str] = set()

_SEEN_PLAYERS_LOCK = threading.Lock()
_SEEN_PLAYERS: set[str] = set()

# PUUIDs whose context has already been fetched (not re-fetched on restart)
_SEEN_CTX_PLAYERS_LOCK = threading.Lock()
_SEEN_CTX_PLAYERS: set[str] = set()

# Records each game's source tier (approx game elo) for a downstream rank feature.
_TIER_TAG_LOCK = threading.Lock()


# ── Initialization ────────────────────────────────────────────────────────────

def init_seen_caches() -> None:
    """Populate all in-memory seen sets from what's on disk."""
    logger.info("Initializing seen caches from disk...")

    for f in MATCH_DIR.glob("match_*.json"):
        _SEEN_MATCHES.add(f.stem.replace("match_", ""))

    for f in CTX_MATCH_DIR.glob("match_*.json"):
        _SEEN_CTX.add(f.stem.replace("match_", ""))
    # Main matches also count as "seen context" so we don't re-fetch them
    _SEEN_CTX.update(_SEEN_MATCHES)

    for f in PLAYER_DIR.glob("player_*.json"):
        _SEEN_PLAYERS.add(f.stem.replace("player_", ""))

    # A PUUID whose context directory contains at least one file is considered done.
    # We track this with a sentinel file: context_matches/ctx_done_{puuid}.flag
    for f in CTX_MATCH_DIR.glob("ctx_done_*.flag"):
        _SEEN_CTX_PLAYERS.add(f.stem.replace("ctx_done_", ""))

    logger.info(
        "Caches: %d main matches, %d context matches, %d players, %d ctx-done players",
        len(_SEEN_MATCHES), len(_SEEN_CTX) - len(_SEEN_MATCHES),
        len(_SEEN_PLAYERS), len(_SEEN_CTX_PLAYERS),
    )


# ── API Key management ────────────────────────────────────────────────────────

def get_current_api_key() -> str:
    with _API_KEY_LOCK:
        return _SHARED_API_KEY


def trigger_key_refresh(old_key: str) -> str:
    global _SHARED_API_KEY
    with _API_KEY_LOCK:
        if _SHARED_API_KEY != old_key:
            logger.info("Another thread already refreshed the key. Resuming.")
            return _SHARED_API_KEY

        # Unattended (personal key): never block on input(). Reload from .env in
        # case the key was rotated; otherwise back off briefly and let the bounded
        # retry in safe_api_call decide whether to give up on this call.
        logger.warning("API call got 401/403; reloading key from .env (unattended, no prompt).")
        new_key = load_api_key()
        _SHARED_API_KEY = new_key
        if new_key == old_key:
            time.sleep(5)
        else:
            logger.info("Key changed in .env; resuming with new key.")
        return new_key


# ── Core API call wrapper ─────────────────────────────────────────────────────

def safe_api_call(fn, *args, current_key: str, **kwargs) -> tuple[Optional[any], str]:
    """
    Execute an API call with retry logic.
    Returns (result, current_key).  current_key may have changed on 403.
    """
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            return fn(*args, **kwargs), current_key

        except ApiError as err:
            status_code = getattr(getattr(err, "response", None), "status_code", None)

            if status_code in (401, 403):
                current_key = trigger_key_refresh(current_key)
                try:
                    fn.__self__._base_api._api_key = current_key
                except AttributeError:
                    pass
                retries += 1   # bounded: don't loop forever on persistent 401/403
                continue

            elif status_code == 429:
                retry_after = int(getattr(err, "response").headers.get("Retry-After", 10))
                logger.warning("Rate limited (429). Sleeping %ds.", retry_after)
                time.sleep(retry_after)
                retries += 1
                continue

            elif status_code in [404, 500, 502, 503, 504]:
                logger.debug("API Error %s: %s", status_code, err)
                return None, current_key

            else:
                logger.error("Unhandled API Error %s: %s", status_code, err)
                raise

    logger.error("Failed after %d rate-limit retries.", max_retries)
    return None, current_key


# ── Player-level helpers ──────────────────────────────────────────────────────

def fetch_context_games(
    watcher:     LolWatcher,
    continent:   str,
    region:      str,
    puuid:       str,
    depth:       int,
    current_key: str,
) -> str:
    """
    Fetch up to `depth` most recent prior games for `puuid` and store them
    in context_matches / context_timelines.  Returns (possibly updated) current_key.

    Writes a sentinel file `ctx_done_{puuid}.flag` when done so this puuid
    is skipped on future runs.
    """
    with _SEEN_CTX_PLAYERS_LOCK:
        if puuid in _SEEN_CTX_PLAYERS:
            return current_key

    match_ids, current_key = safe_api_call(
        watcher.match.matchlist_by_puuid,
        continent, puuid, queue=420, count=depth,
        current_key=current_key,
    )

    if not match_ids:
        _mark_ctx_done(puuid)
        return current_key

    n_fetched = 0
    for mid in match_ids:
        with _SEEN_CTX_LOCK:
            if mid in _SEEN_CTX:
                continue

        match_data, current_key = safe_api_call(
            watcher.match.by_id, continent, mid, current_key=current_key
        )
        if not match_data:
            continue

        timeline_data, current_key = safe_api_call(
            watcher.match.timeline_by_match, continent, mid, current_key=current_key
        )
        if not timeline_data:
            continue

        with open(CTX_MATCH_DIR    / f"match_{mid}.json",    "w", encoding="utf-8") as f:
            json.dump(match_data, f, ensure_ascii=False)
        with open(CTX_TIMELINE_DIR / f"timeline_{mid}.json", "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, ensure_ascii=False)

        with _SEEN_CTX_LOCK:
            _SEEN_CTX.add(mid)

        n_fetched += 1

    if n_fetched > 0:
        logger.debug("[%s] Fetched %d context games for %s...", region, n_fetched, puuid[:8])

    _mark_ctx_done(puuid)
    return current_key


def _mark_ctx_done(puuid: str) -> None:
    """Write a sentinel file so this PUUID's context is not re-fetched."""
    (CTX_MATCH_DIR / f"ctx_done_{puuid}.flag").touch()
    with _SEEN_CTX_PLAYERS_LOCK:
        _SEEN_CTX_PLAYERS.add(puuid)


def _tag_game_tier(game_id: str, tier: str) -> None:
    """Append (game_id, source_tier) — approximate game elo for a rank feature."""
    with _TIER_TAG_LOCK:
        with open(GAME_TIER_FILE, "a", encoding="utf-8") as f:
            f.write(f"{game_id},{tier}\n")


def fetch_tier_puuids(watcher, region: str, tier: str, current_key: str, cap: int) -> tuple[list[str], str]:
    """PUUIDs for one tier in a region. Apex tiers via league endpoints; standard
    tiers via paginated league.entries. Capped + shuffled for cross-elo balance."""
    puuids: list[str] = []
    if tier in APEX_ENDPOINTS:
        method = getattr(watcher.league, APEX_ENDPOINTS[tier])
        league, current_key = safe_api_call(method, region, "RANKED_SOLO_5x5", current_key=current_key)
        if league:
            puuids = [e["puuid"] for e in league.get("entries", []) if e.get("puuid")]
    else:
        for div in DIVISIONS:
            page = 1
            while len(puuids) < cap:
                entries, current_key = safe_api_call(
                    watcher.league.entries, region, "RANKED_SOLO_5x5", tier, div,
                    page=page, current_key=current_key,
                )
                if not entries:
                    break
                puuids.extend(e["puuid"] for e in entries if e.get("puuid"))
                page += 1
            if len(puuids) >= cap:
                break
    if cap and len(puuids) > cap:
        random.shuffle(puuids)
        puuids = puuids[:cap]
    logger.info("[%s] %s: %d players", region, tier, len(puuids))
    return puuids, current_key


# ── Worker threads ────────────────────────────────────────────────────────────

def harvest_player(watcher, region, continent, puuid, tier, context_depth, current_key):
    """Harvest one player's recent ranked games, tagging each game with `tier`."""
    match_ids, current_key = safe_api_call(
        watcher.match.matchlist_by_puuid,
        continent, puuid, queue=420, count=MATCHES_PER_SUMMONER,
        current_key=current_key,
    )
    if not match_ids:
        return current_key

    new_matches_downloaded = 0
    for mid in match_ids:
        with _SEEN_MATCHES_LOCK:
            if mid in _SEEN_MATCHES:
                continue

        match_data, current_key = safe_api_call(
            watcher.match.by_id, continent, mid, current_key=current_key
        )
        if not match_data:
            continue

        timeline_data, current_key = safe_api_call(
            watcher.match.timeline_by_match, continent, mid, current_key=current_key
        )
        if not timeline_data:
            continue

        with open(MATCH_DIR    / f"match_{mid}.json",    "w", encoding="utf-8") as f:
            json.dump(match_data, f, ensure_ascii=False)
        with open(TIMELINE_DIR / f"timeline_{mid}.json", "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, ensure_ascii=False)

        with _SEEN_MATCHES_LOCK:
            _SEEN_MATCHES.add(mid)
        with _SEEN_CTX_LOCK:
            _SEEN_CTX.add(mid)
        _tag_game_tier(mid, tier)
        new_matches_downloaded += 1

        # ── Mastery + context for each participant ──────────────────────────
        participants = match_data.get("metadata", {}).get("participants", [])
        for p in participants:
            with _SEEN_PLAYERS_LOCK:
                if p not in _SEEN_PLAYERS:
                    _SEEN_PLAYERS.add(p)
                    mastery, current_key = safe_api_call(
                        watcher.champion_mastery.by_puuid, region, p,
                        current_key=current_key,
                    )
                    mastery_list = mastery if mastery else []
                    with open(PLAYER_DIR / f"player_{p}.json", "w", encoding="utf-8") as f:
                        json.dump(mastery_list, f, ensure_ascii=False)

            if context_depth > 0:
                current_key = fetch_context_games(
                    watcher, continent, region, p, context_depth, current_key,
                )

    if new_matches_downloaded > 0:
        logger.info("[%s][%s] +%d new matches for %s...",
                    region, tier, new_matches_downloaded, puuid[:8])
    return current_key


def worker_thread(region: str, continent: str, context_depth: int,
                  tiers: list[str], players_per_tier: int) -> None:
    """Harvest games for one region across the requested tiers (low->high elo)."""
    threading.current_thread().name = f"Worker-{region.upper()}"
    current_key = get_current_api_key()
    watcher = LolWatcher(current_key)

    for tier in tiers:
        puuids, current_key = fetch_tier_puuids(watcher, region, tier, current_key, players_per_tier)
        if current_key != watcher._base_api._api_key:
            watcher = LolWatcher(current_key)
        for puuid in puuids:
            latest_key = get_current_api_key()
            if current_key != latest_key:
                current_key = latest_key
                watcher = LolWatcher(current_key)
            current_key = harvest_player(
                watcher, region, continent, puuid, tier, context_depth, current_key
            )


def context_pass_worker(region: str, continent: str, context_depth: int) -> None:
    """
    Worker for --context-pass mode.
    Iterates over all already-seen players and fetches their context history.
    """
    threading.current_thread().name = f"CtxWorker-{region.upper()}"

    current_key = get_current_api_key()
    watcher = LolWatcher(current_key)

    # Get the list of all known players at startup
    with _SEEN_PLAYERS_LOCK:
        all_players = list(_SEEN_PLAYERS)

    logger.info("[%s] Context pass: %d players to process.", region, len(all_players))

    # Distribute players across regions by consistent hashing to avoid
    # all threads hammering the same players
    region_list = list(REGION_MAP.keys())
    my_index    = region_list.index(region)
    my_players  = [p for i, p in enumerate(all_players) if i % len(region_list) == my_index]

    logger.info("[%s] Assigned %d players (1/%d share).", region, len(my_players), len(region_list))

    for puuid in my_players:
        latest_key = get_current_api_key()
        if current_key != latest_key:
            current_key = latest_key
            watcher = LolWatcher(current_key)

        current_key = fetch_context_games(
            watcher, continent, region, puuid, context_depth, current_key,
        )

    logger.info("[%s] Context pass complete.", region)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoL Bulk Harvester")
    p.add_argument("--context-depth", type=int, default=0,
                   help="Fetch this many prior games per player as context history (0=disabled)")
    p.add_argument("--context-pass",  action="store_true",
                   help="Back-fill context history for all existing players (no new game harvest)")
    p.add_argument("--tiers", type=str, default="all",
                   help="'all' or comma list: CHALLENGER,GRANDMASTER,MASTER,DIAMOND,EMERALD,"
                        "PLATINUM,GOLD,SILVER,BRONZE,IRON")
    p.add_argument("--players-per-tier", type=int, default=DEFAULT_PLAYERS_PER_TIER,
                   help="Cap players sampled per tier per region (balance across elos)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global _SHARED_API_KEY

    args = parse_args()
    _SHARED_API_KEY = load_api_key()

    init_seen_caches()

    # Resolve effective context depth for context-pass mode
    effective_depth = args.context_depth
    if args.context_pass and effective_depth == 0:
        effective_depth = 20
        logger.info("--context-pass with no --context-depth; defaulting to depth=20.")

    if args.context_pass:
        logger.info("Starting Context Pass (depth=%d)...", effective_depth)
        mode_fn = lambda region, continent: context_pass_worker(region, continent, effective_depth)
    else:
        tiers = ALL_TIERS if args.tiers.strip().lower() == "all" else \
            [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
        bad = [t for t in tiers if t not in ALL_TIERS]
        if bad:
            logger.error("Unknown tiers: %s. Valid: %s", bad, ALL_TIERS); sys.exit(1)
        logger.info("Starting Bulk Harvester (context_depth=%d)...", effective_depth)
        logger.info("Regions: %s | Tiers: %s | cap/tier: %d",
                    list(REGION_MAP.keys()), tiers, args.players_per_tier)
        mode_fn = lambda region, continent: worker_thread(
            region, continent, effective_depth, tiers, args.players_per_tier)

    threads = []
    for region, continent in REGION_MAP.items():
        t = threading.Thread(target=mode_fn, args=(region, continent))
        threads.append(t)
        t.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Shutdown requested. Waiting for threads to finish current requests...")
    finally:
        logger.info("Main thread exiting.")


if __name__ == "__main__":
    main()
