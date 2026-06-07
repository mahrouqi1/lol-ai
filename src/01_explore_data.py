"""
01 — Data Exploration
=====================
Download a small sample of matches and inspect what data the Riot API actually
provides. Run this **before** bulk harvesting to see the exact structure and keys
of the various API payloads (Match V5, Timeline, League V4, Mastery V4).

Usage
-----
    conda activate lol_shap_env
    python src/01_explore_data.py

Outputs
-------
    data/raw/explore/match_*.json      – full match payloads
    data/raw/explore/timeline_*.json   – full timeline payloads
    data/raw/explore/player_*.json     – player metadata payloads
    Console report detailing the structure of these payloads.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from riotwatcher import ApiError

from utils import (
    ensure_dir,
    get_watcher,
    handle_api_error,
    load_api_key,
    PROJECT_ROOT,
)

# ── Configuration ────────────────────────────────────────────────────────────
REGION = "na1"                 # Platform routing value (NA)
CONTINENT = "americas"         # Regional routing value (for Match V5)
NUM_SUMMONERS = 3              # How many Challenger summoners to sample
MATCHES_PER_SUMMONER = 3       # Recent ranked matches per summoner
EXPLORE_DIR = PROJECT_ROOT / "data" / "raw" / "explore"


# ── Helpers ──────────────────────────────────────────────────────────────────

def save_json(data: dict | list, path: Path) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"    ✓ Saved {path.name}  ({path.stat().st_size:,} bytes)")


def safe_api_call(fn, *args, watcher_box: list, **kwargs):
    """Call *fn* with automatic retry on 403 / 429."""
    while True:
        try:
            return fn(*args, **kwargs)
        except ApiError as err:
            should_retry, new_key = handle_api_error(err)
            if should_retry:
                if new_key:
                    watcher_box[0] = get_watcher(new_key)
                continue
            raise


# ── Report Generators ───────────────────────────────────────────────────────

def _print_dict_structure(d: dict, indent: int = 4, max_depth: int = 2, current_depth: int = 1):
    """Recursively print the keys and types of a dictionary."""
    if current_depth > max_depth:
        print(" " * indent + "...")
        return
    for k, v in sorted(d.items()):
        if isinstance(v, dict) and current_depth < max_depth:
            print(" " * indent + f"• {k} (dict):")
            _print_dict_structure(v, indent + 4, max_depth, current_depth + 1)
        elif isinstance(v, list) and v and isinstance(v[0], dict) and current_depth < max_depth:
            print(" " * indent + f"• {k} (list of dicts, showing first):")
            _print_dict_structure(v[0], indent + 4, max_depth, current_depth + 1)
        else:
            type_name = type(v).__name__
            preview = str(v)[:40] + ("..." if len(str(v)) > 40 else "")
            print(" " * indent + f"• {k:20s}: {type_name:5s} = {preview}")


def analyze_api_response(name: str, data: dict | list, is_list=False) -> None:
    """Print an analysis of what fields are present in the API response."""
    print(f"\n  ── {name} Structure ──")
    if not data:
        print("    (Empty response)")
        return
    
    if isinstance(data, list):
        print(f"    Returned a List (length {len(data)}). Showing keys of first item:")
        _print_dict_structure(data[0])
    else:
        _print_dict_structure(data)


def report_specific_events(timeline: dict, target_events: list[str]) -> None:
    """Find and print the structure of specific target events from the timeline."""
    frames = timeline.get("info", {}).get("frames", [])
    found_events = {e: None for e in target_events}
    
    for frame in frames:
        for event in frame.get("events", []):
            etype = event.get("type")
            if etype in found_events and found_events[etype] is None:
                found_events[etype] = event
                
        # Stop early if we found one of each
        if all(v is not None for v in found_events.values()):
            break

    print(f"\n  ── Detailed Event Structures ──")
    for etype, event_data in found_events.items():
        if event_data:
            print(f"\n    Event: {etype}")
            _print_dict_structure(event_data, indent=6)
        else:
            print(f"\n    Event: {etype} (Not found in this match sample)")


def report_storage_estimates(match_files: list[Path], timeline_files: list[Path]) -> None:
    """Calculate and print estimated storage requirements."""
    if not match_files or not timeline_files:
        return
        
    avg_match_kb = sum(f.stat().st_size for f in match_files) / len(match_files) / 1024
    avg_timeline_kb = sum(f.stat().st_size for f in timeline_files) / len(timeline_files) / 1024
    total_per_game_kb = avg_match_kb + avg_timeline_kb
    
    print(f"\n  ── Storage Size Estimates ──")
    print(f"    • Average Match V5 JSON size:    {avg_match_kb:.1f} KB")
    print(f"    • Average Timeline JSON size:    {avg_timeline_kb:.1f} KB")
    print(f"    • Total per game:                {total_per_game_kb:.1f} KB")
    print(f"    • 10,000 games estimated at:     {(total_per_game_kb * 10000) / 1024 / 1024:.2f} GB")


def report_timeline_events(timeline: dict) -> None:
    """Summarize the types of events found in the timeline."""
    frames = timeline.get("info", {}).get("frames", [])
    print(f"\n  ── Timeline Event Distribution ({len(frames)} frames) ──")
    
    event_counts: dict[str, int] = {}
    for frame in frames:
        for event in frame.get("events", []):
            etype = event.get("type", "UNKNOWN")
            event_counts[etype] = event_counts.get(etype, 0) + 1

    if not event_counts:
        print("    (No events found)")
        return

    for etype, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"    • {etype:30s} : {count} occurrences")


# ── Main Workflow ────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dir(EXPLORE_DIR)

    api_key = load_api_key()
    watcher = get_watcher(api_key)
    wb = [watcher]  # mutable box to safely swap watcher on 403

    # ── 1. League-V4 Setup ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 1: Fetching Challenger league to discover PUUIDs ({REGION})")
    print(f"{'='*60}")
    
    league = safe_api_call(
        lambda w, r: w.league.challenger_by_queue(r, "RANKED_SOLO_5x5"),
        wb[0], REGION, watcher_box=wb
    )
    entries = league.get("entries", [])
    
    if entries:
        print(f"  Successfully fetched {len(entries)} Challenger entries.")
        analyze_api_response("League Entry (League V4)", entries, is_list=True)
    else:
        print("  ❌ No entries found in Challenger league.")
        return

    # Sort by LP to get the top players
    top_entries = sorted(entries, key=lambda e: e.get("leaguePoints", 0), reverse=True)[:NUM_SUMMONERS]

    # ── 2. Download Matches ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 2: Downloading Matches & Timelines")
    print(f"{'='*60}")

    all_match_ids = set()
    sample_match_data = None
    sample_timeline_data = None

    for entry in top_entries:
        # League-V4 now returns 'puuid' directly (since 2024 changes).
        puuid = entry.get("puuid")
        if not puuid:
            print(f"  ⚠ Skipping entry with no PUUID: {entry}")
            continue

        print(f"\n  Fetching matches for PUUID: {puuid[:12]}…")
        try:
            match_ids = safe_api_call(
                lambda w, p, c, cnt: w.match.matchlist_by_puuid(c, p, queue=420, count=cnt),
                wb[0], puuid, CONTINENT, MATCHES_PER_SUMMONER, watcher_box=wb
            )
        except Exception as e:
            print(f"  ⚠ Failed to fetch matches for {puuid[:12]}: {e}")
            continue
        
        print(f"    Found Match IDs: {match_ids}")
        
        for mid in match_ids:
            if mid in all_match_ids:
                continue
            all_match_ids.add(mid)

            try:
                # Match payload
                match_data = safe_api_call(
                    lambda w, m, c: w.match.by_id(c, m), 
                    wb[0], mid, CONTINENT, watcher_box=wb
                )
                save_json(match_data, EXPLORE_DIR / f"match_{mid}.json")
                if not sample_match_data:
                    sample_match_data = match_data

                # Timeline payload
                timeline_data = safe_api_call(
                    lambda w, m, c: w.match.timeline_by_match(c, m), 
                    wb[0], mid, CONTINENT, watcher_box=wb
                )
                save_json(timeline_data, EXPLORE_DIR / f"timeline_{mid}.json")
                if not sample_timeline_data:
                    sample_timeline_data = timeline_data
            except Exception as e:
                print(f"  ⚠ Failed downloading match data for {mid}: {e}")

    # ── 3. Download Metadata ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 3: Downloading Player Metadata")
    print(f"{'='*60}")

    sample_summoner_data = None
    sample_mastery_data = None

    for puuid in [top_entries[0].get("puuid")] if top_entries else []:
        if not puuid:
            continue
        
        # Summoner V4
        try:
            summoner = safe_api_call(
                lambda w, p, r: w.summoner.by_puuid(r, p),
                wb[0], puuid, REGION, watcher_box=wb,
            )
            save_json(summoner, EXPLORE_DIR / f"player_summoner_{puuid[:12]}.json")
            sample_summoner_data = summoner
        except Exception as e:
            print(f"  ⚠ Summoner fetch failed: {e}")

        # Champion Mastery V4
        try:
            mastery = safe_api_call(
                lambda w, p, r: w.champion_mastery.top_by_puuid(r, p, count=5),
                wb[0], puuid, REGION, watcher_box=wb,
            )
            save_json(mastery, EXPLORE_DIR / f"player_mastery_{puuid[:12]}.json")
            sample_mastery_data = mastery
        except Exception as e:
            print(f"  ⚠ Mastery fetch failed: {e}")

    # ── 4. Structure Analysis ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 4: API Structure Analysis")
    print(f"{'='*60}")

    if sample_summoner_data:
        analyze_api_response("Summoner V4", sample_summoner_data)
    if sample_mastery_data:
        analyze_api_response("Champion Mastery V4", sample_mastery_data, is_list=True)
    if sample_match_data:
        analyze_api_response("Match V5 Root / Metadata", sample_match_data)
        if "info" in sample_match_data and "participants" in sample_match_data["info"]:
            analyze_api_response("Match V5 Participant Info", sample_match_data["info"]["participants"], is_list=True)
    if sample_timeline_data:
        if "info" in sample_timeline_data and "frames" in sample_timeline_data["info"]:
            first_frame = sample_timeline_data["info"]["frames"][0]
            if "participantFrames" in first_frame:
                p_frames = first_frame["participantFrames"]
                if p_frames:
                    first_pid = next(iter(p_frames))
                    analyze_api_response(f"Timeline ParticipantFrame (Participant {first_pid})", p_frames[first_pid])
        report_timeline_events(sample_timeline_data)
        
        # 1. Investigate specific rich event types
        report_specific_events(sample_timeline_data, ["CHAMPION_KILL", "BUILDING_KILL", "ELITE_MONSTER_KILL"])

    # 2. Check storage sizes
    match_files = list(EXPLORE_DIR.glob("match_*.json"))
    timeline_files = list(EXPLORE_DIR.glob("timeline_*.json"))
    report_storage_estimates(match_files, timeline_files)

    print(textwrap.dedent(f"""\
    \n{'='*60}
      ✅ Exploration Complete
    {'='*60}
      • DATA STRUCTURES: Review the keys printed above. Notice that Champion Mastery returns 
        points and last play time, but NOT winrate or total games played for that champ.
        (Winrate requires tracking match history manually, which we'll address in the pipeline).
      • EVENT INFO: Notice `CHAMPION_KILL` includes `victimDamageDealt`, `victimDamageReceived` 
        (death recap), and `assistDeltas`.
      • STORAGE: See the estimates above to plan your disk space for bulk harvesting.
      
      Data saved to {EXPLORE_DIR}. 
    """))


if __name__ == "__main__":
    main()
