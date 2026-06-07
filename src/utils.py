"""
LoL Win-Contribution Pipeline — Shared Utilities
=================================================
Provides:
  • load_api_key()   – reads RIOT_API_KEY from .env
  • get_watcher()    – returns a configured RiotWatcher instance
  • handle_api_error – graceful 403 (key expiry) & 429 (rate limit) handling
  • ensure_dir()     – mkdir helper
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, set_key
from riotwatcher import LolWatcher
from requests.exceptions import HTTPError


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


# ── API Key Management ───────────────────────────────────────────────────────

def load_api_key(*, dotenv_path: Path = DOTENV_PATH) -> str:
    """Load RIOT_API_KEY from *dotenv_path*.

    Raises
    ------
    SystemExit
        If the key is missing or still set to the placeholder value.
    """
    load_dotenv(dotenv_path, override=True)
    key = os.getenv("RIOT_API_KEY", "")
    if not key or key.startswith("RGAPI-paste"):
        print(
            "\n╔══════════════════════════════════════════════════════════╗"
            "\n║  RIOT_API_KEY not found or still set to placeholder.    ║"
            "\n║                                                          ║"
            "\n║  1. Go to https://developer.riotgames.com/               ║"
            "\n║  2. Copy your Development API Key.                       ║"
            "\n║  3. Paste it into .env as:                               ║"
            "\n║     RIOT_API_KEY=RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxx  ║"
            "\n╚══════════════════════════════════════════════════════════╝"
        )
        sys.exit(1)
    return key


def get_watcher(api_key: Optional[str] = None) -> LolWatcher:
    """Return a :class:`LolWatcher` configured with the current API key."""
    if api_key is None:
        api_key = load_api_key()
    return LolWatcher(api_key)


# ── Error Handling ───────────────────────────────────────────────────────────

def _prompt_for_new_key() -> str:
    """Pause execution and ask the user to update .env with a fresh key."""
    print(
        "\n╔══════════════════════════════════════════════════════════════╗"
        "\n║  ⚠  API key has EXPIRED (HTTP 403).                        ║"
        "\n║                                                              ║"
        "\n║  1. Go to https://developer.riotgames.com/                   ║"
        "\n║  2. Regenerate your Development API Key.                     ║"
        "\n║  3. Open .env and replace the old key with the new one.      ║"
        "\n║  4. Press ENTER here to continue harvesting.                 ║"
        "\n╚══════════════════════════════════════════════════════════════╝"
    )
    input("\n>>> Press ENTER after updating .env ... ")
    return load_api_key()


def handle_api_error(err: Exception) -> tuple[bool, Optional[str]]:
    """Handle common Riot API errors gracefully.

    Parameters
    ----------
    err : Exception
        The exception raised by RiotWatcher / requests.

    Returns
    -------
    (should_retry, new_api_key)
        *should_retry* is ``True`` when the caller should re-attempt the
        request (after a key refresh or a rate-limit sleep).
        *new_api_key* is the refreshed key string when the key was rotated,
        otherwise ``None``.
    """
    # RiotWatcher wraps HTTP errors in requests.exceptions.HTTPError
    # and stores the response on the exception object.
    response = getattr(err, "response", None)
    status_code = getattr(response, "status_code", None)

    # ── 403: Forbidden — key likely expired ──────────────────────────────
    if status_code == 403:
        new_key = _prompt_for_new_key()
        return True, new_key

    # ── 429: Rate limited ────────────────────────────────────────────────
    if status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 10))
        print(f"  ⏳ Rate-limited. Sleeping {retry_after}s …")
        time.sleep(retry_after)
        return True, None

    # ── Everything else: not recoverable automatically ───────────────────
    print(f"  ✖ Unhandled API error (HTTP {status_code}): {err}")
    return False, None


# ── Filesystem Helpers ───────────────────────────────────────────────────────

def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if it doesn't exist, then return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
