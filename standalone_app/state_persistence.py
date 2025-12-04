from __future__ import annotations
"""Session state persistence utilities.

Stores a JSON file capturing last-used UI selections so the user can resume
quickly. Only non-sensitive local settings are stored (no secrets/keys).

Environment variable VLM_SESSION_FILE can override the destination path.
"""

from typing import Any, Dict, Optional
import json, os

SESSION_FILE = os.environ.get("VLM_SESSION_FILE", "session_state.json")


def load_state(path: Optional[str] = None) -> Dict[str, Any]:
    """Load session JSON.

    Parameters
    ----------
    path : str | None
        Optional explicit path. Falls back to module-level SESSION_FILE.
    """
    target = path or SESSION_FILE
    if not os.path.exists(target):
        return {}
    try:
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def resolve_session_path(path: Optional[str]) -> str:
        """Resolve a session file path.

        Rules:
        - If path is absolute, return as-is.
        - If path is relative, interpret it relative to the package share directory
            (i.e. the directory containing this module's parent package) so ROS nodes
            can specify a simple relative name.
        - If path is None, fall back to module-level SESSION_FILE (which itself may
            be absolute or relative; if relative we resolve it the same way).
        """
        candidate = path or SESSION_FILE
        if os.path.isabs(candidate):
                return candidate
        # Resolve relative to package root (two levels up from this file)
        pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return os.path.join(pkg_root, candidate)


def save_state(state: Dict[str, Any]) -> None:
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out
