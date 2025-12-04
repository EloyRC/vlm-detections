from __future__ import annotations
import logging

"""Centralized prompt manager.

Responsibilities:
* Load per-model prompt presets from YAML files under vlm_detections/config/prompts/*.yaml
* Expose lists of available system and user prompts per model
* Maintain current selections and support override text
* Render user prompt text with placeholders {CLASSES}, {USER_PROMPT}, {WIDTH}, {HEIGHT}, {THRESHOLD}
* Cosmos-Reason1 prompts sourced from vlm_detections/config/cosmos/prompts/*.yaml
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pathlib
import yaml

logger = logging.getLogger(__name__)


# Map App/UI model display names to prompt config file basenames
MODEL_TO_PROMPT_FILE: Dict[str, str] = {
    "OpenAI Vision (API)": "openai_vision.yaml",
    "Qwen2.5-VL": "qwen2_5_vl.yaml",
    "Qwen3-VL": "qwen3_vl.yaml",
    "TRex": "trex.yaml",
    "Florence-2": "florence2.yaml",
    "GroundingDINO": "groundingdino.yaml",
    "OWL-ViT": "owlvit.yaml",
    "InternVL3.5": "internvl3_5.yaml",
    # Cosmos handled specially (from its own folder)
}


def _workspace_root() -> pathlib.Path:
    # File lives at standalone_app/prompt_manager.py
    return pathlib.Path(__file__).resolve().parent


def _prompts_dir() -> pathlib.Path:
    return _workspace_root() / "config" / "prompts"


def _cosmos_prompts_dir() -> pathlib.Path:
    return _workspace_root() / "config" / "cosmos" / "prompts"


@dataclass
class PromptSet:
    system_prompts: Dict[str, str]
    user_prompts: Dict[str, str]


@dataclass
class CosmosPrompt:
    key: str
    system_raw: str
    user_raw: str
    path: pathlib.Path


class PromptManager:
    _prompts: Dict[str, PromptSet] = {}
    _selections: Dict[str, Tuple[Optional[str], Optional[str], bool, str]] = {}
    _cosmos_prompts: Dict[str, CosmosPrompt] = {}
    _warnings: List[str] = []  # Accumulated validation / migration warnings for Cosmos prompts

    @classmethod
    def load_all(cls) -> None:
        """Load (and migrate) all prompt presets.

        For Cosmos we also attempt light-weight migration of legacy formats and
        collect non-fatal warnings so the UI can surface potential issues
        without breaking functionality.
        """
        cls._warnings = []  # reset warnings each load
        # Load standard model prompt YAMLs
        pdir = _prompts_dir()
        if pdir.exists():
            for model_name, fname in MODEL_TO_PROMPT_FILE.items():
                fpath = pdir / fname
                if not fpath.exists():
                    continue
                try:
                    data = yaml.safe_load(fpath.read_text()) or {}
                    sys_p = data.get("system_prompts") or {}
                    usr_p = data.get("user_prompts") or {}
                    # Ensure dicts
                    if isinstance(sys_p, list):
                        # Convert list of objects with name/text to dict
                        sys_p = {str(i): str(v) for i, v in enumerate(sys_p)}
                    if isinstance(usr_p, list):
                        usr_p = {str(i): str(v) for i, v in enumerate(usr_p)}
                    sys_p = {str(k): str(v) for k, v in sys_p.items()}
                    usr_p = {str(k): str(v) for k, v in usr_p.items()}
                except Exception:
                    sys_p, usr_p = {}, {}
                cls._prompts[model_name] = PromptSet(system_prompts=sys_p, user_prompts=usr_p)

        # Perform legacy Cosmos migration BEFORE loading directory
        try:
            cls._migrate_legacy_cosmos_prompts()
        except Exception as e:
            cls._warnings.append(f"Cosmos migration failed: {e}")

        # Load Cosmos prompt YAMLs (each file defines one system + user prompt)
        cls._cosmos_prompts = {}
        cdir = _cosmos_prompts_dir()
        if cdir.exists():
            for p in sorted(cdir.glob("*.yaml")):
                try:
                    text = p.read_text()
                except Exception:
                    cls._warnings.append(f"Failed to read {p.name}")
                    continue
                try:
                    data = yaml.safe_load(text) or {}
                except Exception:
                    cls._warnings.append(f"Invalid YAML in {p.name}; skipping.")
                    continue
                if not isinstance(data, dict):
                    cls._warnings.append(f"Legacy / unsupported structure in {p.name}; expecting mapping.")
                    continue
                sys_raw = str(data.get("system_prompt") or "").rstrip()
                user_raw = str(data.get("user_prompt") or "").rstrip()
                # Basic validation
                if not sys_raw:
                    cls._warnings.append(f"{p.name} missing system_prompt (empty).")
                if not user_raw:
                    cls._warnings.append(f"{p.name} missing user_prompt (empty).")
                key = p.stem
                if key in cls._cosmos_prompts:
                    # Collision – generate unique key
                    i = 1
                    new_key = f"{key}_{i}"
                    while new_key in cls._cosmos_prompts:
                        i += 1
                        new_key = f"{key}_{i}"
                    cls._warnings.append(f"Duplicate prompt key '{key}' – stored as '{new_key}'.")
                    key = new_key
                cls._cosmos_prompts[key] = CosmosPrompt(key=key, system_raw=sys_raw, user_raw=user_raw, path=p)

    # ----------------------- Cosmos Migration & Validation -----------------------
    @classmethod
    def cosmos_warnings(cls) -> List[str]:
        """Return accumulated warnings related to Cosmos prompt loading / migration."""
        return list(cls._warnings)

    @classmethod
    def _migrate_legacy_cosmos_prompts(cls) -> None:
        """Detect and migrate legacy Cosmos prompt formats.

        Supported legacy patterns (best-effort heuristic):
        1. YAML file with top-level key 'prompts': list[ { name, system|instruction, user|prompt|template } ]
           -> Each entry becomes a new individual YAML file <sanitized-name>.yaml with system_prompt & user_prompt.
        2. YAML file with keys 'system'/'instruction' and 'user'/'prompt'/'template'.
           -> Renamed in-place to system_prompt / user_prompt (write back) if target keys absent.
        3. Plain YAML string (not a mapping) -> treat as user_prompt only; create migrated file wrapping it.

        Non-fatal issues generate warnings; original files are preserved unless a clear
        one-to-one rename is performed (case 2). Migration writes are idempotent.
        """
        cdir = _cosmos_prompts_dir()
        if not cdir.exists():
            return
        for f in list(cdir.glob("*.yaml")):
            try:
                raw = f.read_text()
            except Exception:
                cls._warnings.append(f"Could not read {f.name} for migration")
                continue
            try:
                data = yaml.safe_load(raw)
            except Exception:
                cls._warnings.append(f"Skipping {f.name}: invalid YAML (parse error)")
                continue
            # Case 3: plain string
            if isinstance(data, str):
                # Create new file with wrapped structure; keep original untouched
                new_name = f"{f.stem}_migrated.yaml"
                new_path = cdir / new_name
                if not new_path.exists():
                    wrapped = {"system_prompt": "", "user_prompt": data}
                    try:
                        new_path.write_text(yaml.safe_dump(wrapped, sort_keys=False, allow_unicode=True))
                        cls._warnings.append(f"Wrapped legacy string file {f.name} -> {new_name}")
                    except Exception:
                        cls._warnings.append(f"Failed to write migrated file {new_name}")
                continue
            if not isinstance(data, dict):
                # Unsupported structure
                continue
            # Case 1: list of prompts inside 'prompts'
            if isinstance(data.get("prompts"), list):
                for entry in data.get("prompts") or []:
                    if not isinstance(entry, dict):
                        continue
                    name = str(entry.get("name") or entry.get("key") or "prompt").strip() or "prompt"
                    system_val = entry.get("system") or entry.get("instruction") or entry.get("system_prompt") or ""
                    user_val = (
                        entry.get("user")
                        or entry.get("prompt")
                        or entry.get("template")
                        or entry.get("user_prompt")
                        or ""
                    )
                    sanitized = cls._sanitize_filename(name)
                    new_file = cdir / f"{sanitized}.yaml"
                    i = 1
                    while new_file.exists():
                        new_file = cdir / f"{sanitized}_{i}.yaml"
                        i += 1
                    content = {"system_prompt": system_val, "user_prompt": user_val}
                    try:
                        new_file.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True))
                        cls._warnings.append(f"Migrated entry '{name}' from {f.name} -> {new_file.name}")
                    except Exception:
                        cls._warnings.append(f"Failed to migrate entry '{name}' from {f.name}")
                # Keep original file (do not delete) but warn
                cls._warnings.append(f"Compound legacy file {f.name} processed (entries extracted).")
                continue
            # Case 2: simple rename of legacy keys
            updated = False
            if "system_prompt" not in data:
                legacy_sys = data.get("system") or data.get("instruction")
                if legacy_sys is not None:
                    data["system_prompt"] = legacy_sys
                    updated = True
            if "user_prompt" not in data:
                legacy_user = data.get("user") or data.get("prompt") or data.get("template")
                if legacy_user is not None:
                    data["user_prompt"] = legacy_user
                    updated = True
            if updated:
                try:
                    f.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
                    cls._warnings.append(f"Renamed legacy keys in {f.name}")
                except Exception:
                    cls._warnings.append(f"Failed to write updated legacy file {f.name}")

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        import re
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
        return safe or "preset"

    @classmethod
    def system_options(cls, model_name: str) -> List[str]:
        if model_name == "Cosmos-Reason1":
            return list(cls._cosmos_prompts.keys())
        ps = cls._prompts.get(model_name)
        return list(ps.system_prompts.keys()) if ps else []

    @classmethod
    def user_options(cls, model_name: str) -> List[str]:
        if model_name == "Cosmos-Reason1":
            return list(cls._cosmos_prompts.keys())
        ps = cls._prompts.get(model_name)
        return list(ps.user_prompts.keys()) if ps else []

    @classmethod
    def cosmos_yaml_path(cls, key: str) -> Optional[str]:
        cp = cls._cosmos_prompts.get(key)
        return str(cp.path) if cp else None

    @classmethod
    def select(
        cls,
        model_name: str,
        system_key: Optional[str],
        user_key: Optional[str],
        override_user: bool,
        override_text: str,
    ) -> None:
        cls._selections[model_name] = (system_key, user_key, bool(override_user), override_text or "")

    @classmethod
    def get_selected(cls, model_name: str) -> Tuple[Optional[str], Optional[str], bool, str]:
        return cls._selections.get(model_name, (None, None, False, ""))

    @classmethod
    def get_system_prompt(cls, model_name: str, reasoning: Optional[bool] = None) -> str:
        if model_name == "Cosmos-Reason1":
            # Determine selected system key (falls back to first)
            sys_key, _user_key, _ovr, _txt = cls.get_selected(model_name)
            if not sys_key or sys_key not in cls._cosmos_prompts:
                sys_key = next(iter(cls._cosmos_prompts.keys()), None)
            if not sys_key:
                return ""
            cp = cls._cosmos_prompts[sys_key]
            # Load addons
            addons_dir = _cosmos_prompts_dir() / "addons"
            english_txt = ""
            reasoning_txt = ""
            try:
                english_txt = (addons_dir / "english.txt").read_text().rstrip()
            except Exception:
                pass
            add_reasoning = bool(reasoning) if reasoning is not None else True
            if add_reasoning and "<think>" not in (cp.system_raw or ""):
                try:
                    reasoning_txt = (addons_dir / "reasoning.txt").read_text().rstrip()
                except Exception:
                    reasoning_txt = ""
            parts = [t for t in [english_txt, cp.system_raw, reasoning_txt] if t]
            return "\n\n".join(parts)
        ps = cls._prompts.get(model_name)
        if not ps:
            return ""
        system_key, _user_key, _ovr, _txt = cls.get_selected(model_name)
        if system_key and system_key in ps.system_prompts:
            return ps.system_prompts[system_key]
        # default to first
        return next(iter(ps.system_prompts.values())) if ps.system_prompts else ""

    @classmethod
    def render_user_prompt(
        cls,
        model_name: str,
        classes: List[str],
        user_input: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> str:
        """Return the effective user prompt after applying preset selection and placeholders.

        Logic (applies to ALL models including Cosmos):
        1. If override flag is set: use override text, else if empty fall back to user_input.
        2. Otherwise fetch the selected user preset (fall back to the first available preset).
        3. If the resolved preset text is empty/whitespace, fall back to user_input.
        4. Apply placeholder substitution on the final template string:
           {CLASSES}, {USER_PROMPT}, {WIDTH}, {HEIGHT}, {THRESHOLD}
        5. Return resulting string (never None; may be empty string if everything empty).
        """

        # Extract current selection (works for all models)
        _system_key, user_key, override_user, override_text = cls.get_selected(model_name)

        # Step 1: explicit override wins
        if override_user:
            base_template = (override_text or user_input or "")
            return cls._apply_substitutions(base_template, classes, user_input, width, height, threshold)

        # Step 2: fetch template from model-specific storage
        if model_name == "Cosmos-Reason1":
            if not user_key or user_key not in cls._cosmos_prompts:
                user_key = next(iter(cls._cosmos_prompts.keys()), None)
            base_template = cls._cosmos_prompts[user_key].user_raw if user_key else ""
        else:
            ps = cls._prompts.get(model_name)
            if ps:
                if user_key and user_key in ps.user_prompts:
                    base_template = ps.user_prompts[user_key]
                else:
                    # fall back to first available preset if any
                    base_template = next(iter(ps.user_prompts.values()), "") if ps.user_prompts else ""
            else:
                base_template = ""

        # Step 3: fallback to user_input if template empty
        if not base_template or not base_template.strip():
            base_template = user_input or ""

        # Step 4: placeholder substitution
        return cls._apply_substitutions(base_template, classes, user_input, width, height, threshold)

    @staticmethod
    def _apply_substitutions(
        template: str,
        classes: List[str],
        user_input: str,
        width: Optional[int],
        height: Optional[int],
        threshold: Optional[float],
    ) -> str:
        """Apply placeholder substitutions to a template.

        Safety: missing placeholders are left untouched; numeric conversions are best-effort.
        """
        repl = {
            "{CLASSES}": ", ".join(classes or []),
            "{USER_PROMPT}": user_input or "",
        }
        if width is not None:
            repl["{WIDTH}"] = str(int(width))
        if height is not None:
            repl["{HEIGHT}"] = str(int(height))
        if threshold is not None:
            try:
                repl["{THRESHOLD}"] = f"{float(threshold):.2f}"
            except Exception:
                repl["{THRESHOLD}"] = str(threshold)
        out = template or ""
        for k, v in repl.items():
            out = out.replace(k, v)
        return out

    # New utility methods for GUI editing / persistence
    @classmethod
    def get_raw_system_prompt(cls, model_name: str, key: Optional[str]) -> str:
        if model_name == "Cosmos-Reason1":
            cp = cls._cosmos_prompts.get(key or "") if key else None
            return cp.system_raw if cp else ""
        ps = cls._prompts.get(model_name)
        if not ps or not key:
            return ""
        return ps.system_prompts.get(key, "")

    @classmethod
    def get_raw_user_prompt(cls, model_name: str, key: Optional[str]) -> str:
        if model_name == "Cosmos-Reason1":
            cp = cls._cosmos_prompts.get(key or "") if key else None
            return cp.user_raw if cp else ""
        ps = cls._prompts.get(model_name)
        if not ps or not key:
            return ""
        return ps.user_prompts.get(key, "")

    @classmethod
    def update_system_prompt(cls, model_name: str, key: str, text: str) -> bool:
        if model_name == "Cosmos-Reason1":
            # Update cosmos system prompt file
            if not key or key not in cls._cosmos_prompts:
                return False
            cp = cls._cosmos_prompts[key]
            cp.system_raw = text
            try:
                data = yaml.safe_load(cp.path.read_text()) or {}
            except Exception:
                data = {}
            data["system_prompt"] = text
            if "user_prompt" not in data:
                data["user_prompt"] = cp.user_raw
            try:
                cp.path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
                return True
            except Exception:
                return False
        if model_name not in cls._prompts:
            return False
        ps = cls._prompts[model_name]
        if key not in ps.system_prompts:
            return False
        ps.system_prompts[key] = text
        return True

    @classmethod
    def update_user_prompt(cls, model_name: str, key: str, text: str) -> bool:
        if model_name == "Cosmos-Reason1":
            if not key or key not in cls._cosmos_prompts:
                return False
            cp = cls._cosmos_prompts[key]
            cp.user_raw = text
            try:
                data = yaml.safe_load(cp.path.read_text()) or {}
            except Exception:
                data = {}
            data["user_prompt"] = text
            if "system_prompt" not in data:
                data["system_prompt"] = cp.system_raw
            try:
                cp.path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
                return True
            except Exception:
                return False
        if model_name not in cls._prompts:
            return False
        ps = cls._prompts[model_name]
        if key not in ps.user_prompts:
            return False
        ps.user_prompts[key] = text
        return True

    @classmethod
    def add_system_prompt(cls, model_name: str, key: str, text: str) -> bool:
        if model_name == "Cosmos-Reason1":
            # Add a new cosmos system prompt file; pair with current selected user prompt (raw) if available.
            new_key = key.strip()
            if not new_key or new_key in cls._cosmos_prompts:
                return False
            # Determine user prompt to pair
            _sys_sel, user_sel, _ovr, _txt = cls.get_selected(model_name)
            user_raw = cls.get_raw_user_prompt(model_name, user_sel) if user_sel else ""
            filename = cls._unique_cosmos_filename(new_key)
            path = _cosmos_prompts_dir() / filename
            data = {"system_prompt": text, "user_prompt": user_raw}
            try:
                path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
                cls.load_all()  # reload cosmos prompts
                return True
            except Exception:
                return False
        if model_name not in cls._prompts:
            return False
        ps = cls._prompts[model_name]
        if key in ps.system_prompts:
            return False
        ps.system_prompts[key] = text
        return True

    @classmethod
    def add_user_prompt(cls, model_name: str, key: str, text: str) -> bool:
        if model_name == "Cosmos-Reason1":
            new_key = key.strip()
            if not new_key or new_key in cls._cosmos_prompts:
                return False
            _sys_sel, sys_sel, _ovr, _txt = cls.get_selected(model_name)
            system_raw = cls.get_raw_system_prompt(model_name, sys_sel) if sys_sel else ""
            filename = cls._unique_cosmos_filename(new_key)
            path = _cosmos_prompts_dir() / filename
            data = {"system_prompt": system_raw, "user_prompt": text}
            try:
                path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
                cls.load_all()
                return True
            except Exception:
                return False
        if model_name not in cls._prompts:
            return False
        ps = cls._prompts[model_name]
        if key in ps.user_prompts:
            return False
        ps.user_prompts[key] = text
        return True

    @classmethod
    def persist_model(cls, model_name: str) -> bool:
        # Write updated prompts back to the YAML file (in-memory only for Cosmos)
        if model_name == "Cosmos-Reason1":
            # Per-file persistence already handled; nothing to do.
            return True
        if model_name not in MODEL_TO_PROMPT_FILE:
            return False
        ps = cls._prompts.get(model_name)
        if not ps:
            return False
        fpath = _prompts_dir() / MODEL_TO_PROMPT_FILE[model_name]
        try:
            data = {
                "system_prompts": ps.system_prompts,
                "user_prompts": ps.user_prompts,
            }
            fpath.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True))
            return True
        except Exception:
            return False

    @classmethod
    def _unique_cosmos_filename(cls, base_key: str) -> str:
        """Create a unique filename for a new cosmos prompt using base key."""
        import re
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", base_key.strip()).strip("_") or "preset"
        fname = f"{safe}.yaml"
        cdir = _cosmos_prompts_dir()
        if not (cdir / fname).exists():
            return fname
        i = 1
        while (cdir / f"{safe}_{i}.yaml").exists():
            i += 1
        return f"{safe}_{i}.yaml"


# Eagerly load once on import
PromptManager.load_all()
