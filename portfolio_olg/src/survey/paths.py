"""Resolve local survey microdata paths (gitignored config)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

_PORTFOLIO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PORTFOLIO_ROOT / "data"

# Default locations on this machine (Dropbox sibling projects).
_DEFAULTS: Dict[str, str] = {
    "sfs_panel": r"C:\Users\thoma\Dropbox\CA_costsinflation\data\raw\sfs\sfs1999_2023.csv",
    "shs_ry2023": r"C:\Users\thoma\Dropbox\SHS\RY2023",
    "shs_edm": r"C:\Users\thoma\Dropbox\dataSHS\3508_SHS_EDM",
    "shs_consolidated": r"C:\Users\thoma\Dropbox\HRAM2\survey_data\clean\SHS\SHS_consolidated_data.csv",
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping")
        return data
    except ImportError:
        # Minimal key: value parser (no nested structures needed).
        out: Dict[str, Any] = {}
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" not in s:
                continue
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip().strip("\"'")
        return out


def load_local_paths(config_path: Optional[Path] = None) -> Dict[str, Path]:
    """
    Load absolute paths to SFS/SHS microdata.

    Precedence: ``config_path`` argument, then ``data/local_paths.yaml``,
    then environment variables ``SFS_PANEL_PATH``, ``SHS_RY2023_PATH``,
    ``SHS_EDM_PATH``, ``SHS_CONSOLIDATED_PATH``, then built-in defaults.
    """
    cfg: Dict[str, Any] = {}
    path = Path(config_path) if config_path is not None else _DATA_DIR / "local_paths.yaml"
    if path.is_file():
        cfg = _read_yaml(path)

    def pick(key: str, env: str) -> Path:
        if key in cfg and cfg[key]:
            return Path(str(cfg[key])).expanduser()
        env_val = os.environ.get(env, "").strip()
        if env_val:
            return Path(env_val).expanduser()
        return Path(_DEFAULTS[key])

    return {
        "sfs_panel": pick("sfs_panel", "SFS_PANEL_PATH"),
        "shs_ry2023": pick("shs_ry2023", "SHS_RY2023_PATH"),
        "shs_edm": pick("shs_edm", "SHS_EDM_PATH"),
        "shs_consolidated": pick("shs_consolidated", "SHS_CONSOLIDATED_PATH"),
    }


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found at {path}. "
            f"Copy data/local_paths.example.yaml to data/local_paths.yaml and edit paths."
        )
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} directory not found at {path}. "
            f"Copy data/local_paths.example.yaml to data/local_paths.yaml and edit paths."
        )
    return path
