"""Report writing with a mandatory provenance block (plan B1: "runs without it are void").

Every report embeds: UTC timestamp, git describe (best-effort), SHA-256 of the exact
config, per-dataset manifest SHA-256 + instance counts, library versions, and the solve
ledger (plan WP5/B6). Numpy arrays serialize as lists so per-seed / per-instance arrays
persist verbatim.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data.archive import load_manifest, manifest_sha256


def _git_describe() -> str:
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty", "--tags"],
                             capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


def config_sha256(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---- executable pre-registration freeze (plan Sec.5 item 7 / WP1) -------------

PREREG_PLACEHOLDER = "<fill before tagging>"
_PREREG_RE = re.compile(r"CONFIG_SHA256\s*=\s*([0-9a-fA-F]{64}|<fill before tagging>)")


def read_prereg_hash(prereg_file) -> str | None:
    """The hash recorded in PREREG.md, the placeholder, or None if the line is absent."""
    m = _PREREG_RE.search(Path(prereg_file).read_text())
    return m.group(1) if m else None


def stamp_prereg(prereg_file, config: dict) -> str:
    """Write the config's hash into PREREG.md's CONFIG_SHA256 line; returns the hash."""
    p = Path(prereg_file)
    text = p.read_text()
    if not _PREREG_RE.search(text):
        raise ValueError(f"{p}: no CONFIG_SHA256 line to stamp")
    h = config_sha256(config)
    p.write_text(_PREREG_RE.sub(f"CONFIG_SHA256 = {h}", text, count=1))
    return h


def verify_prereg(config: dict, prereg_file) -> str:
    """Raise unless PREREG.md's recorded hash matches this exact config.

    This is the plan's 'post-hoc criterion changes are prohibited' made executable:
    editing the deciding config after tagging makes the run refuse to start.
    """
    p = Path(prereg_file)
    if not p.exists():
        raise ValueError(f"prereg_guard: {p} not found")
    recorded = read_prereg_hash(p)
    if recorded is None:
        raise ValueError(f"prereg_guard: {p} has no CONFIG_SHA256 line")
    if recorded == PREREG_PLACEHOLDER:
        raise ValueError(f"prereg_guard: {p} is unstamped -- run "
                         "`fejepa prereg <config> --stamp`, commit, and git tag "
                         "before the deciding run")
    actual = config_sha256(config)
    if recorded != actual:
        raise ValueError("prereg_guard: config hash mismatch -- the config changed "
                         f"after tagging (recorded {recorded[:12]}..., "
                         f"actual {actual[:12]}...)")
    return actual


def dataset_provenance(data_dir) -> dict:
    m = load_manifest(data_dir)
    return {"dir": str(data_dir), "manifest_sha256": manifest_sha256(data_dir),
            "n_instances": (m.get("n_instances") or len(m.get("pairs", []))),
            "backend": m.get("backend"), "labelled_policy": m.get("labelled_policy")}


def provenance(config: dict, data_dirs: list, seeds: list[int]) -> dict:
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_describe(),
        "config_sha256": config_sha256(config),
        "datasets": [dataset_provenance(d) for d in data_dirs],
        "seeds": list(seeds),
        "versions": versions,
    }


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_report(path, payload: dict) -> Path:
    if "provenance" not in payload:
        raise ValueError("plan B1: reports without a provenance block are void")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, cls=_Encoder))
    return path
