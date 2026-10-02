"""Report writing with a mandatory provenance block (plan B1: "runs without it are void").

Every report embeds: UTC timestamp, git describe (best-effort), SHA-256 of the exact
config, per-dataset manifest SHA-256 + instance counts, library versions, and the solve
ledger (plan WP5/B6). Numpy arrays serialize as lists so per-seed / per-instance arrays
persist verbatim.
"""

from __future__ import annotations

import hashlib
import json
import os
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
#
# A PREREG file carries one `CONFIG_SHA256 = <hash>` line per configuration it
# governs. The Phase-1/Phase-2 files govern one configuration (one unlabelled
# line). wp8 E-series (Stage 1.28): one pre-registration governs several arms,
# each its own configuration, so the lines may carry a label -- the config
# file's stem:  `CONFIG_SHA256[e1_2d_base] = <fill before tagging>`.
# Rules (unlabelled single-line files behave exactly as before):
#   * stamping fills the line labelled with the config's stem (or the single
#     unlabelled line); a multi-line file refuses an unlabelled stamp;
#   * verification refuses while ANY line is still the placeholder -- every
#     arm is frozen before any arm runs (no arm can be tuned after another's
#     result is seen) -- and then requires the config's hash to equal its own
#     labelled line (or, for unlabelled files, one of the recorded hashes).

PREREG_PLACEHOLDER = "<fill before tagging>"
_PREREG_RE = re.compile(r"CONFIG_SHA256(?:\[(?P<label>[^\]\n]+)\])?\s*=\s*"
                        r"(?P<val>[0-9a-fA-F]{64}|<fill before tagging>)")


def read_prereg_entries(prereg_file) -> list:
    """Every (label or None, recorded value) in the file, in order."""
    text = Path(prereg_file).read_text()
    return [(m.group("label"), m.group("val")) for m in _PREREG_RE.finditer(text)]


def read_prereg_hash(prereg_file) -> str | None:
    """The first recorded value (hash or placeholder), or None if absent
    (single-configuration files: the hash; see read_prereg_entries)."""
    e = read_prereg_entries(prereg_file)
    return e[0][1] if e else None


def stamp_prereg(prereg_file, config: dict, label: str | None = None) -> str:
    """Write the config's hash into its CONFIG_SHA256 line; returns the hash.

    `label` selects the line `CONFIG_SHA256[label] = ...` (the CLI passes the
    config file's stem). Without a label the file must hold exactly one line."""
    p = Path(prereg_file)
    text = p.read_text()
    matches = list(_PREREG_RE.finditer(text))
    if not matches:
        raise ValueError(f"{p}: no CONFIG_SHA256 line to stamp")
    h = config_sha256(config)
    labelled = [m for m in matches if m.group("label") is not None]
    if label is not None and labelled:
        target = [m for m in labelled if m.group("label").strip() == label]
        if len(target) != 1:
            raise ValueError(f"{p}: expected exactly one CONFIG_SHA256[{label}] line, "
                             f"found {len(target)} (labels: "
                             f"{[m.group('label') for m in labelled]})")
        m = target[0]
    elif len(matches) == 1:
        m = matches[0]
    else:
        raise ValueError(f"{p}: {len(matches)} CONFIG_SHA256 lines -- stamp each with "
                         "its label (the config file's stem)")
    start, end = m.span("val")
    p.write_text(text[:start] + h + text[end:])
    return h


def verify_prereg(config: dict, prereg_file, label: str | None = None) -> str:
    """Raise unless the pre-registration is fully stamped and records this
    exact config (under its own label when the file is labelled).

    This is the plan's 'post-hoc criterion changes are prohibited' made executable:
    editing the deciding config after tagging makes the run refuse to start.
    """
    p = Path(prereg_file)
    if not p.exists():
        raise ValueError(f"prereg_guard: {p} not found")
    entries = read_prereg_entries(p)
    if not entries:
        raise ValueError(f"prereg_guard: {p} has no CONFIG_SHA256 line")
    open_ = [lab or "(unlabelled)" for lab, v in entries if v == PREREG_PLACEHOLDER]
    if open_:
        raise ValueError(f"prereg_guard: {p} is unstamped ({', '.join(open_)}) -- run "
                         "`fejepa prereg <config> --stamp` for every configuration it "
                         "governs, commit, and git tag before the deciding run")
    actual = config_sha256(config)
    own = [v for lab, v in entries if lab is not None and label is not None
           and lab.strip() == label]
    recorded = own if own else [v for _, v in entries]
    if actual not in recorded:
        raise ValueError("prereg_guard: config hash mismatch -- the config changed "
                         f"after tagging (recorded {[r[:12] + '...' for r in recorded]}, "
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
    # wp9 Stage 0b: atomic (temp file, then rename) -- an interrupted write
    # leaves the previous file or none, never a truncated report that a
    # restart would take for a finished run; the bytes are unchanged
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, cls=_Encoder))
    os.replace(tmp, path)
    return path
