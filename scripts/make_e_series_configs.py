#!/usr/bin/env python3
"""Generate the E-series configurations from the stamped Phase-1/Phase-2b
configurations (wp8-lejepa, PREREG_E1 / PREREG_E2 mechanics).

Every E-series run is a `run-config` on a derived configuration:
  * E2 (M = 512 and 1024): the Phase-2b configuration (whose AR cells are
    E2's baseline) with model.kind = bottleneck, model.n_tokens = M,
    model.decode_k = 4 (the continuous decoder, Stage 1.32), E8 as AR
    pretraining only, P3 in zero-shot-only form, and the FE-JEPA-only probes
    (e6, wp6, e1) disabled. Everything else -- corpus, split, seeds, pool,
    epochs, learning rate, numeric policy -- is the baseline's, byte for byte
    (tests/test_e_series_configs.py pins this).
  * E1 (2D; base, shaped, and the one-seed raw-LN ablation): the Phase-1
    configuration with E8 as AR pretraining only and everything else
    disabled. The shaped and raw arms carry `pretrain.loss_spec`; lambda (and,
    for the head arm, the head width) are FILLED at stamping by re-running
    this generator with --e1-lambda / --e1-head-width -- never by hand -- and
    the runner refuses the null placeholders until then.
Each configuration points at its experiment's PREREG file with the guard on;
the PREREG file carries one labelled line per configuration,
`CONFIG_SHA256[<config stem>] = ...` (report.stamp_prereg / verify_prereg).

    python scripts/make_e_series_configs.py [--e1-lambda L --e1-head-width W]
    python scripts/make_e_series_configs.py --e1-from-pilot runs/wp8/e1_pilot.json \
        --fill-prereg PREREG_E1.md                      # stamping time (Stage 1.31)

Stage 1.31: `--e1-from-pilot` reads lambda and the head width from the pilot
JSON itself (refusing a smoke pilot, NO-GO-AT-PILOT, bought labels or a
changed manifest), and `--fill-prereg` writes LAMBDA, WIDTH and the pilot
record's SHA-256 into PREREG_E1.md's parameter line -- the pre-registration
and the configurations are filled from one source, never typed by hand
(tests/test_e_series_configs.py checks that the committed line and the
committed configurations agree).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

PREREG_E1_PARAMS = re.compile(
    r"LAMBDA = `(?P<lam>[^`]*)`; WIDTH = `(?P<width>[^`]*)`; "
    r"pilot record SHA-256 = `(?P<sha>[^`]*)`")
PLACEHOLDER = "<from the pilot>"


def read_pilot(path: str, phase1: dict, phase1_sha256: str | None = None) -> tuple:
    """(lambda, head_width, sha256) from an E1 pilot JSON, or SystemExit."""
    raw = Path(path).read_bytes()
    p = json.loads(raw)
    if p.get("smoke"):
        raise SystemExit(f"{path}: a smoke pilot cannot configure E1")
    if phase1_sha256 and p.get("config_sha256") != phase1_sha256:
        raise SystemExit(f"{path}: the pilot ran on config {str(p.get('config_sha256'))[:12]}, "
                         f"not this Phase-1 config ({phase1_sha256[:12]})")
    if bool((p.get("numeric_policy") or {}).get("tf32")) != bool(phase1.get("tf32", True)):
        raise SystemExit(f"{path}: the pilot's numeric policy {p.get('numeric_policy')} is not "
                         "the E1 runs' TF32 setting")
    if p.get("selected_lambda") is None:
        raise SystemExit(f"{path}: selected_lambda is null -- NO-GO-AT-PILOT (PREREG_E1 Sec. 3); "
                         "no E1 arm is configured or trained")
    if int((p.get("pilot_ledger") or {}).get("total", -1)) != 0:
        raise SystemExit(f"{path}: the pilot bought labels (ledger {p.get('pilot_ledger')})")
    if not p.get("manifest_sha256_before") or \
            p.get("manifest_sha256_before") != p.get("manifest_sha256_after"):
        raise SystemExit(f"{path}: the corpus manifest changed during the pilot")
    want = {"n_val": int(phase1["split"]["n_val"]), "seed": int(phase1["split"]["seed"])}
    if p.get("split") != want:
        raise SystemExit(f"{path}: pilot split {p.get('split')} is not the E1 split {want}")
    return float(p["selected_lambda"]), int(p["head_width"]), hashlib.sha256(raw).hexdigest()


def fill_prereg_e1(path: str, lam: float, width: int, pilot_sha: str) -> None:
    """Write the pilot's values into PREREG_E1.md's parameter line (placeholders
    only; re-filling with the same values is a no-op, different values refuse)."""
    pf = Path(path)
    text = pf.read_text()
    hits = list(PREREG_E1_PARAMS.finditer(text))
    if len(hits) != 1:
        raise SystemExit(f"{path}: expected exactly one LAMBDA/WIDTH/pilot-record line, "
                         f"found {len(hits)}")
    m = hits[0]
    new = {"lam": repr(float(lam)), "width": str(int(width)), "sha": pilot_sha}
    for k, v in new.items():
        if m.group(k) not in (PLACEHOLDER, v):
            raise SystemExit(f"{path}: {k} already filled with {m.group(k)!r}, refusing {v!r}")
    line = (f"LAMBDA = `{new['lam']}`; WIDTH = `{new['width']}`; "
            f"pilot record SHA-256 = `{new['sha']}`")
    pf.write_text(text[:m.start()] + line + text[m.end():])


def _disable_all_but_e8(exps: dict) -> dict:
    out = {}
    for k, v in exps.items():
        if k == "e8":
            out[k] = dict(v, enabled=True, ar_only=True)
        elif k == "p3_transfer":
            out[k] = dict(v, enabled=True, fewshot_budgets=[], naive_budget=0)
        else:
            out[k] = dict(v or {}, enabled=False)
    return out


E2_DECODE_K = 4
"""Tokens blended per node by the E2 decoder (PREREG_E2 r9, PI decision of
29 Sep 2026: continuous Franke-Little blend of the 4 nearest tokens)."""


def e2_config(phase2b: dict, m_tokens: int) -> dict:
    cfg = json.loads(json.dumps(phase2b))
    cfg["model"] = dict(cfg["model"], kind="bottleneck", n_tokens=int(m_tokens),
                        decode_k=E2_DECODE_K)
    cfg["experiments"] = _disable_all_but_e8(cfg["experiments"])
    cfg["out"] = f"runs/e2_m{m_tokens}/report.json"
    cfg["prereg_file"], cfg["prereg_guard"] = "PREREG_E2.md", True
    cfg["_comment"] = (f"E2 arm M={m_tokens}: bottleneck AR only; baseline = the Phase-2b "
                       "report's AR cells (runs/phase2/report_phase2b.json); see PREREG_E2.md")
    return cfg


def e1_config(phase1: dict, arm: str, lam=None, head_width=None) -> dict:
    """arm in {"base", "shaped", "raw_s0"}."""
    cfg = json.loads(json.dumps(phase1))
    cfg["experiments"] = _disable_all_but_e8(cfg["experiments"])
    cfg["experiments"].pop("p3_transfer", None)          # no transfer set in 2D
    cfg["out"] = f"runs/e1_2d_{arm}/report.json"
    cfg["prereg_file"], cfg["prereg_guard"] = "PREREG_E1.md", True
    lam = None if lam is None else float(lam)
    if arm == "shaped":
        cfg.setdefault("pretrain", {})["loss_spec"] = {
            "reg_mode": "sigreg_ep_head", "lambda_reg": lam,             # FILL at stamping
            "sigreg_n_proj": 256,
            "sigreg_head_width": None if head_width is None else int(head_width)}
    elif arm == "raw_s0":                                # ablation: SIGReg on the raw tokens
        cfg.setdefault("pretrain", {})["loss_spec"] = {
            "reg_mode": "sigreg_ep", "lambda_reg": lam, "sigreg_n_proj": 256}
        cfg["experiments"]["e8"]["seeds"] = 1                # one seed (seed 0), reported only
    cfg["_comment"] = f"E1 2D {arm} arm: AR pretraining only; see PREREG_E1.md"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2b", default="configs/phase2b_v1.json")
    ap.add_argument("--phase1", default="configs/phase1_rec8_v2.json")
    ap.add_argument("--e1-lambda", type=float, default=None,
                    help="the pilot-selected lambda (PREREG_E1 Sec. 3); fills shaped and raw")
    ap.add_argument("--e1-head-width", type=int, default=None,
                    help="the pilot's head width (0 = model dim); fills the shaped arm")
    ap.add_argument("--e1-from-pilot", default=None,
                    help="Stage 1.31: take lambda and head width from this pilot JSON")
    ap.add_argument("--fill-prereg", default=None,
                    help="Stage 1.31: with --e1-from-pilot, also fill this PREREG_E1.md's "
                         "LAMBDA / WIDTH / pilot-record line")
    ap.add_argument("--out-dir", default="configs")
    a = ap.parse_args()
    p2b = json.loads(Path(a.phase2b).read_text())
    p1 = json.loads(Path(a.phase1).read_text())
    if a.fill_prereg and not a.e1_from_pilot:
        ap.error("--fill-prereg needs --e1-from-pilot (the PREREG line records the pilot file)")
    if a.e1_from_pilot:
        lam, width, pilot_sha = read_pilot(
            a.e1_from_pilot, p1, hashlib.sha256(Path(a.phase1).read_bytes()).hexdigest())
        for given, got, name in ((a.e1_lambda, lam, "--e1-lambda"),
                                 (a.e1_head_width, width, "--e1-head-width")):
            if given is not None and given != got:
                ap.error(f"{name} {given} contradicts the pilot's {got}")
        a.e1_lambda, a.e1_head_width = lam, width
        if a.fill_prereg:
            fill_prereg_e1(a.fill_prereg, lam, width, pilot_sha)
    out = Path(a.out_dir)
    written = []
    for m in (512, 1024):
        f = out / f"e2_m{m}.json"
        f.write_text(json.dumps(e2_config(p2b, m), indent=1) + "\n"); written.append(str(f))
    for arm in ("base", "shaped", "raw_s0"):
        f = out / f"e1_2d_{arm}.json"
        f.write_text(json.dumps(e1_config(p1, arm, a.e1_lambda, a.e1_head_width), indent=1)
                     + "\n")
        written.append(str(f))
    print("\n".join(written))


if __name__ == "__main__":
    main()
