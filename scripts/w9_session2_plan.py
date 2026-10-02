#!/usr/bin/env python3
"""wp9 Stage 0b: session 2's plan from session 1's decisions (PREREG_W9 Sec. 4
and 8). Checks the gate, selects the stamped configurations rules 1-2 admit,
sets the run-time flags, pre-flights each configuration (`run-config
--dry-run`: guard verified, holdouts and reused states verified), and writes
the command script session 2 runs.

Gate (any failure: STOP, no command script, exit non-zero):
  * the required session-1 steps exited 0 (ood2d, c0_val, c0_amp2d,
    c0_memory); rules 4-5, the train/val reading and the timing are not
    required;
  * the decision file is the frozen rules' output (their SHA-256) on the
    readings in the session-1 directory (their SHA-256), recomputed here and
    identical, written on the expected commit (`prereg-w9`, which this
    checkout must also be); rules 1-2 decided and rule 2 found a pool; rule 3
    undecided keeps checkpointing on;
  * the seven evaluation sets complete in the record, each of PREREG_W9's size
    and seed, and the manifests on disk and those the amplitude reading used
    are the record's;
  * the C0 evaluation reproduced E1's per-instance validation arrays
    (largest relative deviation <= 1e-4);
  * a GPU, and E1's torch version; at least 5 GB free on the data disk;
  * every selected configuration passes its dry run (stamped and verified).

Order: the 1,024 arm (evaluation only), N_max, the fresh 1,024 baseline
(seeds 3-5; in every session 2 since PREREG_W9 r3), S if rule 1 admitted it,
then 4,096: the primary comparisons first. Workers: rule 2's for N_max, the configuration's (3)
otherwise. Activation checkpointing: rule 3 for every trained arm, kept on for
an arm whose memory would not fit without it (the frozen memory rule's own
test), and on when rule 3 is undecided.

The command script refuses to run twice at once and without E1's GPU stack,
skips an arm whose report is complete, moves an unreadable report aside,
restarts an arm whose states exist without a report (`--reuse-states`, D9;
recorded in the report), keeps every attempt's log and time-stamps each
attempt in status.txt -- so the same command serves the first run and every
restart:

    python scripts/w9_session2_plan.py --session1 runs/w9/session1 --out-dir runs/w9/session2
    bash runs/w9/session2/commands.sh
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_STEPS = ("ood2d.log", "c0_val.log", "c0_amp2d.log", "c0_memory.log")
FAMILIES = ("IB", "F1", "F2", "F3", "F4", "F5", "R")
REPRO_MAX = 1e-4
MIN_FREE_GB = 5.0
NMAX_CONFIG = {25600: "w9_c1_n25600", 12800: "w9_c1_n12800"}
ARM_POOL = {"w9_c1_n1024": 1024, "w9_c1_n4096": 4096, "w9_b_n1024": 1024, "w9_s_n1024": 1024,
            "w9_c1_n25600": 25600, "w9_c1_n12800": 12800}
E1_REPORT = "records/wp8/e1/e1_2d_base/report.json"
SESSION1_FILES = {"amp2d": "c0_amp2d.json", "timing": "profile_2d_w9.json",
                  "memory": "c0_memory.json", "trainval": "c0_trainval.json",
                  "val": "c0_val.json"}
"""The decision script's inputs, by its argument names."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rules():
    path = ROOT / "scripts" / "w9_session1_decisions.py"
    spec = importlib.util.spec_from_file_location("w9_rules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def _git_describe() -> str:
    from fejepa.report import _git_describe as describe

    return describe()


def _readings(s1: Path) -> dict:
    """The decision script's inputs as it reads them (None when missing or
    malformed)."""
    out = {}
    for k, f in SESSION1_FILES.items():
        try:
            out[k] = json.loads((s1 / f).read_text())
        except (OSError, ValueError):
            out[k] = None
    return out


def _status(path: Path) -> dict:
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            m = re.match(r"(\S+) exit=(\d+)", line.strip())
            if m:
                out[m.group(1)] = int(m.group(2))
    return out


def gate(s1: Path, repo: Path = Path("."), gpu_check: bool = True,
         expected_git: str | None = None) -> tuple:
    """(problems, decisions, facts) from session 1's directory. With
    `expected_git`, the decisions and this checkout must both describe as it."""
    from fejepa.data.archive import manifest_sha256
    from fejepa.fe.ood2d import DEFAULT_SEEDS, SET_SIZES

    problems, facts = [], {}
    st = _status(s1 / "status.txt")
    for step in REQUIRED_STEPS:
        if st.get(step) != 0:
            problems.append(f"session 1 step {step}: exit {st.get(step, 'missing')}")
    facts["status"] = st
    try:
        dec = json.loads((s1 / "decisions.json").read_text())
    except (OSError, ValueError) as exc:
        return problems + [f"decisions.json unreadable ({exc})"], None, facts
    rules, rules_path = _rules()
    if dec.get("rules_sha256") != _sha(rules_path):
        problems.append("decisions.json was not written by the frozen rules (rules_sha256)")
    recorded = dec.get("inputs_sha256") or {}
    for f in SESSION1_FILES.values():
        p = s1 / f
        if f in recorded and (not p.is_file() or _sha(p) != recorded[f]):
            problems.append(f"decisions.json was computed from another {f} than the one here")
        elif f not in recorded and p.is_file():
            problems.append(f"decisions.json was computed without the {f} that is here")
    d = dec.get("decisions", {})
    again = rules.decide(**_readings(s1))
    for rule in ("S_enters", "pool_max", "checkpointing_off"):
        if json.dumps((again.get(rule) or {}).get("value"), sort_keys=True) != \
                json.dumps((d.get(rule) or {}).get("value"), sort_keys=True):
            problems.append(f"rule {rule} recomputed from the readings here gives "
                            f"{(again.get(rule) or {}).get('value')}, decisions.json says "
                            f"{(d.get(rule) or {}).get('value')}")
    if expected_git is not None:
        here = _git_describe()
        facts["git"] = here
        if dec.get("git") != expected_git:
            problems.append(f"decisions.json was written on {dec.get('git')!r}, not "
                            f"{expected_git!r}")
        if here != expected_git:
            problems.append(f"this checkout is {here!r}, not {expected_git!r}")
    for rule in ("S_enters", "pool_max"):
        if (d.get(rule) or {}).get("value") is None:
            problems.append(f"rule {rule} undecided ({(d.get(rule) or {}).get('error')})")
    if (d.get("pool_max") or {}).get("value") is not None and d["pool_max"].get("n") is None:
        problems.append("rule 2: no pool fits -- the pool design is revisited before session 2")
    try:
        ood = json.loads((s1 / "ood2d.json").read_text())
        fam = ood.get("families", {})
        bad = [f for f in FAMILIES if (fam.get(f) or {}).get("status")
               not in ("generated", "verified existing")]
        if bad or ood.get("failed"):
            problems.append(f"evaluation sets incomplete: {sorted(set(bad) | set(ood.get('failed') or []))}")
        rec = {f: (fam.get(f) or {}).get("manifest_sha256") for f in FAMILIES}
        facts["ood_manifests"] = rec
        for f in FAMILIES:                               # PREREG_W9 Sec. 3's sizes and seeds
            e = fam.get(f) or {}
            if e.get("manifest_sha256") and \
                    [e.get("n_instances"), e.get("seed")] != [SET_SIZES[f], DEFAULT_SEEDS[f]]:
                problems.append(f"{f}: {e.get('n_instances')} instances from seed "
                                f"{e.get('seed')}; PREREG_W9 Sec. 3 fixes {SET_SIZES[f]} from "
                                f"seed {DEFAULT_SEEDS[f]}")
        for f in FAMILIES:
            p = repo / "runs" / "w9" / "ood2d" / f
            if rec.get(f) and (not (p / "manifest.json").is_file() or manifest_sha256(p) != rec[f]):
                problems.append(f"runs/w9/ood2d/{f}: the manifest on disk is not session 1's record")
        try:
            amp = json.loads((s1 / "c0_amp2d.json").read_text())
            for f in ("F5", "R"):
                if (amp.get(f) or {}).get("manifest_sha256") != rec.get(f):
                    problems.append(f"the amplitude reading used another {f} than the record")
        except (OSError, ValueError):
            pass                                          # c0_amp2d's own step is required above
    except (OSError, ValueError) as exc:
        problems.append(f"ood2d.json unreadable ({exc})")
    try:
        rep = json.loads((s1 / "c0_val.json").read_text())["summary"]["reproduction"]
        vals = [float(v) for v in (*rep["disp"].values(), *rep["egap"].values())]
        dev = max(vals) if vals and all(math.isfinite(v) for v in vals) else float("inf")
        facts["reproduction_max_rel_dev"] = dev
        if not dev <= REPRO_MAX:
            problems.append(f"E1's validation arrays not reproduced (max relative deviation "
                            f"{dev:.3g} > {REPRO_MAX})")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        problems.append(f"c0_val.json: no reproduction reading ({exc})")
    try:
        mem = json.loads((s1 / "c0_memory.json").read_text())
        facts["memory"] = mem
        lim = mem.get("limits") or {}
        facts["usable_host_gb"] = round(float(lim.get("usable", -1)) / 1e9, 1)
        facts["usable_host_source"] = lim.get("usable_source")
    except (OSError, ValueError):
        facts["memory"] = None
    if gpu_check:
        try:
            import torch

            want = json.loads((repo / E1_REPORT).read_text())["provenance"]["versions"]["torch"]
            facts["torch"] = torch.__version__
            if not torch.cuda.is_available() or torch.__version__ != want:
                problems.append(f"no GPU or not E1's torch (torch {torch.__version__}, cuda "
                                f"{torch.cuda.is_available()}; E1 ran {want})")
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"GPU check failed ({type(exc).__name__}: {exc})")
    free = shutil.disk_usage(repo).free / 1e9
    facts["free_gb"] = round(free, 1)
    if free < MIN_FREE_GB:
        problems.append(f"only {free:.1f} GB free on the data disk (needs {MIN_FREE_GB})")
    return problems, dec, facts


def select(dec: dict, memory: dict | None = None) -> tuple:
    """([(arm, config stem, run-time flags)] in run order, notes)."""
    rules, _ = _rules()
    d = dec["decisions"]
    notes = []
    r3 = d.get("checkpointing_off", {}).get("value")
    if r3 is None:
        notes.append("rule 3 undecided: checkpointing on (memory and time only)")
    off = bool(r3)
    nmax_stem = NMAX_CONFIG[int(d["pool_max"]["n"])]
    workers3 = int((memory or {}).get("workers", 3))

    def ckpt(stem, k):
        if not off:
            return "on"
        if memory is not None and stem != nmax_stem:     # rule 3 already tested N_max
            try:
                if not rules.fits(memory, ARM_POOL[stem], k, ckpt_on=False):
                    notes.append(f"{stem}: checkpointing kept on (would not fit without it)")
                    return "on"
            except Exception as exc:                      # noqa: BLE001
                notes.append(f"{stem}: checkpointing kept on (fit not computable: {exc})")
                return "on"
        return "off"

    nk = int(d["pool_max"].get("k") or (1 if d["pool_max"].get("mode") == "sequential"
                                        else workers3))
    arms = [("c1_n1024", "w9_c1_n1024", [])]
    arms.append((nmax_stem[3:], nmax_stem, ["--workers", str(nk),
                                            "--activation-checkpointing", ckpt(nmax_stem, nk)]))
    # r3: the fresh baseline runs in every session 2 (H2's reference, and the
    # on-box check of the training path behind the reuse); S only if admitted
    for stem in ("w9_b_n1024", "w9_s_n1024") if d["S_enters"]["value"] else ("w9_b_n1024",):
        arms.append((stem[3:], stem, ["--workers", str(workers3),
                                      "--activation-checkpointing", ckpt(stem, workers3)]))
    arms.append(("c1_n4096", "w9_c1_n4096",
                 ["--workers", str(workers3),
                  "--activation-checkpointing", ckpt("w9_c1_n4096", workers3)]))
    return arms, notes


def preflight(arms: list, configs: Path) -> list:
    """Dry-run every selected configuration with its run-time flags."""
    from fejepa.experiments.runner import run_config

    problems = []
    for _arm, stem, flags in arms:
        workers = int(flags[flags.index("--workers") + 1]) if "--workers" in flags else None
        ck = (flags[flags.index("--activation-checkpointing") + 1] == "on"
              if "--activation-checkpointing" in flags else None)
        try:
            s = run_config(str(configs / f"{stem}.json"), dry_run=True,
                           workers_override=workers, activation_checkpointing=ck)
        except (Exception, SystemExit) as exc:            # noqa: BLE001
            problems.append(f"{stem}: dry run failed ({type(exc).__name__}: {exc})")
            continue
        if s.get("prereg_status") != "verified":
            problems.append(f"{stem}: pre-registration {s.get('prereg_status')}")
        if (s.get("w9") or {}).get("error"):
            problems.append(f"{stem}: {s['w9']['error']}")
    return problems


def script(arms: list, dec_sha: str, out_dir: str, repo: str, gpu_check: bool = True,
           expected_git: str | None = None) -> str:
    q = shlex.quote
    lines = ["#!/usr/bin/env bash",
             "# wp9 session 2 -- generated by scripts/w9_session2_plan.py from session 1's",
             f"# decisions (SHA-256 {dec_sha}).  Run:  bash {out_dir}/commands.sh",
             "# A complete report: the arm is skipped. An unreadable report is moved aside.",
             "# States without a report: the arm restarts (--reuse-states, recorded in its",
             "# report). Every attempt keeps its log; status.txt time-stamps each attempt.",
             "set -u",
             f"cd {q(repo)} || exit 1",
             f"P={q(out_dir)}; mkdir -p \"$P\"",
             'exec 9>"$P/.lock"',
             'if ! flock -n 9; then echo "STOP: session 2 is already running (tmux ls; '
             'pgrep -af fejepa.cli)"; exit 1; fi']
    if gpu_check:
        lines += ["python - <<'PY' || { echo \"STOP: no GPU, or torch is not E1's\" | "
                  "tee -a \"$P/status.txt\"; exit 1; }",
                  "import json, sys, torch",
                  f"want = json.load(open({E1_REPORT!r}))['provenance']['versions']['torch']",
                  "print('torch', torch.__version__, '| E1', want, '| cuda', "
                  "torch.cuda.is_available())",
                  "sys.exit(0 if torch.cuda.is_available() and torch.__version__ == want else 1)",
                  "PY"]
    lines += ["run() {",
              "  local log=$1; shift",
              '  [ -e "$P/$log" ] && mv "$P/$log" "$P/$log.$(date +%Y%m%d-%H%M%S)"',
              '  echo "$log start $(date \'+%F %T\')" >> "$P/status.txt"',
              '  "$@" 2>&1 | tee "$P/$log"; local rc=${PIPESTATUS[0]}',
              '  echo "$log exit=$rc $(date \'+%F %T\')" | tee -a "$P/status.txt"',
              "}",
              "arm() {",
              "  local name=$1; shift",
              '  local rep="runs/w9/$name/report.json"',
              *([f"  local want={q(expected_git)} code",
                 "  code=$(python -c 'from fejepa.report import _git_describe as d; print(d())')",
                 '  if [ "$code" != "$want" ]; then echo "STOP before $name: the code is '
                 '$code, not $want" | tee -a "$P/status.txt"; return 1; fi']
                if expected_git is not None else []),
              '  if [ -f "$rep" ]; then',
              "    if python -c 'import json,sys; sys.exit(0 if \"provenance\" in "
              "json.load(open(sys.argv[1])) else 1)' \"$rep\" 2>/dev/null; then",
              '      echo "$name: report exists, skipped" | tee -a "$P/status.txt"; return; fi',
              '    mv "$rep" "$rep.unreadable.$(date +%Y%m%d-%H%M%S)"',
              '    echo "$name: unreadable report moved aside" | tee -a "$P/status.txt"',
              "  fi",
              '  local extra=""',
              '  if ls "runs/w9/$name/e8_states/"*.pt >/dev/null 2>&1 || '
              'ls "runs/w9/$name/e8_states/"*.ckpt >/dev/null 2>&1; then',
              '    extra="--reuse-states"; echo "$name: restarting from its states" | '
              'tee -a "$P/status.txt"; fi',
              '  run "$name.log" python -m fejepa.cli run-config "$@" $extra',
              "}"]
    for arm, stem, flags in arms:
        lines.append(f"arm {arm} configs/{stem}.json {' '.join(flags)}".rstrip())
    lines.append('cat "$P/status.txt"')
    return "\n".join(lines) + "\n"


def summary_line(dec: dict, arms: list, facts: dict) -> str:
    d = dec["decisions"]
    nmax = arms[1]
    flags = nmax[2]
    s = "yes" if d["S_enters"]["value"] else "no"
    return (f"S admitted: {s} | N_max {int(d['pool_max']['n']):,} with "
            f"{flags[flags.index('--workers') + 1]} worker(s) at a time | checkpointing "
            f"{flags[flags.index('--activation-checkpointing') + 1]} | usable host memory "
            f"{facts.get('usable_host_gb')} GB ({facts.get('usable_host_source')}) | E1 "
            f"reproduction {facts.get('reproduction_max_rel_dev')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session1", default="runs/w9/session1")
    ap.add_argument("--out-dir", default="runs/w9/session2")
    ap.add_argument("--configs", default=None, help="default: <repo>/configs")
    ap.add_argument("--repo", default=".", help="the repository root the runs use")
    ap.add_argument("--expected-git", default="prereg-w9",
                    help="the commit description the decisions and the runs must have")
    ap.add_argument("--no-preflight", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)  # rehearsals
    a = ap.parse_args()
    s1, out, repo = Path(a.session1), Path(a.out_dir).resolve(), Path(a.repo).resolve()
    configs = Path(a.configs) if a.configs else repo / "configs"
    problems, dec, facts = gate(s1, repo, gpu_check=not a.allow_cpu,
                                expected_git=a.expected_git)
    arms, notes = ([], [])
    if dec is not None and not problems:
        arms, notes = select(dec, facts.get("memory"))
        if not a.no_preflight:
            problems += preflight(arms, configs)
    dec_sha = _sha(s1 / "decisions.json") if (s1 / "decisions.json").is_file() else None
    plan = {"what": "wp9 session-2 plan (PREREG_W9 Sec. 4, 8)", "session1": str(s1),
            "decisions_sha256": dec_sha, "gate_problems": problems, "notes": notes,
            "facts": {k: v for k, v in facts.items() if k != "memory"},
            "arms": [{"arm": arm, "config": f"configs/{stem}.json", "flags": flags}
                     for arm, stem, flags in arms] if not problems else []}
    out.mkdir(parents=True, exist_ok=True)
    if problems:
        (out / "commands.sh").unlink(missing_ok=True)
        (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
        print("STOP -- session 2 is not planned:\n  " + "\n  ".join(problems), flush=True)
        raise SystemExit(1)
    plan["summary"] = summary_line(dec, arms, facts)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    (out / "commands.sh").write_text(script(arms, dec_sha, str(out), str(repo),
                                            gpu_check=not a.allow_cpu,
                                            expected_git=a.expected_git))
    print(json.dumps(plan["arms"], indent=1))
    for n in notes:
        print(f"note: {n}")
    print(plan["summary"])
    print(f"GO -- bash {out}/commands.sh", flush=True)


if __name__ == "__main__":
    main()
