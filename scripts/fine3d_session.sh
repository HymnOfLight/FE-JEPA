#!/usr/bin/env bash
# fine3d session (PREREG_FINE3D Sec. 8; RUNBOOK_FINE3D Sec. 1). From the repository root:
#   bash scripts/fine3d_session.sh
# Steps, in order: R2a's fresh baseline and S (run-config), the post-hoc amplitude
# pass on each (wp8 4e, --remesh 16), then R3's energy arm and its labelled arm
# (scripts/r3_finetune.py; the labelled arm only once the energy arm's report is
# complete: R3 needs both). A step whose output is complete is skipped; an R2a
# run with states but no report restarts with --reuse-states (D9, recorded in
# its report); an R3 arm resumes by itself.
# A failed step does not stop the next one. Every attempt keeps its log
# (<step>.log.<time>), and status.txt time-stamps each attempt. Before every step
# the checkout must still be the tag, unmodified: `git describe --always --dirty
# --tags` (what every report records) prints exactly the tag, and it is the only
# tag on HEAD; and before every step that runs, the GPU must still be there (a
# step started without it would run on the CPU).
set -u
WANT=prereg-fine3d
P=runs/fine3d/session
mkdir -p "$P"
exec 9>"$P/.lock"
if ! flock -n 9; then
  echo "STOP: the session is already running (tmux ls; pgrep -af fejepa)"; exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "${FINE3D_REHEARSAL:-}" = 1 ]; then
  # the sandbox's CPU rehearsal of this script (BRANCH_NOTES Stage 2); the runbook never sets it
  echo "rehearsal: no GPU check" | tee -a "$P/status.txt"
else
python - <<'PY' || { echo "STOP: no GPU" | tee -a "$P/status.txt"; exit 1; }
import sys
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else 1)
PY
fi

run() {
  local log=$1; shift
  [ -e "$P/$log" ] && mv "$P/$log" "$P/$log.$(date +%Y%m%d-%H%M%S)"
  echo "$log start $(date '+%F %T')" >> "$P/status.txt"
  "$@" 2>&1 | tee "$P/$log"; local rc=${PIPESTATUS[0]}
  echo "$log exit=$rc $(date '+%F %T')" | tee -a "$P/status.txt"
}

code_ok() {
  # the string every report records (fejepa.report._git_describe), and the tag alone on HEAD
  local code tags
  code=$(git describe --always --dirty --tags 2>/dev/null)
  tags=$(git tag --points-at HEAD 2>/dev/null)
  if [ "$code" != "$WANT" ] || [ "$tags" != "$WANT" ] \
      || [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "STOP before $1: the code is ${code:-unknown} (tags on HEAD: ${tags:-none}), or modified, not $WANT" \
      | tee -a "$P/status.txt"
    return 1
  fi
}

gpu_ok() {
  # before every step that runs, as before the session
  [ "${FINE3D_REHEARSAL:-}" = 1 ] && return 0
  timeout 300 python - <<'PY' >/dev/null 2>&1 && return 0
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  echo "STOP before $1: no GPU" | tee -a "$P/status.txt"
  return 1
}

r2a() {
  local name=$1 rep="runs/fine3d/$1/report.json" extra=""
  code_ok "$name" || return 1
  if [ -f "$rep" ]; then
    if python -c 'import json,sys; sys.exit(0 if "provenance" in json.load(open(sys.argv[1])) else 1)' \
        "$rep" 2>/dev/null; then
      echo "$name: report exists, skipped" | tee -a "$P/status.txt"; return 0
    fi
    mv "$rep" "$rep.unreadable.$(date +%Y%m%d-%H%M%S)"
    echo "$name: unreadable report moved aside" | tee -a "$P/status.txt"
  fi
  gpu_ok "$name" || return 1
  if ls "runs/fine3d/$name/e8_states/"*.pt >/dev/null 2>&1 \
      || ls "runs/fine3d/$name/e8_states/"*.ckpt >/dev/null 2>&1; then
    extra="--reuse-states"; echo "$name: restarting from its states" | tee -a "$P/status.txt"
  fi
  run "$name.log" python -m fejepa.cli run-config "configs/$name.json" --label-workers 1 $extra
}

amp() {
  local name=$1 rep="runs/fine3d/$1/report.json" out="runs/fine3d/posthoc/amp_$1.json"
  code_ok "amp_$name" || return 1
  if [ ! -f "$rep" ]; then
    echo "amp_$name: no report of $name, skipped" | tee -a "$P/status.txt"; return 1
  fi
  if [ -f "$out" ] && python - "$out" "$rep" <<'PY' 2>/dev/null
import hashlib, json, sys
d = json.load(open(sys.argv[1]))
seeds = list(d["seeds"].values())
ok = bool(seeds) and all((v.get(s) or {}).get("summary", {}).get("complete")
                         for v in seeds for s in ("inband", "fine"))
ok = ok and all(len((v.get("remesh") or {}).get("rows", [])) == 16 for v in seeds)
# measured on this report (an R2a arm run again leaves a record of the old one)
ok = ok and d.get("report_sha256") == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
sys.exit(0 if ok else 1)
PY
  then
    echo "amp_$name: record complete, skipped" | tee -a "$P/status.txt"; return 0
  fi
  gpu_ok "amp_$name" || return 1
  run "amp_$name.log" python scripts/posthoc_amplitude.py --report "$rep" \
      --states-dir "runs/fine3d/$name/e8_states" --remesh 16 --out "$out"
}

r3_done() {
  # an R3 arm's report with all three seeds
  [ -f "runs/fine3d/$1/report.json" ] && python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if sorted(d.get("per_seed", {})) == ["s0", "s1", "s2"] else 1)' \
      "runs/fine3d/$1/report.json" 2>/dev/null
}

r3() {
  # $1 the arm; $2 (optional) the arm whose complete report it needs first
  local name=$1 need=${2:-}
  code_ok "$name" || return 1
  if r3_done "$name"; then
    echo "$name: report exists, skipped" | tee -a "$P/status.txt"; return 0
  fi
  if [ -n "$need" ] && ! r3_done "$need"; then
    echo "$name: not run, $need has no complete report (R3 needs both arms)" \
      | tee -a "$P/status.txt"; return 1
  fi
  gpu_ok "$name" || return 1
  run "$name.log" python scripts/r3_finetune.py "configs/$name.json"
}

r2a r2a_b_m1024
r2a r2a_s_m1024
amp r2a_b_m1024
amp r2a_s_m1024
r3 r3_eft_b16
r3 r3_lft_b16 r3_eft_b16
cat "$P/status.txt"
