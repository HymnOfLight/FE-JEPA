# Addendum to RUNBOOK_C13.md

11 October 2026. Written after the stamp of 10 October 2026 (tag `prereg-c13`, commit `002d19e`). It changes no command of `RUNBOOK_C13.md`, no code and no configuration. It records what the operator was given beyond the runbook.

The operator works from a Chinese translation of `RUNBOOK_C13.md`. Its commands are those of the runbook, verbatim, and the English runbook is authoritative. The translation adds the three items below and nothing else.

## 1. The repository moved

- On 11 October 2026 the repository moved to `https://github.com/RuifengCao/label-free-fe-surrogates`. The old address redirects.
- The move changed no commit. At the new address, the branch `wp10-joint` and the tag `prereg-c13` point to commit `002d19e`.
- This was checked on 11 October 2026 with a fresh single-branch clone of `wp9-pool` from the new address. Section 0 of the runbook ran on it: the fetch, the worktree, the hash check (23 files) and the 30 tests.
- The box keeps its checkout at `~/autodl-tmp/FE-JEPA`, and the package keeps its name `fejepa`.
- Before section 0b the operator looks at the remote address of that checkout. If it is still the old one, the operator updates it as the owner's migration note for the box describes; the redirect also works.

## 2. A second route for section 0b

If the direct fetch fails, or has not ended after 20 minutes, the operator fetches the tag from a git bundle and goes on with the worktree:
```bash
cd ~/autodl-tmp/FE-JEPA
sha256sum ~/wp10-joint.bundle                # 7f0305953c4ef30d464a64a282d64a92d2b21ed1d1af5c98d729e9f00242b1d9
git fetch ~/wp10-joint.bundle +refs/tags/prereg-c13:refs/tags/prereg-c13 && echo FETCH-OK || echo FETCH-FAILED
git rev-parse --short 'prereg-c13^{commit}'  # 002d19e
git worktree add ~/autodl-tmp/FE-JEPA-c13 prereg-c13 2>/dev/null || (cd ~/autodl-tmp/FE-JEPA-c13 && git checkout --detach prereg-c13)
mkdir -p ~/autodl-tmp/c13/records
```
- The bundle holds the branch and the annotated tag on top of commit `92dfc93` (`prereg-w9`), which the box has.
- It was tested on a fresh single-branch clone of `wp9-pool`: the tag resolved to `002d19e`, with the 112 files of `joint/`.

## 3. A provenance line

wp9's session 2 ran on another host with another GPU driver, and its records named neither the host nor the driver nor the CPU (`DEVIATIONS_W9.md` on `wp9-pool`, W1 to W3). The operator therefore runs this line after section 0c, before sections 3 and 5, and whenever the instance has been restarted:
```bash
{ date; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; lscpu | grep -m1 "Model name"; nproc; for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do [ -r $f ] && echo "$f $(cat $f)"; done; } 2>&1 | tee -a ~/autodl-tmp/c13/records/provenance.log
```
- It writes into `records/`, so section 7 returns it.
- The operator reports a restart of the instance, or a change of host (another host name or driver). Training resumes from its checkpoints either way.
