# W9 DEVIATIONS LEDGER

Companion to `PREREG_W9.md` (r3, stamped 9 October 2026, tag `prereg-w9`,
self-hash `0f61e286...`). The stamped file is never edited. Its Sec. 8 has
the adjudication record some deviations itself: a planned arm without a
report, run-time settings other than the plan's, restarts and repeated
attempts, resumed units, a non-zero solve ledger, and a software stack or GPU
other than E1's. The verdict file (`records/wp9/w9_verdict.json`) lists none.
Its Sec. 10 has every departure from the pre-registration or the runbook
recorded. This ledger records the departures the adjudication cannot see,
from the returns. No criterion, threshold, configuration or verdict is
changed by any entry below. `tests/test_w9_returns.py` checks that each
host, driver, CPU, time and percentage quoted here sits with the facts the
records give for it.

## W1 -- session 2 ran on a third host, with another GPU driver (10-11 October 2026)

**Facts.**
- E1 trained its states on host `autodl-container-857b4c9493-2d948778` with
  NVIDIA driver 595.71.05 (`records/wp8/e1/provenance.txt`), a Xeon Platinum
  8470Q with 25 CPUs allotted, as RUNBOOK_W9 states and wp8's post-hoc timing
  records with the same driver (`records/wp8/posthoc/profile_2d_head.json`,
  which names no host).
- Session 1 ran on a Xeon Gold 6459C with 16 CPUs allotted and driver
  595.71.05, as its timing record shows
  (`records/wp9/session1/profile_2d_w9.json`). Its provenance file was
  written later, for the return, with the instance in AutoDL's mode without
  a GPU (W2); it names the host `autodl-container-pwr6zxz5jh-5286d5a4` and
  records none of the session's own limits. That session 1 ran on this host
  is inferred: CM2D's run of 9 October names it with 16 CPUs and driver
  595.71.05 (branch cmame-paper, `records/cmame/cm2d/return/provenance.txt`).
- Session 2 ran from 18:50 on 10 October to 03:34 on 11 October (+0800) on
  host `autodl-container-8j9m4jx891-ed8b652f`, with 25 CPUs and driver
  580.95.05 (`records/wp9/session2/provenance.txt`). RUNBOOK_W9 Sec. 2d's
  provenance block prints neither the CPU model nor the memory limit.
- Sec. 4, run about seven hours later with the same CPU count, driver and
  GPU, records a Xeon Platinum 8470Q and the container memory limit of
  98,784,247,808 bytes, the limit of session 1's host
  (`records/wp9/threads/machine.txt`, which names no host). That session 2
  ran on this CPU and under this limit, and that Sec. 4 ran on session 2's
  host, is inferred from it.
- The GPU model (RTX 5090) and torch (2.12.1+cu130) were E1's and session
  1's throughout, and Python, numpy and SciPy were E1's (session 1's records
  name only torch among them). Both sessions ran the stamped code
  (`prereg-w9`); E1 ran its own (`prereg-e1`).

The operator instruction for session 2 (10 October 2026; not in the
repository) asked that a change of host be reported before session 2
started; none was made, and the return was the first notice of the change.
Neither PREREG_W9 nor the runbook requires that report. RUNBOOK_W9 Sec. 2a
ties its host check to the container's memory limit, which the operator
reads beside the plan's summary line; the summary line itself quotes session
1's memory reading, and the return does not hold the operator's reading. The
new host had the same limit, so that check could not have shown this change;
a host check that is to catch one needs the driver, the CPU model or the
host name.

**Why it is recorded here.** PREREG_W9 Sec. 8 has the adjudication record a
software stack or GPU other than E1's, because H1 compares with states
trained on E1's stack. The adjudication compares the torch version and the
GPU model that each report records. The reports do not record the driver, so
the verdict file lists no deviation. The driver is part of the stack, and
its change is recorded here.

**What the records show about its effect.**
- Inference. The 1,024 arm, E1's three states evaluated in session 2,
  reproduced E1's per-instance validation arrays exactly (largest relative
  deviation 0; `arms.n1024.reproduction_max_rel_dev` in the verdict file).
  The plan's summary line quotes 1.6445513362704568e-10 instead: session 1's
  reading through its own script (`records/wp9/session1/c0_val.json`), whose
  energy gaps differed from E1's by up to that much and whose displacement
  errors were exact. The 1,024 arm's per-instance values on the validation
  split, F5 and R equal session 1's evaluation of the same states on the
  same instances, on the other host and driver
  (`records/wp9/session1/c0_amp2d.json`): the energy gap before and after
  scaling by c*, and c* itself, are identical, and the displacement error,
  before and after scaling, agrees to within a relative 1e-15 (identical on
  the validation split).
- Training. The fresh baseline, trained in session 2 with E1's configuration
  on seeds 3-5, against E1's states (`replication_b1024_vs_n1024` in the
  verdict file): within the guard on every in-band reading, with the
  relative energy gap -2.8% on the validation split and -3.3% on IB and the
  displacement error -2.9% and -3.2% (tau 10%). Of the 43 comparisons on all
  sets, two cross the guard: the displacement error on F5, +11.6% (worse),
  and the c*-scaled energy gap on F3, -12.3% (lower). Without a true
  difference, at seed noise of 10-20%, about 4-6% of such comparisons cross
  the guard each way (PREREG_W9 Sec. 7). These comparisons differ in seeds as
  well as in host.

**Bearing on the verdicts.** H2 compares two arms trained in session 2 on
the same host and stack, so the change does not enter its comparison. H1
compares the 25,600 arm (session 2's host) with E1's states (E1's host).
Against the fresh baseline, trained on the 25,600 arm's own host and stack,
the 25,600 arm differs by +1.1% on the validation split and +0.8% on IB in
relative energy gap (`nmax_vs_b1024_exploratory`): on one stack, too, the
larger pool shows no reduction.

## W2 -- the instance was shut down between session 1 and session 2 (10 October 2026)

**Facts.** RUNBOOK_W9 Sec. 1f has the operator send session 1's return and
go on with Sec. 2 without shutting the instance down. Session 1's return was
assembled with the instance in AutoDL's mode without a GPU: its provenance
file shows 1 CPU, `cpu.max` 50000 100000, a memory limit of 2,147,483,648
bytes and no GPU line (`records/wp9/session1/provenance.txt`). That mode is
chosen when an instance is started, so the instance had been shut down after
session 1 (inferred); session 2 started later, on another host (W1).

**Against the pre-registration.** PREREG_W9 Sec. 8 lets the two sessions
run "in one visit or two", and the session-2 plan checked session 1's files
on disk against session 1's record before any arm ran. The departure is from
the runbook's instruction only.

## W3 -- Sec. 4 ran with 25 threads (11 October 2026)

**Facts.** RUNBOOK_W9 Sec. 4 was to time the CMAME paper's CPU solvers in
the environment of the paper's cost table, which ran on 8 October on a Xeon
Gold 6459C with 16 CPUs and `OMP_NUM_THREADS` and `MKL_NUM_THREADS` 16. It
ran with session 2's CPU count, driver and GPU (W1), in an environment that
sets both to 25. Its precheck recorded this
(`records/wp9/threads/precheck.txt`), and the runbook says to note it and go
on. The two BLAS pools that numpy and SciPy load (both OpenBLAS) followed the
environment: 25 threads each, and 1 each with one thread (`blas_env.log`,
`blas_one.log`). Nothing in PREREG_W9 reads Sec. 4; `records/wp9/README.md`
summarises the timings.
