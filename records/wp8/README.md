# records/wp8 -- the E-series records: what the stamps rest on, and what the runs returned

All files are copied byte for byte from the box's return packages (host
`autodl-container-857b4c9493-2d948778`, RTX 5090, torch 2.12.1+cu130).

## Stamp records (Stage 1.34)

Returned on 29 September 2026 in `wp8_pilot_bench_return_20260930.tgz`
(SHA-256 `8ff3782853645297646314da5b5787a933e96959a84e9c8034c1ec7f12ded34d`;
code `cc9a168`, 281 tests passed there).

- `e1_pilot.json` -- the E1 lambda pilot (PREREG_E1 Sec. 3): lambda = 1.0, head width 6.
  Its SHA-256 is in PREREG_E1's parameter line (Sec. 2).
- `bench_e2_m512.json`, `bench_e2_m1024.json` -- the E2 bench (PREREG_E2 Sec. 3). Their
  SHA-256 are in PREREG_E2 Sec. 3; the E2 verdicts read these files.

`tests/test_e_series_records.py` keeps these files, the two pre-registrations and the
configurations in agreement.

## Run records (Stage 1.35)

Both experiments ran on the stamped commit `cdac731` (tags `prereg-e1`, `prereg-e2`;
286 tests passed on the box), one attempt each, no resumption, solve ledger 0.

- `e1/` -- from `wp8_e1_return_20260930.tgz` (SHA-256
  `7343dad3a98fe1a43343fca9fdf0c799dc05a79db5734e4e16f8f0fbcd42b7a0`): the three arms'
  reports and run logs (`e1_2d_base/`, `e1_2d_shaped/`, `e1_2d_raw_s0/`), the seven
  separation readings (`sep_*.json`), the verdict (`e1_verdict.json`: NO-GO) and the
  box provenance (`provenance.txt`, state SHA-256).
- `e2/` -- from `wp8_e2_return_20261001.tgz` (SHA-256
  `fabc65d377785ac485c163f66aef04284bab84d93d63edf9e8b4a30955b31e55`): the two
  bottleneck runs' reports and run logs (`e2_m512/`, `e2_m1024/`), the verdicts
  (`e2_verdict_M512.json`, `e2_verdict_M1024.json`: both KILLED by K1) and the box
  provenance.
- `e2/baseline/report_phase2b.json` -- the Phase-2b report E2 is judged against
  (SHA-256 `320b6db5060ecae9f4747327228c7705d30877bece67d09af2f68d6c466f2794`, as
  PREREG_E2 Sec. 2 records), copied from the Phase-2b return package of 28 September 2026.
- `restore_data2d.log` -- the restoration of the Phase-1 2D corpus from the data disk's
  archive before the pilot (manifest `3553396d...` unchanged).

`tests/test_e_series_verdicts.py` re-derives every verdict from these files with the
committed adjudicators, checks that each verdict was computed from exactly these files
(input SHA-256), and ties each report to its stamp line.
