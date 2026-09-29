# records/wp8 -- the records the E-series stamp rests on

Returned from the box on 29 September 2026 in `wp8_pilot_bench_return_20260930.tgz`
(SHA-256 `8ff3782853645297646314da5b5787a933e96959a84e9c8034c1ec7f12ded34d`; host
`autodl-container-857b4c9493-2d948778`, RTX 5090, torch 2.12.1+cu130, code `cc9a168`,
281 tests passed there) and copied byte for byte.

- `e1_pilot.json` -- the E1 lambda pilot (PREREG_E1 Sec. 3): lambda = 1.0, head width 6.
  Its SHA-256 is in PREREG_E1's parameter line (Sec. 2).
- `bench_e2_m512.json`, `bench_e2_m1024.json` -- the E2 bench (PREREG_E2 Sec. 3). Their
  SHA-256 are in PREREG_E2 Sec. 3; the E2 verdicts read these files (RUNBOOK_E_SERIES 2d).

`tests/test_e_series_records.py` keeps these files, the two pre-registrations and the
configurations in agreement.
