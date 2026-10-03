# records/phase1 -- the two-dimensional runs the CMAME manuscript reports

Copies of three reports that until now were only in the authors' archives
(PROVENANCE_NOTE.md records their bundles). Committed unchanged so that the
manuscript's two-dimensional numbers are generated from the repository
(`scripts/make_cmame_material.py`); `tests/test_cmame_material.py` checks
the three hashes.

| File | SHA-256 | Source |
|---|---|---|
| `report_rec8_v2.json` | `c6ea5934ff7fee4db63007ca1924b2973be63390d730f311e9bc29840633df51` | the deciding run of 16 July 2026 (configuration `configs/phase1_rec8_v2.json`, gate G1' GO), `runs/report_rec8_v2.json` in the bundle `fejepa_deciding_run_20260716.tar.gz` (SHA-256 `97cb17f0...`); its SHA-256 is the one PROVENANCE_NOTE.md, Run 1, records |
| `report_diag.json` | `dbd7e13dc28dea4b727716e7acd8531200bb1a5e84b4e614681dba248343aa9d` | the D1 diagnostic of 30 July 2026 (exploratory, no pre-registration), `runs/diag_b16/report_diag.json` in the bundle `diag_b16_result.tar.gz` (SHA-256 `6b859c7e...`, PROVENANCE_NOTE.md, Run 2); used only for its six further seeds of the label-free cell |
| `report_wp2_e2.json` | `1a868e9ca6b72d58d682a6c286a804ad4147b9942b19d110d621a1b0fe3c2286` | the WP2/E2 run of 31 July 2026 (configuration `configs/wp2_e2_v2.json`, pre-registration `PREREG_WP2.md`), `runs/wp2_e2/report_wp2_e2.json` in the bundle `wp2_e2_result.tar.gz` (SHA-256 `c563cb66...`, PROVENANCE_NOTE.md, Run 3); used only for its criterion K3, which the manuscript lists with the other pre-registered criteria |

All three runs decode without the load scale that later code multiplies back
(WP7 3D-P0.5): the deciding run used the July code (v2.1.4), the other two
v2.1.5, whose training path is identical. Their absolute two-dimensional
accuracies are those of that code; comparisons inside a run share the code and
are unaffected.
