# Extended end plate joints (stage C)

This folder holds the work towards a label-free surrogate for the extended end plate joints tested by Girão Coelho, Bijlaard and Simões da Silva (2004). It contains the finite element models, the gates that tested them against the experiments, and the pre-registered training gate G-C1.3.

## Contents

**`fejoint/`: the finite element code**
- `hex8i.py`: eight-node hexahedra with incompatible modes.
- `contact.py`: frictionless unilateral contact, solved by the primal-dual active set method.
- `linsolve.py`: the sparse solver.
- `joint.py`: the idealised joint model of G-C1a.
- `specimens.py`: the tested details.
- `joint_hf.py`, `tstub_hf.py`, `tstub.py`, `delft_tstubs.py`: the higher-fidelity joint and T-stub models of G-C1a2 and G-C1b, and the T-stub tests.
- `family.py`, `export.py`: the joint family of stage C1.3, and its condensed nodal problem.

**`c13/`: the run code of gate G-C1.3.** `RUNBOOK_C13.md` gives its commands in order.

**`tests/`: the tests.** From this folder, with the repository one level up:
```bash
FEJEPA_REPO=.. python -m pytest -q -p no:cacheprovider tests
```

**`docs/`: pre-specifications, verdicts and notes**
- `contact_energy_note.md`: the energy objective under frictionless contact.
- G-C1a, the idealised joint model against the joint tests: `PRESPEC_G-C1a.md`, `G-C1a_verdict.md`.
- G-C1a2, the higher-fidelity joint model: `PRESPEC_G-C1a2.md`, `G-C1a2_verdict.md`.
- G-C1b, the T-stubs: `PRESPEC_G-C1b.md`, `G-C1b_verdict.md`, and `G-C1b_addendum_welds.md`, which is not pre-specified.
- G-C1c, the low-moment secants of the published joint curves: `PRESPEC_G-C1c.md`, `G-C1c_reading.md`.
- `source_GiraoCoelho2006.md`: the authors' own finite element model of the T-stubs.
- `C1.3_preparation_note.md`, `C1.3_data_design.md`: exploratory checks before stage C1.3.
- G-C1.3, the training gate: `PRESPEC_G-C1.3.md`, with its stamped code in `G-C1.3_stamped_code/` and the hashes in `G-C1.3_code.sha256`.
- The files ending in `_text.md` and `_numbers.md` are the parts each verdict is assembled from.

**`probes/`: exploratory probes.** None of them is a gate.

## Status on 10 October 2026

| Gate | Result |
|---|---|
| G-C1a | FAIL: the idealised model is 2.05 to 2.99 times stiffer than the joint tests |
| G-C1a2 | FAIL: the higher-fidelity model is 2.11 to 3.04 times stiffer |
| G-C1b | PASS, narrowly: the T-stub model is 1.10 times the T-stub tests in the geometric mean |
| G-C1c | Reading of the published curves at low moments |
| G-C1.3 | Pre-specified; to run on the GPU host |

## Not in this folder

- The run scripts and audit records of gates G-C1a to G-C1c, which the verdicts cite as `checks/...`. They are kept with the project owner's records.
- The papers.
- The logs of the C1.3 cost probes.
- The data cache of the two-dimensional probe, `probes/c13_net/data_cache.npz`; `net_probe.py` rebuilds it on its first run.

**Notes on the probes**
- `probes/c13_scale.py`, `gc1b_weld_bracket.py` and `gc1b_weld_bracket_tables.py` read files in `checks/`, so they do not run from this folder alone.
- `probes/c13_segment_effect.json` has no script. A one-off command solved FS1 to FS4 with `fejoint.joint.analyse` under G-C1a's options, on the 10 mm mesh with one layer, with the full beam and with segments of 300 mm and 400 mm. FS4 with the 300 mm segment was recomputed on 10 October 2026 and matched to every digit.
- In `probes/c13_cost/pilot_energy.json` and `pilot_supervised.json`, the local path of the repository checkout was replaced by its branch and commit. Nothing else in them was changed.

## Reference

Girão Coelho, A.M., Bijlaard, F.S.K., Simões da Silva, L. (2004). Experimental assessment of the ductility of extended end plate connections. *Engineering Structures* 26(9), 1185-1206. doi:10.1016/j.engstruct.2000.09.001
