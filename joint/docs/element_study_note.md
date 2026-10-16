# Element kind, mesh and integration rule

An exploratory check after gate G-C1.3, 16 October 2026. It is not a gate and was not pre-specified. The scripts are in `probes/elements/` and the records in `records/elements/`.

## Question

The idealised joint model is 2.05 to 2.99 times stiffer than the joint tests (G-C1a). The reference model of G-C1.3 is that model on a 10 mm mesh with a 300 mm beam segment. Could the gap come from the discretisation: elements that lock in bending, meshes too coarse, or the integration rule?

## Answer

No. The discretisation error is more than ten times smaller than the gap.
- To match the tests, the converged stiffness would have to fall by 51% to 55% (FS1) and by 64% to 67% (FS3).
- On the C1.3 mesh (10 mm, one plate layer), the current element H8I is 1.1% (FS1) and 1.2% (FS3) stiffer than the converged value. For FS1 this depends on two nodes just outside the washer patches, and it can reach 2.7% (see *Washer patches*).
- The five kinds that do not lock converge to the same stiffness: about 37.5 kNm/mrad for FS1 and 65.0 for FS3, each to within about 1%. The model stays 2.1 to 2.2 times (FS1) and 2.8 to 3.0 times (FS3) stiffer than the tests.

## Set-up

**Model.** `joint_generic.py` solves the C1.3 reference model on any element kind of `elements.py`:
- a rigid column flange with frictionless unilateral contact;
- bolt heads bearing unilaterally, with tilt springs;
- a 300 mm beam segment whose end section is rigid and carries the statically equivalent load;
- the specimens' measured moduli and the paper's rotation.

It follows `fejoint.joint.analyse` step by step. With H8I the two agree to the last digit on FS1 and FS3, each on two meshes (`check_generic.py`).

**Specimens.** FS1 (10.4 mm plate) and FS3 (20.02 mm plate, the thickest). FS2 (15.01 mm) and FS4 (10.06 mm) were not run.

**Element kinds**

| Kind | Element | Nodes | Quadrature |
|---|---|---|---|
| H8I | hexahedron with incompatible modes (the current one) | 8 | 2 × 2 × 2 |
| H8 | trilinear hexahedron | 8 | 2 × 2 × 2 |
| H20 | serendipity hexahedron | 20 | 3 × 3 × 3 |
| H20R | serendipity hexahedron, reduced integration | 20 | 2 × 2 × 2 |
| H27 | Lagrange hexahedron | 27 | 3 × 3 × 3 |
| T4 | linear tetrahedron | 4 | 1 point |
| T10 | quadratic tetrahedron | 10 | 4 points |

- The tetrahedral meshes split each hexahedron of the structured mesh into six tetrahedra about a main diagonal. The result is conforming (checked).
- One-point hexahedra with hourglass control were not run, because their results depend on the hourglass parameters.

**Meshes.** In-plane spacing 20, 10, 5 and 2.5 mm, with 1, 1, 2 and 4 layers through the plate.
- The quadratic kinds ran from 20 mm to 5 mm. The linear kinds ran from 10 mm to 2.5 mm, and also with 8 plate layers at 2.5 mm; they cannot run at 20 mm, where a washer patch holds fewer than 3 nodes.
- At 5 mm the linear kinds also ran with 1, 4 and 8 plate layers, and H20 with 1 and 4.
- There are 70 runs in all, up to 184 130 unknowns, and they took 40 minutes.

**What changes with the element besides the interpolation.** The quadratic kinds bring their mid nodes into the contact face, the washer patches and the rigid end section.
- At 5 mm a washer patch holds 25 to 28 nodes with H8, H8I and T4; 79 to 84 with H20 and H20R; and 105 to 113 with H27 and T10.
- Node-to-node contact then behaves differently (`review_checks.py`). On the strip of the column face behind the compression flange, FS1 at 10 mm, all 20 corner nodes are in contact with H8I and with H27, none with H20, 1 with H20R and 10 with T10. Every mid node there is in contact. This is consistent with the consistent nodal loads of a uniform pressure, which are negative at the corners of a serendipity face, zero at the corners of a six-node triangle and positive on a nine-node face.
- The effect on S is small: H20 and H27 differ by at most 0.54% at 10 mm and finer. But the contact-zone outputs of H20, H20R and T10 are not comparable with those of the other kinds.
- Kinds with the same node sets: H8, H8I and T4; H20 and H20R; H27 and T10.

## Checks of the elements

`check_elements.py`, with its record in `checks.json`:
- Rigid-body modes, and the constant-strain patch test on a distorted mesh: residuals and errors below 2e-14 for all seven kinds.
- A cantilever (length 200 mm, depth 20 mm, width 10 mm, tip shear). Tip deflection over Timoshenko beam theory, with the numbers of elements along the length, depth and width:

| Kind | 10 × 1 × 1 | 20 × 2 × 1 | 40 × 4 × 2 | 80 × 8 × 4 |
|---|---|---|---|---|
| H8I | 0.987 | 0.988 | 0.993 | 0.994 |
| H8 | 0.658 | 0.870 | 0.959 | 0.986 |
| T4 | 0.211 | 0.514 | 0.795 | 0.933 |
| H20 | 0.983 | 0.992 | 0.994 | 0.995 |
| H20R | 0.986 | 0.993 | 0.995 | 0.995 |
| H27 | 0.987 | 0.993 | 0.995 | 0.995 |
| T10 | 0.978 | 0.991 | 0.994 | 0.995 |

The limit is 0.995 rather than 1 because of the clamp. It holds every component on the root face, so it also restrains the Poisson contraction of the root section, which beam theory leaves free. With ν = 0, where there is no contraction to restrain, the same clamp gives 1.000002 and 1.000009 with H20 on the two finest meshes. A clamp that holds u_x on the root face, u_y only on its line y = H/2 and u_z only on its line z = B/2 gives 1.0006 and 1.0007 (`review_checks.py`).

## Results

Initial stiffness S in kNm/mrad. Every run is tabulated in `records/elements/tables.md` and plotted in `records/elements/convergence.png`.

**Finest mesh of each kind**, against the converged value: the median of the finest runs of H8I, H20, H20R and H27, which is 37.47 for FS1 and 65.01 for FS3. The tests are FS1a 18.19, FS1b 16.84, FS3a 23.23 and FS3b 21.56.

| Kind | Mesh | FS1 | FS3 |
|---|---|---|---|
| H8I | 2.5 mm, 8 plate layers | 37.25 (−0.58%) | 64.90 (−0.17%) |
| H8 | 2.5 mm, 8 plate layers | 37.73 (+0.71%) | 65.20 (+0.29%) |
| T4 | 2.5 mm, 8 plate layers | 39.63 (+5.78%) | 66.56 (+2.39%) |
| H20 | 5 mm, 4 plate layers | 37.46 (−0.03%) | 64.98 (−0.04%) |
| H20R | 5 mm, 2 plate layers | 37.48 (+0.03%) | 65.03 (+0.04%) |
| H27 | 5 mm, 2 plate layers | 37.54 (+0.20%) | 65.11 (+0.16%) |
| T10 | 5 mm, 2 plate layers | 37.91 (+1.18%) | 65.43 (+0.65%) |

**Locking.** H8 and T4 are stiff on coarse meshes and fall monotonically with refinement.
- At 10 mm with one plate layer, T4 is 60% (FS1) and 28% (FS3) above the converged value, and H8 17% and 7%.
- On the finest meshes, T4 is still 5.8% and 2.4% above it, and H8 0.7% and 0.3%. That T4 reaches the same limit is an extrapolation.
- Locking errs in the same direction as the gap, so the direction cannot rule it out. The magnitude does.

**Integration.** H20 and H20R differ by about 3%, 1.2% and 0.5% at 20, 10 and 5 mm for FS1, and by 1.1%, 0.8% and 0.3% for FS3. Reduced integration can err soft: H20R at 20 mm is 3.3% below the converged value for FS1.

**Plate layers.** H8I at 5 mm is not monotone in the number of plate layers: FS1 gives 37.03, 37.35, 37.06 and 36.91 for 1, 2, 4 and 8 layers. All runs of the non-locking kinds at 5 mm and finer lie within 36.91 to 38.18 (FS1) and 64.90 to 65.89 (FS3), that is −1.5% to +1.9% and −0.2% to +1.4% of the converged values. Their finest runs lie within −0.6% to +1.2% and −0.2% to +0.7%, so the converged values are given to about 1%.

**Ratio to the tests.** With the converged values: 2.06 to 2.22 (FS1) and 2.80 to 3.02 (FS3). Over the finest runs of the five non-locking kinds: 2.05 to 2.25 and 2.79 to 3.03.

**Washer patches** (`washer_radius.py`, with its record in `washer_radius.jsonl`). A patch is the set of plate nodes within washer_r of the bolt's axis, so S can jump when a node crosses the circle. H8I:

| washer_r (mm) | FS1, 10 mm | FS1, 5 mm | FS3, 10 mm | FS3, 5 mm |
|---|---|---|---|---|
| 12.5 | 35.82 | 36.39 | 64.61 | 64.87 |
| 13.0 | 37.34 | 36.39 | 65.49 | 64.98 |
| 14.1 (the model) | 37.86 | 37.35 | 65.80 | 65.58 |
| 14.12 | 38.49 | 37.65 | 65.80 | 65.61 |
| 15.0 | 38.49 | 38.63 | 65.80 | 66.06 |

- On FS1's 10 mm mesh, two of the three patches each have a node just outside washer_r = 14.1 mm, at 14.102 and 14.110 mm. A radius of 14.12 mm is physically the same but takes them in, and S rises by 1.65%. At 5 mm the same change gives +0.8%; FS3 does not change at 10 mm.
- On three members of the C1.3 family, taking one node out of each patch of 10 changes S by 0.16% to 0.31% (`review_checks.py`).

**A sliver in FS1.** FS1's measured beam flange (150.50 mm) is wider than its plate (149.84 mm). So the beam flanges carry a strip of cells 0.33 mm wide along their outer edges, over the whole beam segment. `fejoint/family.py` snaps such overhangs to zero, so the C1.3 family has none.

## Review

An independent review compared the element matrices with scikit-fem 12.0.2 (largest difference 1.1e-15) and recomputed the numbers of this note. Its scripts are not in this folder; `review_checks.py` reproduces the three of its checks that this note cites.

## Commands

From `joint/`:
```bash
python -I probes/elements/check_elements.py records/elements/checks.json
python -I probes/elements/check_generic.py
python -I probes/elements/run_study.py records/elements/study.jsonl
python -I probes/elements/washer_radius.py records/elements/washer_radius.jsonl
python -I probes/elements/review_checks.py records/elements/review_checks.json
python -I probes/elements/write_report.py records/elements/study.jsonl records/elements/checks.json records/elements
```
- `run_study.py` skips the runs already in its record. Its 70 runs used PARDISO; the runs of `washer_radius.py` and `review_checks.py` used `FEJOINT_SOLVER=legacy`.
- The two solvers agree to round-off: `check_generic.py` gives the same S to about 12 digits with either.
