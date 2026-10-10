# Pre-specification of gate G-C1a

Stamped 10 October 2026, 00:05 BST, before any specimen of Girão Coelho et al. (2004) was solved.

## Question

Can a linear elastic model with frictionless unilateral contact explain the eight measured initial rotational stiffnesses $S_{j,ini}$? The model in question uses:
- a rigid column flange;
- bolts that are not preloaded, with heads that bear on the plate only in compression and may tilt.

The eight stiffnesses are those of Girão Coelho, Bijlaard and Simões da Silva (2004), Table 8, from 16.18 to 23.39 kNm/mrad.

This is a finite element question, not a machine learning one. Its answer decides whether the elastic contact model can serve as the reference ("truth") for stage C1.3. The tests are used for checking only.

## Authority for the thresholds

- The thresholds below were proposed to the project owner on 9 October 2026 at 22:40 BST.
- The owner replied at 23:55 BST with the paper's PDF and the instruction to keep going. The thresholds were not countermanded, so they are adopted exactly as proposed.
- The owner may still change them. Any change made after this stamp will be reported as a change after results.

## Configuration

The configuration is held in `checks/gc1a_config.json`.
- Its canonical JSON (sorted keys, no whitespace) has SHA-256 `5c07343e45110058acf2fba72bcf44169c728391de386698a03af57a5da76b87`.
- The run script refuses to start unless it is given this hash.

SHA-256 of the code at stamping:

| File | SHA-256 |
|---|---|
| `checks/gc1a_run.py` | `126022c88ddbe5983031276e4ff5739b7cdef02ecbaad8368399a4726e0cb4d5` |
| `fejoint/joint.py` | `874eda60279759a018431a2e7c7e12b4f998fee3988109c6bb062a1e583f6b75` |
| `fejoint/specimens.py` | `dc7df5acb125e506c849e41af6dd89d4716256c44ee9d7c34247e126df588dfb` |
| `fejoint/contact.py` | `e0b906ee77d51061593bf7a373df63b877a3b3652c04c1e152c0bc985a697747` |
| `fejoint/hex8i.py` | `27e4964ef755c1055fd9159e860db2f7542952fac9e1237053ef3cdd1babc0f7` |
| `fejoint/linsolve.py` | `86f9ff29b99251e55a1c1141ad1638b8f48b1967d5a1aea2038c16c56a50b722` |

## Model, fixed in advance

All inputs come from the published paper unless marked otherwise. The paper was read in full on 10 October 2026; see `fejoint/specimens.py` for each source.

**Geometry**
- Measured geometry from Table 2: end plate, bolt layout, and the beam section $h_b$, $b_b$, $t_{fb}$, $t_{wb}$.
- Root radii of the beam are ignored (a hexahedral mesh).
- Column flange 40.21 mm thick. It enters only the bolt grip: the column is rigid, as stated in Sec. 2.1 and in eq. (4) of the paper.
- Half model about the beam web.

**Materials**
- Young's moduli from Table 5 (end plate by series; beam flange and web) and Table 6 (bolts).
- Poisson's ratio 0.3 (assumed; not given in the paper).

**Bolts**
- M20, tensile stress area 245 mm².
- Full thread, 22 mm holes; holes are not modelled (solid plate).
- No washers: none is mentioned in the paper, and none is visible in Figs 9 and 22.
- Head bearing disc of radius 14.1 mm (assumed: ISO 4017 head, bearing face about 28.2 mm).
- Bolt length $L_b = t_p + t_{fc} + (12.5 + 18)/2$ mm, following EN 1993-1-8 (assumed ISO 4017 head and ISO 4032 nut heights).
- Bolt model `head_contact`:
  - a rigid head plane in frictionless unilateral contact with the plate;
  - the head is free to tilt against a bending stiffness $4 E I_s / L_b$;
  - axial stiffness $E A_s / L_b$;
  - shear springs per EN 1993-1-8 Table 6.11;
  - weak regularising spring $10^{-6}$.
- This follows the practice of Bursi and Jaspart (1997, Sec. 3.3), who model the head as a solid in contact with the flange and the shank as a cylinder of area $A_s$.
- No preload. The paper describes hand tightening followed by a 45 degree turn of an ordinary spanner; preload was not measured.

**Contact with the column**
- The column flange is a rigid flat support with frictionless unilateral contact.
- There is no initial gap.

**Rotation, as the paper defines it** (eqs 2 to 5):
$$\phi = \arctan(\delta_{DT1}/900) - \delta_{EB}(900)/900, \qquad M = P\, L_{load}.$$
- $\delta_{DT1}$ is the mean vertical displacement of the beam section 900 mm from the plate's contact face.
- $\delta_{EB}$ is the Euler-Bernoulli deflection of a cantilever of length $L_{load}$, clamped at the contact face, with the model's own beam bending stiffness.
- As in the paper, the beam's shear deformation and the plate's vertical displacement are not subtracted.
- Distances are taken from the plate's contact face, which is our reading of "the face of the end plate" in the paper.
- $S_{FE} = M/\phi$. The model is linear in the load, so the load level is immaterial.

**Meshes.**
- Medium: $h = 5$ mm, 2 layers in the plate.
- Fine: $h = 2.5$ mm, 4 layers in the plate. **The fine mesh is the primary result.**

## Criteria

**C1.** Let $R = S_{FE}(\text{series}) / S_{j,ini}(\text{test})$. For each of the eight tests, $R$ must lie in $[0.80, 1.25]$, with $S_{FE}$ taken on the fine mesh.

**C2 (trend with plate thickness).** Both conditions must hold:
- $\min(S_{FE}^{FS2}, S_{FE}^{FS3}) > \max(S_{FE}^{FS1}, S_{FE}^{FS4})$;
- $S_{FE}^{FS1} / S_{FE}^{FS4} \in [0.90, 1.10]$.

The order of FS2 and FS3 is not tested: their measured means differ by 1.3%, which is less than the scatter within each pair.

**Verdict.** G-C1a passes if and only if both C1 and C2 hold.

**Mesh flag.** Any series with $|S_{fine}/S_{medium} - 1| > 0.02$ is flagged. The flag is reported but is not part of the verdict.

## Reported, not part of the verdict

**Secondary check.** The ratio of the end plate gap at DT9 to the rotation, in mm/mrad. DT9 is taken at the contact face, at mid-thickness of the tension flange, at the plate edge. Values read from Fig. 20 at about 60 kNm:

| Test | Gap ratio (mm/mrad) |
|---|---|
| FS1b | 0.155 |
| FS2a | 0.125 |
| FS3b | 0.097 |
| FS4b | 0.145 |

Reading uncertainty is about ±0.015.

**Sensitivities** (medium mesh):

| Code | Change from the model above |
|---|---|
| S1 | Bearing radius 18.5 mm (washer size) |
| S2 | Bending stiffness $1 \cdot EI/L$ |
| S3 | Agerskov effective bolt length, as quoted by Bursi and Jaspart |
| S4 | Plate bonded to the column flange |
| S5 | Rotation measured at the beam section next to the plate |
| S6 | Column rotation: $S/(1 + c)$ for $c \in \{0.05, 0.10, 0.15\}$ |
| S7 | Rigid washer patch, bonded, no tilt |

For S6, Fig. 12(c) shows a column-to-beam rotation ratio of about 0.05 to 0.12 in the elastic range for FS1a. The paper's rotation includes this ratio, while the model's rigid column excludes it.

## If the gate fails

No training is started. The candidate causes listed below were written before the run:
- column rotation (S6);
- bolt preload from the 45 degree turn;
- initial gaps from welding;
- bolt holes not modelled;
- bearing radius and bolt length;
- the reading of "face of the end plate";
- the unloading-branch definition of $S_{j,ini}$.

Whatever is found is reported.

## References

- Bursi, O.S., Jaspart, J.P. (1997). Benchmarks for finite element modelling of bolted steel connections. *Journal of Constructional Steel Research* 43(1-3), 17-42. doi:10.1016/S0143-974X(97)00031-X
- Girão Coelho, A.M., Bijlaard, F.S.K., Simões da Silva, L. (2004). Experimental assessment of the ductility of extended end plate connections. *Engineering Structures* 26(9), 1185-1206. doi:10.1016/j.engstruct.2000.09.001
