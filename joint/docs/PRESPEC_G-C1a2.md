# Pre-specification of gate G-C1a2

Stamped 10 October 2026, 13:49 BST, before any of the four joint details was solved with the model below.

## Question

Gate G-C1b found that a higher-fidelity elastic contact model reproduces the isolated welded T-stubs of the same laboratory within the band, on average. G-C1a2 carries that model into the joint and asks G-C1a's question again: does it reproduce the eight measured initial stiffnesses $S_{j,ini}$ of Table 8 of Girão Coelho, Bijlaard and Simões da Silva (2004)?

This is the step that G-C1b's pre-specification named for a pass. The model is also the reference that stage C1.3 would train against.

**Expected outcome, stated before the run.**
- G-C1a's model was 2.05 to 2.99 times too stiff.
- At the level of the T-stub, the higher-fidelity changes soften the model by 7% to 15% (G-C1b).
- The welds, which G-C1a left out and this model includes, stiffen it: by about 10% on the synthetic joint VER1.
- **The gate is therefore expected to fail.** Its purpose is to measure, not to rescue.

## What was seen before the stamp

- All the results of G-C1a, G-C1b and the reading G-C1c.
- The joint paper, including Fig. 3(a), which gives the weld throats.
- The model was developed and verified on the synthetic joint VER1 only (`tests/test_joint_hf.py`).
- No run of this model on FS1 to FS4 was made before the stamp. Their meshes were built once without a solve, to size the meshes.

## Authority

- On 10 October 2026 at 09:58 BST the project owner instructed work to proceed with the recommended options.
- The interim report sent after G-C1b said this gate would follow; no countermand was received.
- The criteria are G-C1a's, which the owner adopted. Any change after this stamp will be reported as a change after results.

## Configuration

The configuration is held in `checks/gc1a2_config.json`.
- Its canonical JSON has SHA-256 `585c7736b1132639a6017158b8ce6e895093c9ba50f926375e4a09aeb56e3dfd`.
- The run script refuses to start unless it is given this hash, and resumes if interrupted.

SHA-256 of the code at stamping:

| File | SHA-256 |
|---|---|
| `checks/gc1a2_run.py` | `c4057e12c87c30844d05e4bfbbea555030e9cc21846aa9c19b7e882295a33b8b` |
| `checks/gc1a2_evaluate.py` | `ee9e6c501be86e5a17defef961f8e1f78b2f70cb804297283b6a92f95f094012` |
| `fejoint/joint_hf.py` | `e719e0f0da8d55069435afa8d44c052eccaf3fd24742ba6e3d26e7b06b62548f` |
| `fejoint/joint.py` | `23b7fb48f9a587384c797b597408e85b8c8e368dad69631af90c16c564a5c044` |
| `fejoint/specimens.py` | `dc7df5acb125e506c849e41af6dd89d4716256c44ee9d7c34247e126df588dfb` |
| `fejoint/tstub_hf.py` | `ed7f8b8f966a0ae2b7908c20e414799484e93c8ae290d312d1a8ff17c52c1c4f` |
| `fejoint/contact.py` | `75885802990155b4634a8be58035658ead51510f0758ad1ce2d2778b5750db8a` |
| `fejoint/hex8i.py` | `27e4964ef755c1055fd9159e860db2f7542952fac9e1237053ef3cdd1babc0f7` |
| `fejoint/linsolve.py` | `a829f4f5d6becd7c1081db542f3ade923dc95ed7725527a2ad9f948a6eb6739a` |

A copy of these files is kept in `checks/gc1a2_stamped_code`.

## Model, fixed in advance (`fejoint/joint_hf.py`)

Inputs as G-C1a (`fejoint/specimens.py`: the paper's Tables 2, 5 and 6), with these changes:

**Bolt holes**
- 22 mm, cut through the end plate ("22 mm drilled holes", Sec. 2.1).

**Solid bolts**
- Each bolt's head and the part of its shank inside the end plate are one solid:
  - head of diameter 30 mm across flats and height 12.5 mm;
  - shank of tensile stress area 245 mm², not touching the hole wall.
- The head bears on the plate only on its bearing face, $11 \le r \le 14.1$ mm, unilaterally and without friction. The bearing face diameter is 28.19 mm (ISO 4017, grade A; assumed).
- **The rest of the bolt**, inside the rigid column flange and down to the nut, acts on the shank's section in the contact plane. That section moves as a plane and carries:
  - an axial spring, calibrated on each mesh so that the whole bolt's compliance from its bearing face to the column is $L_{eff}/(E_b A_s)$. Here $L_{eff}$ is Agerskov's effective length (Bursi and Jaspart 1997) for the grip $t_p + t_{fc}$, with the bolt threaded through the grip;
  - bending springs $E_b I_s / L_{rest}$, with $L_{rest} = t_{fc} + m/2$: the bolt below the plate as a cantilever from the nut, free to move sideways in its hole.

**Welds** (new against G-C1a)
- 45 degree fillet welds join the beam to the plate: both faces of both flanges and both faces of the web.
- Throats are 5.75 mm on the flanges and 3.75 mm on the web, the midpoints of the ranges in Fig. 3(a) (5.5 to 6 and 3.5 to 4).
- They are stair-stepped by the rule of the T-stub model; the meshed weld volume is reported.

**Shear**
- The vertical shear reaches the column through the EN 1993-1-8 bolt shear springs, as in G-C1a. They are spread over each bolt's bearing ring.
- With frictionless contact, no other path exists.

**Kept from G-C1a**
- Rigid column flange with frictionless unilateral contact and no initial gap.
- No preload; linear elasticity.
- Half model about the web.
- The load on the beam section at the load point. The paper's transverse stiffener sits there (Fig. 2, Fig. 3(b)).
- Root radii ignored.

**Boundary conditions checked against the test** (the lesson of G-C1b's audit)
- The column was bolted to a reaction wall and is taken as rigid, as in the paper (Sec. 2.1, eq. 4). Its measured flexibility is reported as a variant.
- The beam was loaded at its stiffened section, beyond which it is free, as in the model.
- No part is free to tilt where the test held it.

**Rotation**
- $S_{FE} = M/\phi$, with $\phi$ from the paper's eqs (2) to (5) and the model's own Euler-Bernoulli chord subtracted, exactly as in G-C1a.

**Meshes**

| Mesh | In-plane spacing | Cells across the bearing ring | Layers in the plate | Unknowns |
|---|---|---|---|---|
| Medium | 5 mm | 3 | 2 | about 230 000 |
| Fine | 3.33 mm | 3 | 3 | about 380 000 |

- Welds are meshed at a quarter of the in-plane spacing or a fifth of the smaller leg, whichever is smaller.
- **The fine mesh is the primary result.** A finer mesh does not fit in memory.
- On VER1, each step of refinement changed $S$ by about 0.9%.

## Criteria (those of G-C1a)

**C1.** For each of the eight tests, $R = S_{FE}(\text{series})/S_{j,ini}(\text{test}) \in [0.80, 1.25]$, on the fine mesh.

**C2.** Both conditions must hold:
- $\min(S_{FE}^{FS2}, S_{FE}^{FS3}) > \max(S_{FE}^{FS1}, S_{FE}^{FS4})$;
- $S_{FE}^{FS1}/S_{FE}^{FS4} \in [0.90, 1.10]$.

**Verdict.** G-C1a2 passes if and only if C1 and C2 hold.

**Mesh flag.** Any series with $|S_{fine}/S_{medium} - 1| > 0.02$ is flagged. The flag is reported but is not part of the verdict.

## Reported, not part of the verdict

**Comparison with G-C1a.** $S_{FE}$ against G-C1a's fine-mesh values.

**Other definitions of the rotation**

| Definition | Test side | Model side |
|---|---|---|
| No beam subtraction on either side | $1/(1/S_{j,ini} + 0.032)$ | gross stiffness |
| The model's chord subtracted on both sides | chord subtracted | chord subtracted |

The value 0.032 mrad/kNm is the paper's subtraction, read from Fig. 11(b) by G-C1a's audit.

**Column.** The measured column flexibility, 0.0036 mrad/kNm (Fig. 12(b), FS1a, read by G-C1a's audit), is put in series with the model.

**Gap at DT9 per rotation** against Fig. 20, as in G-C1a.

**Sensitivities** (medium mesh):

| Code | Change |
|---|---|
| S1 | No welds (the geometry of G-C1a) |
| S2 | Bending springs $4 E_b I_s / L_{rest}$ |
| S3 | EN 1993-1-8 bolt length, $t_p + t_{fc} + (k + m)/2$ |

## Interpretation, fixed in advance

**FAIL with $R$ above the band** (expected).
- The model that reproduces the isolated T-stubs is still too stiff for the joints, by a factor to be compared with G-C1a's.
- That confirms the gap at the level of the joint.
- The model becomes the reference for stage C1.3 under option A, with the gap stated.
- The raw records remain the way to locate the gap.

**PASS.** Contrary to expectation, the higher-fidelity changes close the gap. The model becomes the reference without qualification.

**FAIL on C2 only.** The level is right but the order with plate thickness is wrong. This is reported as such.

## References

- Bursi, O.S., Jaspart, J.P. (1997). Benchmarks for finite element modelling of bolted steel connections. *Journal of Constructional Steel Research* 43(1-3), 17-42. doi:10.1016/S0143-974X(97)00031-X
- Girão Coelho, A.M., Bijlaard, F.S.K., Simões da Silva, L. (2004). Experimental assessment of the ductility of extended end plate connections. *Engineering Structures* 26(9), 1185-1206. doi:10.1016/j.engstruct.2000.09.001
- Girão Coelho, A.M., Bijlaard, F.S.K., Gresnigt, N., Simões da Silva, L. (2004). Experimental assessment of the behaviour of bolted T-stub connections made up of welded plates. *Journal of Constructional Steel Research* 60(2), 269-311. doi:10.1016/j.jcsr.2003.08.008
