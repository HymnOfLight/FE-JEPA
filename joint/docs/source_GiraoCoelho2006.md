# Source note: the original authors' finite element model of the T-stubs (2006)

Girão Coelho, A.M., Simões da Silva, L., Bijlaard, F.S.K. (2006). Finite-element modeling of the nonlinear behavior of bolted T-stub connections. *Journal of Structural Engineering* 132(6), 918-928. doi:10.1061/(ASCE)0733-9445(2006)132:6(918).

The bibliographic record was checked on the publisher's page on 10 October 2026. Everything below was read from the PDF supplied by the project owner; page numbers are the journal's.

## What the paper models

**Software and elements** (p. 919)
- LUSAS, eight-node bricks (HX8M) with 2 × 2 × 2 Gauss points.
- Node-to-node joint elements (JNT4) for contact, with Coulomb friction where friction is assumed.
- Updated Lagrangian kinematics; von Mises plasticity with isotropic hardening, from measured true stress-strain curves (Fig. 3).
- Newton-Raphson iteration under prescribed displacement.

**Model of the welded T-stubs** (pp. 919-920)
- One eighth of the assembly, by symmetry. The plane between the two flanges is modelled by contact on a rigid foundation ($E = 10^{15}$ MPa).
- The web plate is **not fused** to the flange plate: "a continuous 45° fillet weld (throat thickness $a_w$) links the flange and the web, though the two plates are not necessarily in contact". Contact elements model the interface between the web and the flange plates.
- **Bolt**: an "equivalent bolt" of half the Eurocode 3 length $L_b$, with a threaded part of area $A_s$ and an unthreaded part of the nominal diameter, in proportion to the real bolt. The head or nut and the washer, where there is one, are one body with the bolt ("slightly stiffer deformation behavior").
- **Friction**: none between the flanges; $\mu = 0.25$ between the flange and the washer or bolt head, and between the web and flange plates.
- **Load**: a uniform prescribed displacement of 0.1 mm at the top of the web.
- **Geometry** (Table 1, nominal): WT1 has $h = 200$, $t_f = t_w = 10$, $w = 90$, $n = 30$, $p/2 = 25$, $e = 20$, $a_w = 5$ and $d_0 = 14$ mm, with M12 short-threaded bolts and no washers.
- **Mesh** (Fig. 4): 3,588 elements and 5,680 nodes for the one-eighth model.

## Results used here

**Table 2 gives the stiffness per bolt row** ("The stiffness and strength values that appear in the table are computed per bolt row", p. 926). The bold rows are averaged test results; underlined values include the deformation of the web.

| Specimen | $k_{e.0}$ of their model, kN/mm per row | $k_{e.0}$ measured, kN/mm per row | Model over test |
|---|---|---|---|
| WT1 | 69.29 | 71.09 | 0.975 |
| WT4A | 88.12 | 86.96 | 1.013 |
| WT7_M12 | 84.26 | 91.18 | 0.924 |
| T1 (rolled, Bursi and Jaspart 1997) | 83.54 | 49.00 (includes the web) | 1.70 as tabulated |

**The tests behind the bold values**, checked against Table 9 of the T-stub paper (Girão Coelho et al. 2004, *J. Constr. Steel Res.* 60, 269-311), which gives the stiffness of the whole specimen:
- WT1: 71.09 is half the mean of WT1g (137.16) and WT1h (147.17), the two tests the paper selects (p. 922).
- WT4A: 86.96 is half of WT4Ab (173.91).
- WT7_M12: one bolt row, so 91.18 is the specimen value.
- This confirms that Table 9's values are for the whole specimen, both rows, which is the basis on which gate G-C1b compared its model.

**Their reading of the welded case** (abstract, p. 922): the model "yields stiffer results than the experiments, although the agreement is good". For welded T-stubs the differences are larger than for rolled ones, which the authors attribute to residual stresses and the changed properties near the weld toe. Both affect mainly the plastic range.

## Comparison with the model of gate G-C1b

Our values are G-C1b's stamped fine-mesh stiffness of the whole specimen, halved for the two-row specimens (`checks/gc1b_verdict.json`). G-C1b left the loaded end of the web free to tilt; the 2006 model prescribes a uniform displacement there, which holds it. For the single-row WT7_M12, G-C1b's audit found that holding the end raises our stiffness by 12.9% on the medium mesh (`checks/audit_gc1b/probes_posthoc.json`, 82.38 to 93.00 kN/mm). The row "end held" applies that ratio to the fine-mesh value, an estimate. For the two-row specimens the end condition makes no difference (WT1: 151.67 free, 151.76 held, medium mesh).

| Specimen | Ours, kN/mm per row | Ours over theirs | Ours over the tests above |
|---|---|---|---|
| WT1 | 74.28 | 1.072 | 1.045 |
| WT4A | 95.59 | 1.085 | 1.099 |
| WT7_M12, end free (as stamped) | 80.97 | 0.961 | 0.888 |
| WT7_M12, end held (estimate) | about 91.4 | about 1.085 | about 1.00 |

- With the end held the same way, our model is **7% to 9% stiffer** than the authors' model on all three specimens. Theirs is within −8% to +1% of the tests they selected.
- With the welds drawn to contain the full fillet (`docs/G-C1b_addendum_welds.md`), ours rises by a further 2.0% to 2.7% on these three.
- **Differences between the two models**, none of them tested here:
  - The fused web. Ours fuses the web to the flange plate, theirs links them only by the welds and contact. In tension the joint face can open in theirs, so ours should be the stiffer one.
  - The bolt length. Ours uses Agerskov's length, theirs Eurocode 3's. G-C1b's sensitivity S2 found this worth 0.1% to 0.3%.
  - The hole. Ours is 13 mm for M12, theirs 14 mm.
  - The head. Ours bears without friction on its bearing face; theirs is one body with the bolt and bears with friction 0.25.
  - The mesh. Theirs is much coarser, with fully integrated bricks, which tends to make a bending model stiffer.
  - The kinematics. Theirs is geometrically nonlinear; at initial stiffness that should not matter.
- The paper does not model the end plate joints. It says nothing directly about the factor of 2 to 3 found at the joint level in gates G-C1a and G-C1a2.

## What this changes

1. **The T-stub level is now supported by an independent model.** The original authors' own finite element model, with a different code, mesh, bolt idealisation and weld connection, gives initial stiffnesses 7% to 9% below ours and within 8% of their selected tests. G-C1b's statistical pass was narrow; this agreement is a second, independent line.
2. **The gap remains at the joint level.**
   - Two finite element models agree with each other and with the T-stub tests.
   - Eurocode 3, as the authors applied it, is on average 16% below the T-stub tests: geometric mean 0.84, 0.56 to 1.21 per test (Table 8 of the T-stub paper, G-C1b's records). For the joints it is 1.97 to 2.31 times the tests (Table 10 of the joint paper); our model is 2.1 to 3.0 times.
   - So the same methods that match the T-stubs, or err soft on them, are about twice too stiff for the joints. What differs between the joint tests and the T-stub tests, in the specimens, the setup or the definition of the stiffness, is the open question. Without the raw records it stays open.
3. **A modelling choice to note for the joint model.** Like the T-stub model, the joint model fuses the beam to the end plate. The authors would link them by the welds and contact only. Its effect on the joint has not been computed; by the T-stub comparison it is likely of the order of 10%, far from a factor of 2.
4. **The rolled T-stub T1.** The authors' model is 1.70 times the measured stiffness as tabulated, but the measured value includes the web's deformation and the computed one does not. Our idealised code found 1.15 to 1.5 on the Bursi and Jaspart T-stubs (G-C1a). Neither comparison is like for like.
