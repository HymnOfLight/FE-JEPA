# Branch notes: `wp7-3d` (WP7 Phase-2 3D statics development)

Created 6 August 2026 from the v2.1.5 baseline (byte-identical to the attested
package `762bcf42…`). Governing documents: RUN_PLAN_2026-08-05 (§3, the 3D-P0 →
S1 → B1 → G1 sequence), PROJECT_STATUS_2026-08-06 (§4, §6 Menu B), Manual v3
§14.2–§14.3, §16.3–§16.4. British English; dates DD Month YYYY.

## What this branch deliberately lifts

`fe/tet3d.py`'s v2.1.5 docstring recorded, under the intentionally-absent
register (Manual §16.3): *"3D features/encoder (2D features hard-code 2
components by design)"* — gated on Gate G1′. **Gate G1′ = GO (16 July 2026)**
unblocked WP7; this branch implements exactly the gated items the 5 August
audit named, and nothing beyond them:

1. **P0.1** `models/fejepa.mesh_adjacency` + a shared `element_edges` helper —
   generic over (E, 3) triangles and (E, 4) tetrahedra; the MGN edge builder
   now consumes the same helper. For triangles both are **bit-identical** to
   v2.1.5 (proven in `tests/test_wp7_p0.py`), so 2D region-mask RNG
   consumption and MGN graphs are unchanged.
2. **P0.2** `geometry_descriptor` dispatched on `meta.extra.dim`; the 3D
   branch swaps the 2D mean-radius channel for the depth extent so
   `GEOMETRY_DIM` stays 6. 2D branch untouched.
3. **P0.3** `load_summary` gains the z component sum behind a defaulted
   `spatial_dim` argument (2D vector element-identical); `build_features` and
   `FeatureSpec` carry `spatial_dim` (default 2 ⇒ every v2.1.5 shape and value
   preserved; `FeatureSpec().dim == 16` asserted). A loud mismatch guard
   replaces any silent reshape.
4. **P0.4** `metrics.vm_suite` dispatches to `tet_von_mises` on 3D instances.
5. **S1 enablement** (the status document's "3D features / encoder I/O /
   metric-suite extension" line): `FieldDecoder` and the MGN head emit
   `spatial_dim` components; MGN edge features size `spatial_dim + 1`;
   baselines (E5′) and `theory._interp_to` (Prop. 1 machinery) reshape by the
   instance's dimension; E3′ threads `spatial_dim` from the data.
6. **tet3d corpus backend** `generate_tet3d_dataset` (standard manifest
   contract, unlabelled by default per WP5) + `backend: "tet3d"` in the runner
   and CLI; E4 multires for tet3d raises `NotImplementedError` (3D-G1 work).

## Red-line compliance (RUN_PLAN §5)

- `main` / v2.1.5 behaviour untouched: all work on this branch; every 2D code
  path is default-argument-preserved and covered by bit-identity golden tests.
- The original 82-test suite passes unchanged (82 pass / 5 skip in a torch-free
  environment); this branch adds `tests/test_wp7_p0.py` (22 numpy tests + 2
  torch-gated), total 104 pass / 7 skip.
- Frozen configs and run directories are not read, written, or rerun.
- Every run configuration added here is exploratory (`prereg_guard: false`).
- No pre-registered run occurs on this branch before a fresh SHA-256 stamp and
  the two F1′ residual closures (RUN_PLAN §5 item 4).

## Still absent, by design (next in the sequence)

3D-B1 solve benchmark → costed compute envelope memo; 3D-G1 gmsh 3D generator
(holes/cavities) and SimJEB ingestion; linear-attention encoder revisit;
AMP/`torch.compile` validation; the Phase-2 pre-registration (Gate G2 restated
directionally). Per RUN_PLAN §3.4, corpus scale and labelling budget for
3D-G1/3D-D1 are **outputs** of the B1 envelope, not inputs.

## 3D-P0.5 (20 August 2026): scale-equivariant decode

Motivated by the 17 August technical memo and its 18 August review (both
execution-verified): the feature pipeline divides the load battery by the
assembly-level ``fscale`` and v2.1.5 never multiplied it back, so predictions
could not track absolute load amplitude -- an input-side loss with an estimated
irreducible ~0.11 mean relative-L2 displacement floor (~0.025 on the energy
gap). The repair multiplies the decoded field by ``battery_fscale(arch.F)``
inside ``forward_instance`` -- FEJEPA and the MGN baseline alike -- behind the
model flag ``scale_decode`` (default **true** on this branch; legacy configs
without the key resolve to true). Exact by linearity; zero extra cost; the
anchor, supervised loss and evaluation all consume the physically scaled field.
``battery_fscale`` is now the single source shared by features and decode; the
2D feature bit-identity goldens pass unchanged over the refactor.

Tests: ``tests/test_wp7_p05.py`` -- 4 numpy (single-source, 2D+3D feature
invariance goldens, config round-trip) + 4 torch-gated (FEJEPA 2D/3D and MGN
equivariance u(aF) = a u(F), each with the scale_decode=false blindness
control). Suite: 108 pass / 11 skip (the original 82 unchanged).

Configs: ``wp7_s1_smoke.json`` sets ``scale_decode`` explicitly;
``wp7_s1_smoke_scaleoff.json`` is the ablation twin for the box-side A/B that
measures the removed floor. The torch-free S1 pre-flight is model-free (WP6
only) and is unaffected. Governance: this lands **before** the Phase-2
pre-registration stamp, as the review requires; the 2D line stays frozen --
v2.1.5/main behaviour is untouched and no frozen artefact is re-run. E5'
polynomial baselines remain scale-blind by design (disclosed); the ScaleAware
baseline already receives log fscale. Version marker: ``2.1.5+wp7.1``.

## Full-dependency verification and in-harness S1 execution (20 August 2026)

With torch 2.13.0, gmsh 4.15.2 and scikit-fem 12.0.2 installed, the complete
suite executes with **zero skips: 135 passed** -- every torch path (all
equivariance tests included) and the gmsh generator path ran for real. One
portability fix landed on the way: the generator test's bitwise K == K.T
assertion is environment-fragile (newer skfem assembly leaves a one-ulp
asymmetry, 1.5e-16 relative; pattern exact; the 1e-8 residual test unchanged);
the branch asserts pattern equality plus machine-tolerance values.

The S1 smoke pair then ran end to end through the standard runner on CPU
(`wp7_s1_smoke.json` then `wp7_s1_smoke_scaleoff.json`): corpus generation,
economy labelling (64 solves in-ledger on the first run; idempotent zero on the
second), E1' 12 units each, E5', WP6, gate (fail-closed NO-GO, exploratory as
designed). WP6 reproduces the 6 August pre-flight to the digit (premise 0.431;
witness 0.0122 < 0.1313). The corpus geometry is bit-reproducible across
environments; the manifest hash differs from the pre-flight's only in the eight
pool-prefix ``labelled`` flags the smoke's E1 arm purchases (verified record by
record). Honest reading of the A/B at this toy scale (24 training steps per
unit): the arms differ by optimisation warm-up, not by the asymptotic scale
floor -- scale-ON starts near the zero predictor because the normalised target
is not yet learned; the floor claim is asymptotic and its clean demonstration
remains the executed equivariance tests. The pair stands ready for a
larger-epoch box run when Phase-2 scales are set.
