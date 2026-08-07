# PROVENANCE NOTE -- FE-JEPA 2D phase (frozen record)

Compiled 1 August 2026; terminal-log status finalised 6 August 2026;
arXiv ID recorded 7 August 2026.
This note attests the code state, data lineage, and
artefact fingerprints of the three executed runs of the 2D phase. All SHA-256
digests below were computed independently from the received archives.

## Code lineage
- v2.1.4: deployed for the 2026-07-16 deciding run (entry point resolved to
  /root/autodl-tmp/FE-JEPA; confirmed post hoc from warning/traceback paths
  and the absence of the v2.1.5 labelling-verification line).
- v2.1.5: asis fail-fast guard + full label verification + version-string fix;
  training path byte-identical to v2.1.4. Deployed 2026-07-30 (reinstall
  confirmed live by the "[labelling] asis: verified" console line). Package:
  FE-JEPA-v2_1_5.zip, SHA-256
  762bcf42096425e264b73e5c47472dc7387d5f9005b5d74aca3313fe374435b6.
- git describe was unavailable in-run for all three runs (executions outside
  a git checkout). Criteria integrity was enforced by the executable
  config-hash guards throughout; code state is attested post hoc by tree
  identity against the packages above. Remediation: this repository, tags
  v2.1.5 / deciding-run-2026-07-16 / prereg-wp2-e2; all future runs execute
  inside a git clone.

## Data lineage
- Corpus runs/data2d: n = 30,000, gmsh backend, regenerated on the run box
  (Option A); NOT the original Phase-1 files (gmsh bit-identity across
  versions not guaranteed; manifest SHA-256 pinned in the run reports).
- Labels: 1,280 instances (val 256 + pool prefix 1,024) x 4 loads = 5,120
  reference solves, purchased OFFLINE via `fejepa label` on 15 July 2026.
  In-run ledgers therefore undercount: true reference-solve total for the
  deciding run = 5,120 offline + 2,048 in-run (multires val) = 7,168.

## Run 1 -- Deciding run (2026-07-16T12:35:12Z, Gate G1' = GO)
- Config phase1_rec8_v2.json, SHA-256
  62b26ad868d424ef5527c8cb7d826c818aa1ba5cebbc76c7bfe665062781f0ce
  (stamped PREREG.md verified in-run; stamped copy SHA-256
  b9470e686e4459509744dd203882394ffcb3fbf4b6850863452d538318c60e97).
- Verdict: GO (a/b/c all pass at decision budget 64). Kills K5, K6 triggered
  (component retirements); K1/K2/K4/C1/C5 not triggered.
- Artefacts (verified): report_rec8_v2.json c6ea5934ff7fee4db63007ca1924b297
  3be63390d730f311e9bc29840633df51; RESULTS.md c5c5e58da10d86249ccc44ab194f3e
  aaf0feef9c5fb1ca5768c832928018b432; figure1_energy_gap.png c96d42d055f86ff8
  bccfa3af2ca0f9ae6c74a3a8c8810f80f2c0d74fe9f60230; e8_states ar_p1024_s0/1/2
  d49405f21a86ffc360241dc477a9d46f4c2e1af12b068181916b116d0f77b464 /
  cb70bcf142ec7c6c13e316e0cc5e9a5e22376446aa1af0e933da126c5fba38ea /
  eed8f3c4bc65fa825846de4e869f7d8b90abd12eaa5c82f1febffd8ba4978dca.
- Bundle as received: fejepa_deciding_run_20260716.tar.gz, SHA-256
  97cb17f00a524d7a1b75c67dfde8f4f3984ce0741568c37c5ea354e8ca6c5453.

## Run 2 -- D1 diagnostic (2026-07-30T14:30:24Z, exploratory, prereg off)
- Config diag_anchor_b16.json, SHA-256 1a85eeabf2646568e8ac17f3119abe980f568
  52cf44b05b684b72fa3d1619b78. Executed on v2.1.5 (console line confirmed).
- Findings: gradient-balanced anchor at budget 16 intrinsically unstable --
  divergence in BOTH harnesses (E1' 3/6 seeds, E8 2/6; pooled with the
  deciding run 7/18). Fixed-lambda arm converged 6/6 (disp 0.2127 +/- 0.0086,
  +17.1% vs no-anchor, t = 7.06). AR (0 labels) replicated on 6 fresh seeds:
  disp 0.1658 +/- 0.0023, egap 0.0821 +/- 0.0019 (9 seeds total).
- Bundle: diag_b16_result.tar.gz, SHA-256
  6b859c7e8bc09a5319e10e571fcd729b17e60ab466de7490041b12c5aeea4bba.

## Run 3 -- WP2/E2 one-shot verdict (2026-07-31T22:25:26Z, pre-registered)
- Config wp2_e2_v2.json, canonical SHA-256
  60088135b1fecb2768d882ad6fee37add0fc3463160decee98a34b511159cb2c
  (stamped PREREG_WP2.md line 41; verified in-run; independently recomputed;
  config file byte-identical to the delivered original, file SHA-256
  fef75971527728881f8fbcf20b755536d38245db18f31cff30aabc166bb372c4).
- Verdict: K3 not triggered under the frozen absolute-band implementation ->
  publication branch A. Composition disclosed: 3 of 4 out-of-band cells have
  JEPA worse; the single JEPA-better cell (b64 egap +92.5%) sits on an
  unstable ar_ft baseline (per-seed [0.81, 6.99, 1.66] vs 0.2012 +/- 0.0272
  for the same arm in the deciding run). Raw AR (0 labels) dominates all
  fine-tuned variants. Criterion-drafting lesson: future kills to be phrased
  directionally.
- Artefacts: report_wp2_e2.json 1a868e9ca6b72d58d682a6c286a804ad4147b9942b19
  d110d621a1b0fe3c2286; RESULTS.md fa6bfcd50555b032a42888c0f7da20bdb8789cad2
  c10d8ee237d37377027d742; stamped PREREG_WP2.md 550cf7c66c360cf79546c9d359d
  fc95d8a6593f3e64ad11d8f77522235535306. Bundle: wp2_e2_result.tar.gz,
  SHA-256 c563cb66258df3847902076a735b582de8a529ecdf3bff812adb91def9fe7bf4.
- Terminal log: partial (tail) recovery. wp2_e2_console.txt is archived in
  this repository, SHA-256
  93702e975ec9660c18b745b37003054e85737fc406048f07c29e88190637ea31,
  byte-identical to the copy received off-box. The preserved scrollback
  opens mid AR-s0 pretraining and runs to completion: 33/33 E2 units plus
  the gate and solve-ledger lines, all consistent with report_wp2_e2.json.
  The labelling-verification line predates the preserved window; v2.1.5
  execution for this run therefore rests on the 2026-07-30 reinstall
  verification (Run 2 console evidence) and the unchanged environment.
  Recorded as final, 6 August 2026.

## Publication record
- Theory note: the identities these runs exercise are proved in R. Cao and
  X. Song, "Discrete energy as an exact label-free training objective for
  finite-element surrogates", arXiv:2608.05437v1 [cs.CE], submitted
  5 August 2026, announced 7 August 2026.
  https://arxiv.org/abs/2608.05437

## Standing prohibitions
- configs/phase1_rec8_v2.json and configs/wp2_e2_v2.json are one-shot and
  must never be re-executed.
- The stamped PREREG.md / PREREG_WP2.md and their configs are frozen.
- The retracted preprint arXiv:2604.01349 must not appear in any
  external-facing material.