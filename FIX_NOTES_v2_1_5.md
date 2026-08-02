# FE-JEPA v2.1.5 — Fix Notes

Date: 2026-07-14 · Base: v2.1.4 (`FE-JEPA-v2_1_4.zip`) · Scope: repair path "Option A" for the failed deciding run, plus a full code audit.

---

## 1. Context — the failure being fixed

The deciding run (`configs/phase1_rec8_v2.json`, `labelled_policy="asis"`) died on the AutoDL box with:

```
ValueError: labelled_policy='asis' but corpus has no labels
```

Root cause, verified against the shipped `outputs_JEPA_nonpz` artefacts: `runs/data2d` did not exist on the fresh box, so `_ensure_dataset` regenerated it — and the v2 generators write archives **unlabelled by design** (WP5 economy) for every policy except `"all"`. The result was a 30,000-instance corpus with `"labelled": false` on every manifest record (measured: 30,000/30,000), rejected immediately afterwards by the `asis` check. In short: hours of gmsh generation followed by a certain crash. Two code defects enabled this: `asis` could trigger generation at all, and the `asis` check sampled a single file (`val_files[0]`), so a partially labelled corpus would have passed and died mid-experiment instead.

## 2. What changed

| # | Fix | File | Behaviour |
|---|-----|------|-----------|
| F1 | **Fail-fast**: `asis` + missing manifest now raises *before* any generation, with the exact `fejepa generate` / `fejepa label` commands in the message. | `src/fejepa/experiments/protocol.py` (new `require_asis_corpus`), called from `src/fejepa/experiments/runner.py::_ensure_dataset` | The 2026-07-14 failure now costs milliseconds, not hours. |
| F2 | **Full verification**: the `asis` branch now verifies labels on the *entire* set the run consumes — all `n_val` validation files plus the pool prefix of depth `label_need` (E1′/E8 consume deterministic prefixes, `pool_files[:b]` / `pool_files[:p]`; verified at `e1_anchor.py:56`, `e8_regimes.py:93,111,134`). Manifest `labelled` flags answer in O(1); files the manifest does not vouch for fall back to an archive-level `U_star` check, so Phase-1-style corpora whose manifests predate the flag still verify. The error message echoes the exact `fejepa label` command with the run's own numbers. | `src/fejepa/experiments/protocol.py` (new `asis_missing_labels`), used in `src/fejepa/experiments/runner.py` labelling stage | Partially labelled corpora are rejected up front; the previously misleading `label_need` log line is now the verification depth, and a `[labelling] asis: verified …` line is printed on success. |
| F3 | **Version string**: `__version__` was stale at `"2.0.0"`; bumped to `"2.1.5"`. Not consumed by report provenance (which records python/numpy/scipy/torch), so cosmetic. | `src/fejepa/__init__.py` | Package version matches the release name. |
| — | **Packaging hygiene**: removed from the archive an empty junk directory `{configs,tests,src…` (shell brace-expansion artefact), `__pycache__` trees, and the redundant outer copy of `PLAN_MAP.md`. No source content was affected. | archive only | Clean drop-in tree. |

New tests: `tests/test_asis_guard.py` — 6 cases covering fail-fast on a missing corpus, pass-through for `economy`/`all`, rejection of a fully unlabelled corpus, `max_report` short-circuit, acceptance of a pre-labelled val + prefix **and** rejection beyond the prefix (the exact hole the retired single-sample check left open), and the Phase-1 fallback (manifest stripped of `labelled` keys, `U_star` baked in).

**Everything else is byte-identical to v2.1.4** — verified by `cmp` for `configs/phase1_rec8_v2.json`, `configs/smoke.json`, `PREREG.md`, `README.md`, `PLAN_MAP.md`, `pyproject.toml`, `.gitignore`; `diff -rq` confirms the only deltas are the three source files above plus the new test file.

## 3. Verification

Flags follow the project convention: **[E]** verified by execution here, **[A]** verified by analysis (this sandbox has no torch; torch-dependent paths compile and are reviewed, and run on the GPU box).

- [E] Full suite: **82 passed, 5 skipped** (baseline v2.1.4: 76 passed, 5 skipped; +6 new, no regressions). Skips are the torch/gmsh `importorskip` tests, unchanged.
- [E] `python -m pyflakes src tests`: clean (also clean on the v2.1.4 baseline).
- [E] `python -m compileall src tests`: clean.
- [E] The two new helpers, including the manifest-fallback path and the partial-labelling rejection, are exercised end-to-end on the synthetic backend.
- [A] The two runner call sites (`_ensure_dataset` guard; labelling-stage verification). `runner.py` imports torch transitively, so the integration is verified by compile + review here and by the test suite on the GPU box (`pytest tests/test_experiments_smoke.py` exercises `run_config` with `economy`; the `asis` branches are covered by the new protocol-level tests).
- [E] Prereg invariance: `config_sha256(configs/phase1_rec8_v2.json)` = `62b26ad868d424ef5527c8cb7d826c818aa1ba5cebbc76c7bfe665062781f0ce` before and after — code changes cannot enter the hash, which is computed from the parsed config dict only (`report.py::config_sha256`).

## 4. Runbook — Option A on the AutoDL box

The 30k corpus already generated at `/root/autodl-tmp/FE-JEPA/runs/data2d` is an asset; do not delete it. From the repo root:

```bash
# 1. Deploy the fix: overwrite src/fejepa and tests only (see §5 before copying).
# 2. Pre-label exactly what the deciding run consumes
#    (deterministic split, seed 1 → identical file sets; ~1,280 instances × 4 loads
#    = 5,120 direct solves; minutes with 8 workers):
fejepa label runs/data2d --n-val 256 --split-seed 1 --pool-prefix 1024 --workers 8

# 3. (Recommended) project the run cost before committing GPU hours:
fejepa bench --config configs/phase1_rec8_v2.json --device cuda

# 4. Re-run the deciding config, unchanged:
fejepa run-config configs/phase1_rec8_v2.json
```

Step 2 is idempotent (`_label_one` skips already-labelled archives), and with the fixed code a forgotten step 2 now fails in seconds with this exact command printed, instead of after generation.

**Bookkeeping note (honesty):** under `asis` the report's data-economy block records `asis-preexisting-corpus n=0`; the 5,120 pre-labelling solves happen outside the run's ledger. This is consistent with the config's own `_comment` ("economy not demonstrated on this corpus — WP5 regenerates an unlabelled pool later"), but the run log should record that the corpus is a **fresh regeneration plus offline labelling on the run box**, not the original Phase-1 files (gmsh meshing is not guaranteed bit-identical across versions; the manifest SHA-256 in the report's provenance block makes the actual corpus traceable either way).

## 5. Pre-registration and git state

- The **config is untouched**, so the executable freeze holds: the stamped `PREREG.md` on the run box (recording `62b26ad8…`) continues to pass `verify_prereg`. **Do not overwrite the box's `PREREG.md`** — the copy in this package is the unstamped template (`CONFIG_SHA256 = <fill before tagging>`), preserved byte-for-byte from v2.1.4.
- The **code** has changed since the prereg tag. No criterion, threshold, split, seed, or config field changed, and the failed run produced no results (it died before any training), so this is an infrastructure amendment made pre-results. Recommended record: commit, tag `v2.1.5`, and note in the run log that the deciding run executes on `v2.1.5` code against the unchanged `prereg-v2.0` config hash.

## 6. Audit — other findings (checked, clean)

Beyond the fixed defects, the following were examined and found sound:

- **Prefix-consumption contract** [A]: E1′/E8 stream `pool_files[:budget]` / `pool_files[:pool_size]`; E2/E3′/E5′/E6/E7 use the in-memory prefix `pool_files[:pool_hi]` with `pool_hi = 1024` for the phase-1 config — all within the labelled prefix `label_need = 1024`, so Option A's coverage is exact.
- **E4 seed contract** [A]: the runner's multires val-labelling permutation (`np.random.default_rng(e4.seed).permutation`, `perm[:n_val]`) matches `e4_meshviews._pairs_split` exactly.
- **Gate G1′** [E via suite, A review]: fails closed when E5′/E8 are absent; `_cell` tolerates both `int` and `str` budget keys (renderer/producer boundary).
- **Crash-safe reporting** [A]: `report_*.json` is written *before* RESULTS.md/figure rendering; rendering failures warn and point to `fejepa results` rather than killing a finished run.
- **Multiprocessing** [A]: all pools use the `spawn` context (labelling, generation, experiment workers) — CUDA-safe; ledger accounting aggregates in the parent.
- **`torch.load`** [A, E on box]: single call site, `map_location="cpu", weights_only=True` — correct under torch ≥ 2.6 defaults; exercised on the box's torch 2.11 by the 2026-07-14 smoke run.
- **Static sweep** [E]: no bare `except:`, no mutable default arguments, no hard-coded `.cuda()`, pyflakes clean across `src` and `tests`.
- **Smoke evidence** [E, from shipped outputs]: the 2026-07-14 smoke run (`99b147bc…`, git `b365af5`, RTX PRO 6000 Blackwell, torch 2.11.0+cu128) completed E1′–E8 + WP6 + gate + report + figure end-to-end; its NO-GO and kill triggers at budgets 4/8 on synthetic data carry no scientific weight. Note **E2 is implemented and executed** in that run — the earlier "E2 unimplemented / gate branch dead code" status is superseded.

No further defects were found in this pass.

---

# FE-JEPA v2.1.5 — 修复说明（中文摘要）

**背景。** 正式判定运行（`phase1_rec8_v2.json`，`labelled_policy="asis"`）在云端新机器上崩溃：`runs/data2d` 不存在，`_ensure_dataset` 按 WP5 设计重新生成了 30,000 个**无标签**实例（manifest 逐条核实 30,000/30,000 均为 `labelled: false`），随后 `asis` 检查必然拒绝——先烧数小时生成、再必然崩溃。旧检查只抽查一个 val 文件，部分标注的语料还会先通过、再在实验中期崩溃。

**修复（三处源码改动 + 一个新测试文件，其余文件与 v2.1.4 逐字节一致）。**
F1 快速失败：`asis` + manifest 缺失时在任何生成之前抛错，错误信息含完整修复命令（`protocol.py::require_asis_corpus`，由 `runner.py::_ensure_dataset` 调用）。
F2 全量校验：`asis` 分支改为校验运行实际消耗的全部标注集合——全部 val + 深度为 `label_need` 的 pool 前缀（E1′/E8 消耗确定性前缀，已核实行号）；manifest 标志 O(1) 应答，未标记文件回退到逐档案 `U_star` 检查，兼容无标志的 Phase-1 式 manifest；报错信息回显带本次运行参数的 `fejepa label` 命令。
F3 版本号：`__version__` 由过期的 `"2.0.0"` 更正为 `"2.1.5"`（provenance 不消费此字段，属一致性修正）。
打包卫生：移除空的花括号残留目录、`__pycache__`、外层重复的 `PLAN_MAP.md`。

**验证。** 全套测试 82 通过 / 5 跳过（基线 76/5，新增 6，无回归）[执行验证]；pyflakes 与 compileall 全绿 [执行验证]；runner 两处调用点因沙盒无 torch 为分析验证（编译 + 审查，GPU 机上由 smoke 测试覆盖）；`config_sha256` 前后均为 `62b26ad8…`——代码改动不进入配置哈希，prereg 冻结不受影响 [执行验证]。

**方案 A 操作步骤（AutoDL 机，30k 语料勿删）：**
```bash
fejepa label runs/data2d --n-val 256 --split-seed 1 --pool-prefix 1024 --workers 8
fejepa bench --config configs/phase1_rec8_v2.json --device cuda   # 建议先投影时长
fejepa run-config configs/phase1_rec8_v2.json
```
标注步骤幂等；若遗漏，修复后的代码会在数秒内报错并打印上述确切命令。

**Prereg 与 git。** 配置未动，机器上已 stamp 的 `PREREG.md`（记录 `62b26ad8…`）继续通过校验；**切勿用本包内的 PREREG.md 覆盖机器上的副本**（包内是未 stamp 的模板，与 v2.1.4 逐字节一致）。代码在 tag 之后有改动：属结果产生前的基础设施修订（无任何判据/阈值/种子/配置字段变化，失败运行未产生任何结果），建议提交并打 `v2.1.5` tag，并在运行日志中记录"以 v2.1.5 代码执行未变的 prereg-v2.0 配置"。诚实性注记：`asis` 下报告的 economy 表记 `asis-preexisting-corpus n=0`，5,120 次预标注求解发生在账本之外——与配置 `_comment` 自述一致，但运行日志应写明语料为"云端重新生成 + 离线标注"，非原始 Phase-1 文件。

**审计其余结论。** 前缀消耗契约、E4 种子契约、Gate 的 fail-closed 与 str/int 键容错、"先写报告后渲染"的崩溃保护、全 spawn 多进程、`torch.load(weights_only=True)`、静态危险模式扫描——均核查通过；另确认 E2 已实现并在 2026-07-14 的 smoke 中实际执行（更新此前"E2 未实现"的状态记录）。本轮未发现其他缺陷。
