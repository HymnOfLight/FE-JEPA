"""Guards for ``labelled_policy='asis'`` (v2.1.5).

Regression for the 2026-07-14 deciding-run failure: 'asis' pointed at a
directory that did not exist on the run box, `_ensure_dataset` regenerated a
30k corpus -- *unlabelled* by WP5 design -- and the labelling stage then
rejected it after the generation cost had already been paid. Two guards make
that failure mode impossible:

  1. `require_asis_corpus`: asis + missing manifest fails fast, before any
     generation (exercised by `_ensure_dataset`).
  2. `asis_missing_labels`: label coverage is verified for the full required
     set (val + the needed pool prefix), not one sampled file, so partially
     labelled corpora are rejected up front instead of dying mid-experiment.
     Manifests predating the per-record ``labelled`` flag (Phase-1 exports)
     verify via a per-archive ``U_star`` fallback.

Torch-free: everything runs on the synthetic backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fejepa.data.archive import (MANIFEST_NAME, add_labels, load_instance,
                                 load_manifest, mark_labelled)
from fejepa.experiments.protocol import (asis_missing_labels, load_split,
                                         require_asis_corpus)
from fejepa.fe.solve import SolveLedger, solve_fe_displacement
from fejepa.fe.synthetic import generate_synthetic_dataset


def _label_like_cli(files, data_dir) -> None:
    """Mirror of `fejepa label` / runner._label_one without importing the
    runner (which needs torch): solve, attach U*, mark the manifest."""
    done = set()
    for f in files:
        arch = load_instance(f)
        if arch.labelled:
            continue
        U, _ = solve_fe_displacement(arch.K, arch.F, arch.free_mask,
                                     method="direct")
        add_labels(f, U)
        done.add(Path(f).name)
    mark_labelled(data_dir, done)


@pytest.fixture()
def unlabelled_corpus(tmp_path):
    d = tmp_path / "data"
    generate_synthetic_dataset(d, 12, 0, labelled="none", ledger=SolveLedger())
    return d


# ---------------------------------------------------- fail-fast (guard 1) ----

def test_require_asis_missing_corpus_raises(tmp_path):
    dcfg = {"dir": str(tmp_path / "nowhere"), "labelled_policy": "asis"}
    with pytest.raises(ValueError) as exc:
        require_asis_corpus(dcfg)
    msg = str(exc.value)
    assert "pre-existing corpus" in msg
    assert "fejepa label" in msg          # the fix command is part of the contract


def test_require_asis_other_policies_pass(tmp_path):
    for policy in ("economy", "all", None):
        dcfg = {"dir": str(tmp_path / "nowhere")}
        if policy is not None:
            dcfg["labelled_policy"] = policy
        require_asis_corpus(dcfg)         # must not raise: generation is allowed


def test_require_asis_existing_corpus_passes(unlabelled_corpus):
    require_asis_corpus({"dir": str(unlabelled_corpus),
                         "labelled_policy": "asis"})


# ------------------------------------------- full verification (guard 2) ----

def test_unlabelled_corpus_is_rejected(unlabelled_corpus):
    split = load_split(unlabelled_corpus, 3, 1)
    required = list(split.val_files) + list(split.pool_files[:4])
    missing = asis_missing_labels(required, unlabelled_corpus)
    assert missing, "fully unlabelled corpus must be reported"
    assert all(m.endswith(".npz") for m in missing)
    # short-circuit contract: never collects more than max_report names
    assert len(asis_missing_labels(required, unlabelled_corpus,
                                   max_report=2)) == 2


def test_prelabelled_prefix_passes_and_beyond_prefix_fails(unlabelled_corpus):
    split = load_split(unlabelled_corpus, 3, 1)
    need = 4
    _label_like_cli(list(split.val_files) + list(split.pool_files[:need]),
                    unlabelled_corpus)

    required = list(split.val_files) + list(split.pool_files[:need])
    assert asis_missing_labels(required, unlabelled_corpus) == []

    # The exact hole the retired single-sample check left open: a partially
    # labelled corpus. Files beyond the labelled prefix must still be flagged.
    beyond = list(split.pool_files[need:need + 2])
    assert asis_missing_labels(beyond, unlabelled_corpus) == \
        [Path(f).name for f in beyond]


def test_manifest_without_flags_falls_back_to_archives(tmp_path):
    """Phase-1-style corpora: U_star baked in, manifest records lack the
    ``labelled`` key. Verification must trust the archives."""
    d = tmp_path / "v1_style"
    generate_synthetic_dataset(d, 6, 0, labelled="all", ledger=SolveLedger())
    m = load_manifest(d)
    for r in m["instances"]:
        r.pop("labelled", None)
    (d / MANIFEST_NAME).write_text(json.dumps(m, indent=1))

    split = load_split(d, 2, 1)
    required = list(split.val_files) + list(split.pool_files[:2])
    assert asis_missing_labels(required, d) == []
