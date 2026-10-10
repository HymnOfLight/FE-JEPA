"""The configuration of gate G-C1.3's deciding run and its canonical hash.

Everything that changes a number of the run is here; the pre-specification stamps the canonical JSON's SHA-256.
The repository code the run imports is pinned by the SHA-256 of each file (repository_code); training and the bench
refuse to run against other files.
"""
from __future__ import annotations

import hashlib
import json

from fejoint.family import RANGES, CLEAR, SNAP, WASHER_R

# Seeds used before the stamp, kept out of the run's samples: tests (1, 3, 7, 8), the pilot (101, 102), the smoke
# test (999), the cost probe (20261010), and the two seeds first planned for this gate (20261011, 20261012), retired
# because the test suite solved members drawn with them.
SEEDS_SEEN = [1, 3, 7, 8, 101, 102, 999, 20261010, 20261011, 20261012]

# The repository files the run imports (an import trace of train and bench), at wp9-pool 92dfc93.
REPOSITORY_CODE = {
    "src/fejepa/__init__.py": "6b4a69dde91fbd83c5c16707a00e4c9e6a2e02647aca1d3132b313e4b20f9fdc",
    "src/fejepa/anchor/__init__.py": "88d852286c71e468491b7d0b34cec2c467bd4c96724a26ca8ebcc6b8f92306e4",
    "src/fejepa/anchor/energy.py": "2f8afe7243ee2cfbeb23ed505475d1f780de9ad020eedf0216f6a36cb272e802",
    "src/fejepa/models/__init__.py": "88d852286c71e468491b7d0b34cec2c467bd4c96724a26ca8ebcc6b8f92306e4",
    "src/fejepa/models/features.py": "0af43e0a87fda6bc177e9b446a720ac11e3e116bc29e445446edf846162c208d",
    "src/fejepa/models/fejepa.py": "3263988641477cfab564c5c622f8cf3406d98e45078134da8c6905d4b410b847",
    "src/fejepa/train/__init__.py": "88d852286c71e468491b7d0b34cec2c467bd4c96724a26ca8ebcc6b8f92306e4",
    "src/fejepa/train/schedule.py": "d53b55877c65efc37ee0a3877362451d8227ef84f581df64138ac1b469bd57c5",
}


def config() -> dict:
    return {
        "gate": "G-C1.3",
        "family": {"ranges": {k: list(v) for k, v in RANGES.items()}, "clear_mm": CLEAR, "snap_mm": SNAP,
                   "washer_r_mm": WASHER_R, "fixed": "beam, column flange, bolts and load arm of FS1; E 210000, nu 0.3"},
        "data": {"train_seed": 20261013, "eval_seed": 20261014, "n_train": 1024, "n_eval": 128,
                 "tested": ["FS1", "FS2", "FS3", "FS4"], "seeds_seen_before_stamp": SEEDS_SEEN,
                 "mesh": {"h": 10.0, "n_tp": 1, "n_tf": 1, "n_tw": 1, "growth": 1.15, "hx_max": 40.0},
                 "beam_segment_mm": 300.0, "P_full_N": 10000.0, "k_rot_factor": 4.0, "reg": 1e-6},
        "repository_code": dict(REPOSITORY_CODE),
        "model": {"dim": 256, "depth": 8, "heads": 8, "activation_checkpointing": True,
                  "decoder": "the repository's FieldDecoder (node latent and pooled latent, two GELU layers), "
                             "last layer zeroed at the start",
                  "features": "centred RMS-normalised coordinates (3), Dirichlet flags (3), load over its largest "
                              "component (3), node flags (6), family parameters over their range maxima (8)",
                  "output_scale": "label-free displacement scale of the export (u_scale)"},
        "map": {"kind": "softplus", "T": 1e-3,
                "on": "u_x of the column face and the separation slots of the bearing patches"},
        "training": {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "clip": 1.0,
                     "schedule": "cosine with linear warm-up over the first 5% of steps (the repository's)",
                     "batch": "one instance per step, instances in a fresh random order each epoch",
                     "energy_loss": "energy of the instance over its label-free energy scale (P/2) u_scale / 2",
                     "supervised_loss": "mean squared error of the field in units of u_scale",
                     "precision": "float32 weights and activations; float32 matmul precision 'highest' (no TF32)",
                     "non_finite": "a non-finite loss stops the run; the stop is final and is not retried"},
        "arms": {
            "label_free": {"loss": "energy", "labels": 0, "epochs": 200, "seeds": [0, 1, 2]},
            "supervised_16": {"loss": "supervised", "labels": 16, "epochs": 200, "seeds": [0, 1, 2]},
            "supervised_64": {"loss": "supervised", "labels": 64, "epochs": 200, "seeds": [0, 1, 2]},
            "supervised_256": {"loss": "supervised", "labels": 256, "epochs": 200, "seeds": [0, 1, 2]},
            "supervised_1024": {"loss": "supervised", "labels": 1024, "epochs": 200, "seeds": [0, 1, 2]},
        },
        "evaluation": {
            "sets": ["eval", "tested"],
            "contact_tolerance": "u_x <= 1e-3 u_scale on the column face, for prediction and reference alike",
            "metrics": ["S_paper over the reference's, minus 1", "contact intersection over union",
                        "energy gap (Pi - Pi*) / |Pi*|", "energy-norm error", "displacement relative L2 error",
                        "von Mises relative L2 error", "zero-field screening: Pi(u) < Pi(0)",
                        "the field rescaled along its ray, alpha = f.u / u.K.u: its stiffness error and energy gap"],
        },
        "criteria": {
            "C1": "label-free arm: the median over its three seeds of the median over the 128 held-out members of "
                  "|S_pred / S_ref - 1| is at most 0.05",
            "C2": "label-free arm: the median over its three seeds of the median over the 128 held-out members of "
                  "the column-face contact intersection over union is at least 0.9",
            "verdict": "PASS iff C1 and C2",
            "non_finite_seed": "a label-free seed whose run stopped on a non-finite loss enters the medians over "
                               "seeds with |S_pred / S_ref - 1| = infinity and intersection over union = 0",
            "incomplete": "INCOMPLETE while a label-free seed has neither predictions nor a non-finite stop",
            "reported_not_judged": "every metric for every arm and seed; the supervised arms against the "
                                   "label-free one, with no winner set in advance; the tested layouts",
        },
    }


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()
