"""Torch runtime policy for this project's hardware profile (plan Sec.9).

TF32 (tensor-float32 matmul/conv on Ampere+; the RTX 5090 included) is enabled by
default: it accelerates float32 matmuls ~1.5-2x with a mantissa reduction that is
well inside batch-1 SGD noise. It does NOT touch float64 -- the FE solves, the numpy
metric identities, and the machine-precision anchor tests are unaffected.

AMP (bf16) and ``torch.compile`` remain intentionally absent per the plan (Sec.2.5
tags them unvalidated; WP7 validates them in Phase 2). TF32 is the validated-safe
subset shipped now. Disable with ``"tf32": false`` in the config.
"""

from __future__ import annotations


def setup_torch(device: str, tf32: bool = True) -> dict:
    """Apply the precision/thread policy; returns a dict echoed into provenance."""
    policy = {"tf32": bool(tf32), "device": device}
    try:
        import torch

        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        policy["torch"] = torch.__version__
        if device.startswith("cuda") and torch.cuda.is_available():
            policy["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:                    # numpy-only environments
        policy["note"] = f"torch unavailable ({type(e).__name__})"
    return policy
