#!/usr/bin/env python3
"""wp9-pool Stage 0f, for the CMAME paper's cost table (RUNBOOK_W9 Sec. 4):
the thread pools of the BLAS and OpenMP libraries loaded in a process that has
imported numpy and scipy's sparse solvers (SuperLU links a BLAS of its own),
read at run time from the libraries themselves, as threadpoolctl reads them
(it is not installed on the box). Reported only; nothing in PREREG_W9 reads
it.

Prints one JSON object: the environment's thread variables, the CPU affinity
and `os.cpu_count()` (which can report the host's CPUs rather than the
container's), numpy's, scipy's and torch's versions, torch's intra-op threads
(torch is imported last, after the BLAS pools are read, so that its own
OpenMP runtime does not change them), and, per loaded library that exports a
known thread query, its path (numpy's own BLAS sits in `numpy.libs`, the one
SciPy's SuperLU links in `scipy.libs` for pip wheels), the query and the
thread count.

    python scripts/blas_threads.py
"""
from __future__ import annotations

import ctypes
import json
import os
import re

ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS")
QUERIES = (("openblas", re.compile(r"openblas"),
            ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
             "openblas_get_num_threads64_", "openblas_get_num_threads")),
           ("mkl", re.compile(r"mkl_rt|mkl_core"), ("MKL_Get_Max_Threads",)),
           ("blis", re.compile(r"blis"), ("bli_thread_get_num_threads",)),
           ("openmp", re.compile(r"lib(g|i)?omp"), ("omp_get_max_threads",)))
"""(kind, library file-name pattern, thread queries tried in order)."""


def loaded_libraries() -> list:
    """The shared libraries mapped into this process, in address order."""
    out = []
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            path = parts[-1] if len(parts) >= 6 else ""
            if path.startswith("/") and ".so" in os.path.basename(path) and path not in out:
                out.append(path)
    return out


def pools(paths=None) -> list:
    out = []
    for path in loaded_libraries() if paths is None else paths:
        name = os.path.basename(path).lower()
        for kind, pattern, queries in QUERIES:
            if not pattern.search(name):
                continue
            try:
                lib = ctypes.CDLL(path)              # the loaded copy: same file, same handle
            except OSError:
                continue
            for q in queries:
                fn = getattr(lib, q, None)
                if fn is None:
                    continue
                fn.restype, fn.argtypes = ctypes.c_int, []
                out.append({"kind": kind, "library": os.path.basename(path), "path": path,
                            "query": q, "num_threads": int(fn())})
                break
    return out


def main() -> None:
    import numpy
    import scipy
    import scipy.sparse.linalg  # noqa: F401  (loads SuperLU and the BLAS it links)

    res = {"env": {k: os.environ.get(k) for k in ENV},
           "affinity": len(os.sched_getaffinity(0)), "os_cpu_count": os.cpu_count(),
           "numpy": numpy.__version__, "scipy": scipy.__version__, "pools": pools()}
    try:
        import torch

        res["torch"], res["torch_threads"] = torch.__version__, torch.get_num_threads()
    except Exception as exc:                          # the pools are what matters here
        res["torch"], res["torch_threads"] = f"not imported: {exc!r}"[:200], None
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
