"""On-disk instance archives and PyTorch datasets for FE-JEPA."""

from fejepa.data.archive import (
    InstanceArchive,
    load_problem,
    save_problem,
    write_manifest,
)

__all__ = [
    "InstanceArchive",
    "load_problem",
    "save_problem",
    "write_manifest",
]
