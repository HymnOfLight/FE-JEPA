"""Lightweight run-progress reporting (stage banners + unit counters with ETA).

Supports plan Sec.9 operability: multi-day runs (E1' has 84 trainings at rec8 scale,
E8 ~30) must show where they are. Print-based and dependency-free; every experiment
wraps its training units in a :class:`Task`, trainers echo ~10 in-training milestones
(``log_every``: 0 = auto 10%, -1 = silent, N = every N steps).
"""

from __future__ import annotations

import time


def _hms(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def stage(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


class Task:
    """Counts completed units of equal-ish cost and prints progress + ETA."""

    def __init__(self, label: str, total: int):
        self.label, self.total = label, max(1, int(total))
        self.k = 0
        self.t0 = time.perf_counter()
        print(f"[{self.label}] starting: {self.total} units", flush=True)

    def step(self, desc: str = "") -> None:
        self.k += 1
        el = time.perf_counter() - self.t0
        eta = el / self.k * (self.total - self.k)
        tail = f" | {desc}" if desc else ""
        print(f"[{self.label}] {self.k}/{self.total} "
              f"({100.0 * self.k / self.total:.0f}%) "
              f"elapsed {_hms(el)} eta {_hms(eta)}{tail}", flush=True)

    def done(self) -> float:
        el = time.perf_counter() - self.t0
        print(f"[{self.label}] done in {_hms(el)}", flush=True)
        return el
