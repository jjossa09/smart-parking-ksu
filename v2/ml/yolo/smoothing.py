"""
Per-spot rolling-window state machine.

A spot's committed state only flips when the last K raw observations all
agree. Kills single-frame flicker from detector jitter without adding much
latency (K frames * frame_skip frames * ~1/fps seconds).
"""

from collections import deque
from typing import List

from . import config


class SpotSmoother:
    """
    Per-spot rolling window that kills single-frame detection flicker.

    Keeps the last K raw observations per spot. A committed state only flips
    when all K recent observations agree — so a one-frame false positive or
    missed detection doesn't visibly change the UI. Tradeoff: state changes
    take K frames to propagate (~1 second of video at default settings).

    Usage:
        smoother = SpotSmoother(n_spots=29)
        smoothed = smoother.update(raw_statuses)   # call every processed frame
    """

    def __init__(self, n_spots: int, window_k: int = None):
        self.n_spots = n_spots
        self.k = window_k or config.SMOOTHING_WINDOW_K
        # One observation history per spot.
        self.history: List[deque] = [deque(maxlen=self.k) for _ in range(n_spots)]
        # Committed state per spot. Start empty (0).
        self.committed: List[int] = [0] * n_spots

    def update(self, raw_statuses: List[int]) -> List[int]:
        """Push a new raw observation array, return the smoothed/committed array."""
        if len(raw_statuses) != self.n_spots:
            raise ValueError(
                f"expected {self.n_spots} statuses, got {len(raw_statuses)}"
            )

        for i, raw in enumerate(raw_statuses):
            self.history[i].append(raw)
            # Only commit a flip when the buffer is full and unanimous.
            if len(self.history[i]) == self.k:
                if all(v == 1 for v in self.history[i]):
                    self.committed[i] = 1
                elif all(v == 0 for v in self.history[i]):
                    self.committed[i] = 0
                # otherwise: leave committed state alone (mixed = uncertain)

        return list(self.committed)

    def reset(self) -> None:
        for h in self.history:
            h.clear()
        self.committed = [0] * self.n_spots


# Quick standalone test you can run:
#   python -m ml.yolo.smoothing
if __name__ == "__main__":
    sm = SpotSmoother(n_spots=3, window_k=3)
    sequence = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],   # spot 0 not yet unanimous
        [1, 1, 0],   # spot 0 now 3-in-a-row -> commits to 1
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 1],   # spot 0 not yet unanimous on 0 again
    ]
    for i, raw in enumerate(sequence):
        out = sm.update(raw)
        print(f"step {i}: raw={raw}  committed={out}")