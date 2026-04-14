"""
Shared, in-memory app state for the backend.

Everything mutable that the FastAPI endpoints and the inference worker
both touch lives here, behind a single lock. Critical sections are kept
tiny — grab the lock, read or write a field, release.

No persistence. Server restart = clean slate. Fine for a demo.
"""

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Optional


@dataclass
class AppState:
    # ---- selected video / spots (set by POST /select-video) ----
    video_path: Optional[Path] = None
    spots_path: Optional[Path] = None

    # When the user switches spot layouts mid-detection via /switch-spots,
    # this holds the new path. The worker checks it each iteration, reloads
    # spots, resets the smoother, and clears this back to None.
    pending_spots_path: Optional[Path] = None

    # ---- latest inference outputs (set by the worker, read by endpoints) ----
    # The JSON contract from the project doc — same shape as Day 1's file output.
    latest_status: dict = field(default_factory=dict)
    # JPEG-encoded bytes of the latest annotated frame. Endpoint serves directly.
    latest_frame_jpeg: Optional[bytes] = None

    # ---- worker control ----
    worker_thread: Optional[Thread] = None
    is_running: bool = False
    stop_requested: bool = False

    def reset_outputs(self) -> None:
        """Wipe inference outputs. Called on stop and on new video selection."""
        self.latest_status = {}
        self.latest_frame_jpeg = None


# Single module-level instance. Import this everywhere.
state = AppState()

# Single module-level lock. Every read or write of `state` fields from
# either the worker thread or an endpoint goes through this.
state_lock = Lock()