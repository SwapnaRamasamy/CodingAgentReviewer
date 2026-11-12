from __future__ import annotations

import logging
import threading
import time


from reviewer.diagnostics import safe_error

logger = logging.getLogger(__name__)


class Activity:
    """Thread-safe metadata log. Never records file contents, tokens, or model output."""
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def add(self, agent, tool, status, seconds=0.0, summary=""):
        if status == "failed":
            logger.warning("Review step failed: agent=%s tool=%s reason=%s", agent, tool, summary[:240])
        with self.lock:
            self.events.append({
                "agent": agent, "tool": tool, "status": status,
                "duration_s": round(seconds, 2), "summary": summary[:240],
            })

    def run(self, agent, tool, function, *args, **kwargs):
        started = time.monotonic()
        self.add(agent, tool, "running")
        try:
            value = function(*args, **kwargs)
        except Exception as error:
            self.add(agent, tool, "failed", time.monotonic() - started, safe_error(error))
            raise
        self.add(agent, tool, "completed", time.monotonic() - started)
        return value

    def snapshot(self):
        with self.lock:
            return list(self.events)
