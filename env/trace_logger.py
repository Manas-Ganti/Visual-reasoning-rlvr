"""Append episode traces to a JSONL file for offline browser replay.

Training and evaluation produce one :meth:`InvestigationEnv.get_trace` dict
per episode; this writer appends them to a log. GRPO emits ``G`` rollouts per
prompt across many steps, so training logging is *sampled* (``sample_every``) to
keep the file manageable. The visualizer reads the log back with :meth:`load`.
"""

from __future__ import annotations

import json
import os


class TraceLogger:
    def __init__(self, path: str, sample_every: int = 1):
        self.path = path
        self.sample_every = max(1, int(sample_every))
        self._seen = 0
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

    def log(self, episode: dict, force: bool = False) -> bool:
        """Append one episode trace. Returns True if it was written.

        Every call counts toward the sampling cadence; set ``force=True`` (e.g.
        for evaluation episodes) to always write regardless of ``sample_every``.
        """
        self._seen += 1
        if not force and self._seen % self.sample_every != 0:
            return False
        with open(self.path, "a") as f:
            f.write(json.dumps(episode) + "\n")
        return True

    @staticmethod
    def load(path: str) -> list[dict]:
        """Read all episode traces from a JSONL log (skips malformed lines)."""
        episodes: list[dict] = []
        if not os.path.exists(path):
            return episodes
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return episodes
