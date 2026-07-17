"""Live WebSocket route factory placeholder.

The live route remains registered in backend.main because it still depends on
record conversion helpers, meeting registry, runtime config, and the processing
executor. The runtime/session logic has been prepared for service extraction.
"""

from __future__ import annotations


def create_live_router(*_args, **_kwargs):
    raise NotImplementedError("live routes are currently registered by backend.main")
