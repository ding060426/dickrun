"""Upload route factory placeholder.

The upload implementation still lives in backend.main while its state and job
lifecycle are owned by services.upload_service. This module marks the API layer
boundary for the next mechanical route move without changing public routes.
"""

from __future__ import annotations


def create_upload_router(*_args, **_kwargs):
    raise NotImplementedError("upload routes are currently registered by backend.main")
