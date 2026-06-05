#!/usr/bin/env python3
"""Start the CricGiri analytics API server."""

from __future__ import annotations

import uvicorn

from api.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
