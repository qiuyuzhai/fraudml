"""Uvicorn entry point for the FraudML scoring API.

Run with::

    python -m src.serving.main
    # or
    uvicorn src.serving.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uvicorn

from src.serving.app import app  # noqa: F401  (re-exported for uvicorn target)


def main() -> None:
    uvicorn.run(
        "src.serving.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
