
"""
ThreatShield AI | 2026
__main__.py
"""

import uvicorn

from app.config import settings


def main() -> None:
    """
    Run the ThreatShield AI API server
    """
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        access_log=True,
        log_config=None,
    )


if __name__ == "__main__":
    main()
