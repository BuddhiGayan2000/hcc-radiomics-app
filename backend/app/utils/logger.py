"""
Logging setup.

Privacy rule (see docs/../Section 7 of the original spec): request/response
logs must never include the actual feature values or images. Route handlers
must only log metadata (which model was requested, timing, success/failure) —
never the `features` dict itself. This module just configures format/level;
enforcing what gets logged is the responsibility of the calling code.
"""
import logging


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    return logging.getLogger("hcc_api")
