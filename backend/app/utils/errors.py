"""Custom exceptions and the handlers that turn them into HTTP responses."""
import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("hcc_api")


class InvalidFeaturePayload(Exception):
    """Raised when a request is missing required feature keys."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Missing required feature keys: {missing}")


async def invalid_feature_payload_handler(request: Request, exc: InvalidFeaturePayload):
    return JSONResponse(
        status_code=422,
        content={"detail": f"Missing required feature keys: {exc.missing}"},
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak stack traces or internal details to the client. Full details
    # go to the server log only (and the log itself must never contain the
    # request's feature values — see logger.py).
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check the backend log for details."},
    )
