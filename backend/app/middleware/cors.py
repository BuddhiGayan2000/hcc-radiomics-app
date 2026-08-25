"""CORS setup — restricted to configured frontend origins, never a wildcard."""
from fastapi.middleware.cors import CORSMiddleware

import config


def add_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
