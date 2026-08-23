"""
FastAPI application entry point. Run via `python run.py` from the backend/
directory (see backend/SETUP.md) — do not run this module directly, since it
relies on `config` being importable from the backend/ root.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.middleware.cors import add_cors
from app.models.loader import ModelRegistry
from app.routes import health, predict
from app.utils.errors import InvalidFeaturePayload, invalid_feature_payload_handler, unhandled_exception_handler
from app.utils.logger import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    registry.load_all()
    app.state.registry = registry
    yield


app = FastAPI(
    title="HCC Radiomics Inference API",
    description="Local inference API serving the trained BCLC-staging and necrosis models.",
    version="0.1.0",
    lifespan=lifespan,
)

add_cors(app)

app.add_exception_handler(InvalidFeaturePayload, invalid_feature_payload_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(health.router)
app.include_router(predict.router)
