from fastapi import APIRouter, Request

from app.schemas.responses import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    registry = request.app.state.registry
    return HealthResponse(status="ok", models_loaded=registry.loaded_model_names())
