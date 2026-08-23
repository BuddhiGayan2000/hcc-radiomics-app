import logging

from fastapi import APIRouter, Request

import config
from app.models.inference import predict_necrosis, predict_stage
from app.schemas.requests import NecrosisRequest, StageRequest
from app.schemas.responses import NecrosisResponse, StageResponse
from app.utils.validators import require_features

logger = logging.getLogger("hcc_api")
router = APIRouter(prefix="/predict")


@router.post("/stage", response_model=StageResponse)
def predict_stage_endpoint(body: StageRequest, request: Request):
    registry = request.app.state.registry
    model = registry.stage_models[body.model]

    require_features(body.features, model.feature_names)
    stage_probs, predicted_stage, contributions = predict_stage(model, body.features)

    # Log metadata only — never the feature values themselves (privacy policy,
    # see backend/README.md Security section).
    logger.info("predict/stage model=%s predicted=%s", body.model, predicted_stage)

    return StageResponse(
        stageProbs=stage_probs,
        predicted_stage=predicted_stage,
        contributions=contributions,
    )


@router.post("/necrotic", response_model=NecrosisResponse)
def predict_necrosis_endpoint(body: NecrosisRequest, request: Request):
    registry = request.app.state.registry
    model = registry.necrosis_model

    require_features(body.features, model.feature_names)
    necrotic_prob, contributions = predict_necrosis(model, body.features)

    logger.info("predict/necrotic necrotic_prob=%.3f", necrotic_prob)

    return NecrosisResponse(necroticProb=necrotic_prob, contributions=contributions)
