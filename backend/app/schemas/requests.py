from typing import Literal

from pydantic import BaseModel

import config


class StageRequest(BaseModel):
    model: Literal["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"] = config.DEFAULT_STAGE_MODEL
    features: dict[str, float]


class NecrosisRequest(BaseModel):
    features: dict[str, float]
