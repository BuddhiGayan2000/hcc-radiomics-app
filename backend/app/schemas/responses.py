from pydantic import BaseModel


class Contribution(BaseModel):
    name: str
    value: float


class StageResponse(BaseModel):
    stageProbs: dict[str, float]
    predicted_stage: str
    contributions: list[Contribution]


class NecrosisResponse(BaseModel):
    necroticProb: float
    contributions: list[Contribution]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
