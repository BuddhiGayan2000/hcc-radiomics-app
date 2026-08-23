"""
Loads every trained pipeline once at startup and keeps it (plus a SHAP
explainer built against its classifier step) in memory for the lifetime of
the process. This is what makes per-request latency acceptable — model
deserialization and explainer construction are the slow parts, not inference.
"""
import logging

import joblib
import shap

import config

logger = logging.getLogger("hcc_api")


class LoadedModel:
    """A trained pipeline plus everything needed to explain its predictions."""

    def __init__(self, name: str, pipeline):
        self.name = name
        self.pipeline = pipeline
        self.scaler = pipeline.named_steps["scaler"]
        self.classifier = pipeline.named_steps["clf"]
        # feature_names_in_ is the exact subset (and order) this model was
        # trained on — never assume it matches another model's subset.
        self.feature_names = list(pipeline.feature_names_in_)

        # shap.TreeExplainer does not support multiclass GradientBoostingClassifier
        # (a known shap limitation, not a bug in this code — see
        # backend/docs/MODEL_LOADING.md). Degrade gracefully: predictions for
        # this model still work, only its SHAP contributions are unavailable.
        try:
            self.explainer = shap.TreeExplainer(self.classifier)
        except Exception as exc:
            logger.warning(
                "SHAP explainer unavailable for model %s (%s); "
                "predictions will still work but contributions will be empty.",
                name, exc,
            )
            self.explainer = None


class ModelRegistry:
    """Holds every loaded model. One instance lives on app.state for the process lifetime."""

    def __init__(self):
        self.stage_models: dict[str, LoadedModel] = {}
        self.necrosis_model: LoadedModel | None = None

    def load_all(self):
        for name, filename in config.STAGE_MODEL_FILES.items():
            path = config.MODELS_DIR / filename
            logger.info("Loading staging model %s from %s", name, filename)
            pipeline = joblib.load(path)
            self.stage_models[name] = LoadedModel(name, pipeline)

        logger.info("Loading necrosis model from %s", config.NECROSIS_MODEL_FILE)
        necrosis_pipeline = joblib.load(config.MODELS_DIR / config.NECROSIS_MODEL_FILE)
        self.necrosis_model = LoadedModel("necrosis", necrosis_pipeline)

        logger.info(
            "All models loaded: stage=%s, necrosis=ok",
            list(self.stage_models.keys()),
        )

    def loaded_model_names(self) -> list[str]:
        names = list(self.stage_models.keys())
        if self.necrosis_model is not None:
            names.append("Necrosis-RandomForest")
        return names
