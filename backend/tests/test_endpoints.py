def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "Necrosis-RandomForest" in body["models_loaded"]
    for name in ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"]:
        assert name in body["models_loaded"]


def test_predict_stage_xgboost(client, sample_features):
    resp = client.post("/predict/stage", json={"model": "XGBoost", "features": sample_features})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["stageProbs"].keys()) == {"Healthy", "A", "B", "Advanced"}
    assert abs(sum(body["stageProbs"].values()) - 1.0) < 1e-6
    assert body["predicted_stage"] in body["stageProbs"]
    assert len(body["contributions"]) > 0


def test_predict_stage_gradient_boosting_has_no_contributions(client, sample_features):
    """Known limitation: shap.TreeExplainer doesn't support multiclass
    GradientBoostingClassifier. See backend/docs/MODEL_LOADING.md."""
    resp = client.post("/predict/stage", json={"model": "GradientBoosting", "features": sample_features})
    assert resp.status_code == 200
    assert resp.json()["contributions"] == []


def test_predict_stage_defaults_to_xgboost(client, sample_features):
    resp = client.post("/predict/stage", json={"features": sample_features})
    assert resp.status_code == 200


def test_predict_necrotic(client, sample_features):
    resp = client.post("/predict/necrotic", json={"features": sample_features})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["necroticProb"] <= 1.0
    assert len(body["contributions"]) > 0


def test_predict_stage_missing_features_returns_422(client):
    resp = client.post("/predict/stage", json={"model": "XGBoost", "features": {"Std": 1.0}})
    assert resp.status_code == 422
    assert "Missing required feature keys" in resp.json()["detail"]


def test_predict_stage_invalid_model_returns_422(client, sample_features):
    resp = client.post("/predict/stage", json={"model": "NotAModel", "features": sample_features})
    assert resp.status_code == 422  # pydantic Literal validation
