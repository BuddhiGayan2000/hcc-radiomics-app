# models_storage/

The trained model pipelines from the original research study, copied here unmodified. These are the actual artifacts that were validated (accuracy, ROC-AUC, calibration, decision-curve analysis) in the study — do not regenerate or re-export them casually.

| File | Model | Predicts | Features | Classes |
|---|---|---|---|---|
| `best_model_XGBoost.joblib` | XGBoost | BCLC stage | 12 (SHAP-selected) | 4 |
| `model_LightGBM.joblib` | LightGBM | BCLC stage | 12 (SHAP-selected) | 4 |
| `model_RandomForest.joblib` | Random Forest | BCLC stage | 12 (SHAP-selected) | 4 |
| `model_GradientBoosting.joblib` | Gradient Boosting | BCLC stage | 12 (SHAP-selected) | 4 |
| `best_necrotic_vs_others_model_RandomForest.joblib` | Random Forest | Necrosis | 25 (all) | 2 |

`metadata.json` in this folder is a copy of the original `run_summary.json` — the best model's cross-validated performance and its 12 selected features.

Each file is a complete `imblearn.pipeline.Pipeline` (scaler + SMOTE + classifier) — see [../docs/MODEL_LOADING.md](../docs/MODEL_LOADING.md) for what that means for how they're loaded and used. There is no separate scaler file because none is needed.

**Do not commit new/retrained models here without updating [../docs/MODEL_LOADING.md](../docs/MODEL_LOADING.md)'s class-mapping section** — a new model may have been trained with a different label encoding, and the mapping in `config.py` would silently misinterpret its output.
