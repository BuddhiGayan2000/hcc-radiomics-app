#!/usr/bin/env python3
"""
Standalone sanity check: load every model in backend/models_storage/ and print
its structure (pipeline steps, feature count/order, classes). Useful when:
  - setting up the project for the first time (confirms your Python env can
    actually deserialize all five models)
  - after adding a new/retrained model file
  - debugging an "InvalidModelError" or similar startup failure

Run from the repo root:  python scripts/verify_models.py
"""
import sys
import warnings
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "backend" / "models_storage"

MODEL_FILES = [
    "best_model_XGBoost.joblib",
    "model_LightGBM.joblib",
    "model_RandomForest.joblib",
    "model_GradientBoosting.joblib",
    "best_necrotic_vs_others_model_RandomForest.joblib",
]


def main():
    warnings.filterwarnings("ignore")  # InconsistentVersionWarning is expected, see docs/MODEL_LOADING.md
    ok = True

    for filename in MODEL_FILES:
        path = MODELS_DIR / filename
        print(f"\n=== {filename} ===")
        if not path.exists():
            print(f"  MISSING: {path}")
            ok = False
            continue
        try:
            pipeline = joblib.load(path)
        except Exception as exc:
            print(f"  FAILED TO LOAD: {exc}")
            ok = False
            continue

        steps = [name for name, _ in pipeline.steps] if hasattr(pipeline, "steps") else "n/a (not a Pipeline)"
        print(f"  type: {type(pipeline).__name__}")
        print(f"  pipeline steps: {steps}")
        print(f"  n_features_in_: {getattr(pipeline, 'n_features_in_', 'n/a')}")
        print(f"  feature_names_in_: {list(getattr(pipeline, 'feature_names_in_', []))}")
        print(f"  classes_: {list(getattr(pipeline, 'classes_', []))}")

    print("\n" + ("All models loaded successfully." if ok else "One or more models FAILED — see above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
