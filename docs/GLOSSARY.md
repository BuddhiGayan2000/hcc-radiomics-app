# Glossary

Plain-language definitions for anyone on the team without a radiology/medical-ML background.

**HCC (Hepatocellular Carcinoma)** — the most common type of primary liver cancer. This project predicts its stage and tissue characteristics from CT scans.

**BCLC stage (Barcelona Clinic Liver Cancer staging)** — a standard clinical system for classifying how advanced a liver tumor is. This project uses 4 categories: Healthy, Stage A (early), Stage B (intermediate), Advanced.

**Necrosis / necrotic tissue** — dead tissue within a tumor, often a sign of how the tumor is responding to treatment or how aggressively it's growing. This project classifies ROI tissue as necrotic vs. non-necrotic ("Others").

**Post-contrast / pre-contrast (CT phases)** — CT scans taken before and after injecting a contrast agent (dye) that makes blood vessels and certain tissues show up more clearly. Subtracting the two highlights tissue that absorbed the contrast — often tumor tissue.

**ROI (Region of Interest)** — the area a user draws on the image (in this app, a freehand outline around the tumor) that all feature calculations are based on.

**Radiomic features** — quantitative measurements extracted from medical images (shape, intensity statistics, texture patterns) that can be fed into a machine learning model, as an alternative to a human visually reading the scan.

**GLCM (Gray-Level Co-occurrence Matrix)** — a texture-analysis technique that looks at how often pairs of pixel intensities occur next to each other, used to quantify how "rough" or "smooth" a tissue region looks.

**SHAP (SHapley Additive exPlanations)** — a technique for explaining an individual model prediction by attributing it to each input feature (e.g. "this prediction leaned toward Stage B mainly because of a high `GLCM_Correlation` value"). This is what powers the "evidence strip" in the UI.

**Pipeline (scikit-learn/imblearn sense)** — a bundled sequence of preprocessing + model steps saved as one object, so you don't have to remember to apply the same scaling separately every time you predict. See [backend/docs/MODEL_LOADING.md](../backend/docs/MODEL_LOADING.md).

**SMOTE (Synthetic Minority Oversampling Technique)** — a technique used *during training only* to balance a dataset when one class (e.g. "Healthy") has far fewer examples than others. It does nothing at prediction time.

**StandardScaler** — rescales each feature to have zero mean and unit variance, matching the scale the model was trained on. Must always use the exact scaler fitted during training, never a freshly-fit one.

**CV (Cross-Validation) / F1-macro / ROC-AUC** — standard machine-learning evaluation metrics referenced in `run_summary.json` and the original study's performance figures. Not specific to this codebase — see any ML reference for definitions.
