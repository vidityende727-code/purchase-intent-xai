import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from src.api.schemas import PredictionResponse, SessionInput
from src.explain.shap_analysis import build_shap_explainer, compute_shap_values
from src.features.engineering import create_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Purchase Intent Prediction API")

MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent / "models" / "catboost_model.joblib"
)
PREPROCESSOR_PATH = (
    Path(__file__).resolve().parent.parent.parent / "models" / "preprocessor.joblib"
)
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


@app.post("/predict", response_model=PredictionResponse)
def predict(session: SessionInput):
    logger.info("Received prediction request for session data.")
    input_df = pd.DataFrame([session.model_dump()])
    input_df = create_features(input_df)

    X_transformed = preprocessor.transform(input_df)

    probability = model.predict_proba(X_transformed)[0, 1]

    threshold = 0.40
    prediction = int(probability >= threshold)
    if probability <= 0.20 or probability >= 0.80:
        confidence = "High"
    else:
        confidence = "Low"

    explainer = build_shap_explainer(model)
    shap_values = compute_shap_values(explainer, X_transformed)

    feature_names = preprocessor.get_feature_names_out()
    row_shap_values = shap_values[0]

    contributions = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": row_shap_values,
        }
    ).sort_values("shap_value", key=abs, ascending=False)

    top_features = contributions.head(5).to_dict(orient="records")

    logger.info(
        "Prediction complete: probability=%.4f, prediction=%d, confidence=%s",
        probability,
        prediction,
        confidence,
    )

    return PredictionResponse(
        purchase_probability=round(float(probability), 4),
        prediction=prediction,
        top_features=top_features,
        confidence=confidence,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": type(model).__name__,
        "threshold": 0.40,
    }
