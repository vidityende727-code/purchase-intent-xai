import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import config, load_data
from src.explain.shap_analysis import (
    build_shap_explainer,
    compute_shap_values,
    explain_single_prediction,
    get_global_importance,
    load_trained_artifacts,
    plot_summary,
)
from src.features.engineering import create_features
from src.preprocessing.pipeline import split_data

logger = logging.getLogger(__name__)

N_LOCAL_EXAMPLES = 5


def main():
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model, preprocessor = load_trained_artifacts()

    df = load_data()
    df = create_features(df)
    _X_train, X_test, _y_train, y_test = split_data(df)

    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer = build_shap_explainer(model)
    shap_values = compute_shap_values(explainer, X_test_transformed)

    # Global importance CSV
    importance_df = get_global_importance(shap_values, feature_names)
    importance_df.to_csv(models_dir / "shap_importance.csv", index=False)
    logger.info("SHAP importance saved to %s", models_dir / "shap_importance.csv")

    # Summary plot PNG
    plot_summary(
        shap_values,
        X_test_transformed,
        feature_names,
        save_path=models_dir / "shap_summary.png",
    )

    # Local explanation examples
    rng = np.random.default_rng(config["random_seed"])
    sample_indices = rng.choice(
        X_test_transformed.shape[0], size=N_LOCAL_EXAMPLES, replace=False
    )

    local_rows = []
    for idx in sample_indices:
        contributions = explain_single_prediction(
            model, explainer, shap_values, X_test_transformed, feature_names, int(idx)
        )
        proba = model.predict_proba(X_test_transformed[idx : idx + 1])[0, 1]
        top5 = contributions.head(5)
        local_rows.append(
            {
                "row_index": int(idx),
                "predicted_proba": float(proba),
                "actual_label": int(y_test.iloc[idx]),
                "top_features": "; ".join(
                    f"{r.feature}: {r.shap_value:+.3f}" for r in top5.itertuples()
                ),
            }
        )

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(models_dir / "shap_local_examples.csv", index=False)
    logger.info(
        "Local SHAP examples saved to %s", models_dir / "shap_local_examples.csv"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
