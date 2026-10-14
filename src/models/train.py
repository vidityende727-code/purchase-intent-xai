import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.catboost
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_loader import config, load_data
from src.features.engineering import create_features
from src.preprocessing.pipeline import apply_smote, build_preprocessor, split_data

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("purchase-intent-xai")


def evaluate_model(y_true, y_pred, y_pred_proba, model_name: str) -> dict:
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
    }
    cm = confusion_matrix(y_true, y_pred)

    logger.info(
        "%s -> Accuracy: %.3f| Precision: %.3f| Recall: %.3f| F1: %.3f| ROC-AUC: %.3f| PR-AUC: %.3f",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )
    logger.info("Confusion Matrix:\n%s", cm)

    return metrics


def train_logistic_regression():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_resampled, y_resampled = apply_smote(X_train_transformed, y_train)

    model = LogisticRegression(random_state=config["random_seed"], max_iter=1000)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")

    return model, metrics


def prepare_data():
    df = load_data()
    df = create_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_resampled, y_resampled = apply_smote(X_train_transformed, y_train)

    return X_resampled, y_resampled, X_test_transformed, y_test, preprocessor


def get_models() -> dict:
    seed = config["random_seed"]
    return {
        "Logistic Regression": LogisticRegression(random_state=seed, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(random_state=seed),
        "XGBoost": XGBClassifier(random_state=seed, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(random_state=seed, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=seed, verbose=0),
    }


def train_all_models():
    X_resampled, y_resampled, X_test_transformed, y_test, preprocessor = prepare_data()

    models = get_models()
    all_metrics = []

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_resampled, y_resampled)

            y_pred = model.predict(X_test_transformed)
            y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_pred_proba, name)
            all_metrics.append(metrics)

            mlflow.log_params(model.get_params())
            logged_metrics = {k: v for k, v in metrics.items() if k != "model"}
            mlflow.log_metrics(logged_metrics)

            if name == "CatBoost":
                signature = infer_signature(X_resampled, model.predict(X_resampled))
                mlflow.catboost.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name="purchase-intent-catboost",
                )

    results_df = pd.DataFrame(all_metrics)

    save_model(models["CatBoost"], preprocessor)
    save_comparison(results_df)

    return results_df, models


def find_best_threshold(model, X_test_transformed, y_test):
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    results = []

    for t in thresholds:
        y_pred_at_t = (y_pred_proba >= t).astype(int)

        results.append(
            {
                "threshold": t,
                "precision": precision_score(y_test, y_pred_at_t),
                "recall": recall_score(y_test, y_pred_at_t),
                "f1": f1_score(y_test, y_pred_at_t),
            }
        )
    return pd.DataFrame(results)


def run_cross_validation(model, X, y, scoring="average_precision"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["random_seed"])
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    logger.info(
        "%d-Fold CV (%s): mean = %.4f, std = %.4f, scores = %s",
        cv.get_n_splits(),
        scoring,
        scores.mean(),
        scores.std(),
        [round(s, 4) for s in scores],
    )

    return scores


def save_model(model, preprocessor):
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "catboost_model.joblib"
    preprocessor_path = models_dir / "preprocessor.joblib"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info("Model saved to %s", model_path)
    logger.info("Preprocessor saved to %s", preprocessor_path)


def save_comparison(results_df):
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = models_dir / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    logger.info("Model comparison saved to %s", csv_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results_df, models = train_all_models()
    print(results_df.round(3))
