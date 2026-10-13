import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from src.models.train import prepare_data
from src.explain.shap_analysis import load_trained_artifacts


def build_lime_explainer(X_train, feature_names):
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()

    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=["No Purchase", "Purchase"],
        mode="classification",
    )
    return explainer


def explain_instance_lime(explainer, model, X_transformed, index, num_features=5):
    row = X_transformed[index]
    if hasattr(row, "toarray"):
        row = row.toarray()
    row = np.array(row).flatten()

    explanation = explainer.explain_instance(
        data_row=row,
        predict_fn=model.predict_proba,
        num_features=num_features,
    )
    return explanation