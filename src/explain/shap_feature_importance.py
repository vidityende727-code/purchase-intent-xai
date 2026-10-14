import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.explain.shap_analysis import build_shap_explainer, compute_shap_values
from src.features.engineering import create_features
from src.models.train import train_all_models
from src.preprocessing.pipeline import build_preprocessor, split_data

results_df, models = train_all_models()
catboost_model = models["CatBoost"]

df = load_data()
df = create_features(df)
X_train, X_test, y_train, y_test = split_data(df)

preprocessor = build_preprocessor()
preprocessor.fit(X_train)
X_test_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

explainer = build_shap_explainer(catboost_model)
shap_values = compute_shap_values(explainer, X_test_transformed)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }
).sort_values("mean_abs_shap", ascending=False)

print(shap_importance.head(15))
