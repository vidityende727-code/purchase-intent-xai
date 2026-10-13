from src.models.train import train_all_models
from src.data_loader import load_data
from src.features.engineering import create_features
from src.preprocessing.pipeline import split_data, build_preprocessor
import pandas as pd

results_df, models = train_all_models()
catboost_model = models["CatBoost"]

df = load_data()
df = create_features(df)
X_train, X_test, y_train, y_test = split_data(df)

preprocessor = build_preprocessor()
preprocessor.fit(X_train)

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": catboost_model.feature_importances_,
}).sort_values("importance", ascending=False)

print(importance_df.head(15))