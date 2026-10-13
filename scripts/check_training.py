from src.models.train import train_all_models

results_df, models = train_all_models()
print(results_df)