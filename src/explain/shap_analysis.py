import logging 
import shap 
import joblib 
import pandas as pd 
from pathlib import Path 
import matplotlib.pyplot as plt 
from src.data_loader import config 

logger = logging.getLogger(__name__) 

def load_trained_artifacts(): 
    models_dir = Path(config["paths"]["models_dir"]) 

    model = joblib.load(models_dir / "catboost_model.joblib") 
    preprocessor = joblib.load(models_dir / "preprocessor.joblib") 

    logger.info("Loaded trained model and preprocessor from %s", models_dir) 

    return model, preprocessor  

def build_shap_explainer(model): 
    explainer = shap.TreeExplainer(model) 
    logger.info("SHAP TreeExplainer built for model: %s", type(model).__name__) 
    return explainer 

def compute_shap_values(explainer, X_transformed): 
    shap_values = explainer.shap_values(X_transformed) 
    logger.info("SHAP values computed for %d rows", X_transformed.shape[0]) 
    return shap_values  

def get_global_importance(shap_values, feature_names) -> pd.DataFrame: 
    mean_abs_shap = abs(shap_values).mean(axis=0) 

    importance_df = pd.DataFrame({ 
        "feature": feature_names, 
        "mean_abs_shap": mean_abs_shap, 

    }).sort_values("mean_abs_shap", ascending = False) 
    logger.info("Top 5 most important features:\n%s", importance_df.head(5).to_string(index = False)) 

    return importance_df  

def plot_summary(shap_values, X_transformed, feature_names, save_path = None): 
    plt.figure() 
    shap.summary_plot(shap_values, X_transformed, feature_names = feature_names, show = False) 

    if save_path: 
        plt.savefig(save_path, bbox_inches = "tight") 
        logger.info("Summary plot saved to %s", save_path) 

    plt.close()  


def explain_single_prediction(model, explainer, shap_values, X_transformed, feature_names, index: int): 
    prediction_proba = model.predict_proba(X_transformed[index: index + 1])[0,1] 

    row_shap_values = shap_values[index] 

    contributions = pd.DataFrame({ 
        "feature": feature_names, 
        "shap_value": row_shap_values, 

    }).sort_values("shap_value", key = abs, ascending = False) 

    logger.info( 
        "Prediction for row %d: %.2f%% probability of purchase", 
        index, 
        prediction_proba * 100,
    ) 
    logger.info("Top contributing features:\n%s", contributions.head(10).to_string(index=False)) 

    return contributions 