import numpy as np 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer 
import logging 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE

from src.data_loader import load_data, config 

logger = logging.getLogger(__name__) 

def split_data(df: pd.DataFrame): 
    target_col = config["target"]["column"] 

    X = df.drop(columns =[target_col]) 
    y = df[target_col] 

    X_train, X_test, y_train, y_test = train_test_split( 
        X, 
        y, 
        test_size = config["split"]["test_size"], 
        random_state = config["random_seed"], 
        stratify = y if config["split"]["stratify"] else None, 

    ) 

    logger.info( 
        "Split complete: train = %d rows, test = %d rows", 
        X_train.shape[0], 
        X_test.shape[0], 

    ) 
    return X_train, X_test, y_train, y_test  

def build_preprocessor() -> ColumnTransformer: 
    skewed_cols = config["columns"]["skewed_numeric"] 
    all_numeric_cols = (
        config["columns"]["numeric"] + config["columns"]["engineered_numeric"]
    )
    normal_numeric_cols = [
        col for col in all_numeric_cols if col not in skewed_cols
]
    categorical_cols = config["columns"]["categorical"] 

    skewed_pipeline = Pipeline(steps =[ 
        ("log_transform", FunctionTransformer(np.log1p, feature_names_out = "one-to-one")), 
        ("scale", RobustScaler()), 
    ]) 
    normal_numeric_pipeline = Pipeline(steps =[ 
        ("scale", RobustScaler()),
    ]) 
    categorical_pipeline = Pipeline(steps =[ 
        ("encode", OneHotEncoder(handle_unknown ="ignore")),
    ]) 
    preprocessor = ColumnTransformer(transformers =[ 
        ("skewed", skewed_pipeline, skewed_cols), 
        ("normal_numeric", normal_numeric_pipeline, normal_numeric_cols), 
        ("categorical", categorical_pipeline, categorical_cols),
    ]) 
    logger.info( 
        "Preprocessor built: %d skewed, %d normal numeric, %d categorical columns", 
        len(skewed_cols), 
        len(normal_numeric_cols), 
        len(categorical_cols),
    ) 
    return preprocessor 

def apply_smote(X_transformed, y): 
    smote = SMOTE(random_state = config["random_seed"]) 
    X_resampled, y_resampled = smote.fit_resample(X_transformed, y) 

    logger.info( 
        "SMOTE applied: %d rows before, %d rows after (target now %.2f%% positive)", 
        X_transformed.shape[0], 
        X_resampled.shape[0], 
        y_resampled.mean() * 100,
    ) 
    return X_resampled, y_resampled 