import xgboost as xgb
from pathlib import Path 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV 

import tabmini 
from tabmini.estimators import get_available_methods 
from tabmini.types import TabminiDataset 

method_name = "XBoosting Classifier"

# define pipeline 
pipe = Pipeline(
    [
        ("scaling", MinMaxScaler()), 
        ("classify", xgb.XGBClassifier(random_state=42))
    ]
)

# define hyper-parameters 
REGULARIZATION_OPTIONS = ["l2"] 
LAMBDA_OPTIONS = [0.5, 0.01, 0.002, 0.0004] 
param_grid = [
    {
        "classify__penalty": REGULARIZATION_OPTIONS,
        "classify__C": LAMBDA_OPTIONS,
    }
]




