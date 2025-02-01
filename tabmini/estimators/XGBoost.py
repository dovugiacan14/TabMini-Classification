import numpy as np 
import pandas as pd 
from xgboost import XGBClassifier 

from sklearn.utils import check_X_y 
from sklearn.metrics import f1_score 
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted, check_array 
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBoost(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            time_limit: int= 3600, 
            device = "cuda", 
            seed: int = 42, 
            kwargs: dict= {}, 
            small_dataset: bool = False 
    ):
        self.time_limit = time_limit
        self.device = device 
        self.seed = seed 
        self.kwargs = kwargs 
        self.result_df = None 

        self.param_grid = {
            'n_estimators': [10, 20, 30] if small_dataset else [50, 100, 200],
            'max_depth': [3, 5] if small_dataset else [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1] if small_dataset else [0.01, 0.1, 0.3]
        }
        
        self.n_classes = 2 
        self.classes_ = [0, 1]
    
    def fit(self, X, y) -> "XGBoost": 
        X, y = check_X_y(X, y, accept_sparse= True)
        results = []
        param_combinations =  [dict(zip(self.param_grid.keys(), v)) for v in np.array(np.meshgrid(*self.param_grid.values())).T.reshape(-1, len(self.param_grid.keys()))]
        for param in param_combinations: 
            param['n_estimators'] = int(param['n_estimators'])
            param['max_depth'] = int(param['max_depth'])
            current_model = XGBClassifier(
                **param,
                objective= 'binary:logistic',
                eval_metric='auc',
                use_label_encoder = False, 
                random_state = self.seed
            )
            current_model.fit(X, y)

            # make predictions 
            y_pred = current_model.predict(X)
            f1 = f1_score(y, y_pred, average='binary')
            acc = accuracy_score(y, y_pred)

            results.append({
                **param,
                'accuracy': acc,
                'f1_score': f1
            })

        self.result_df = pd.DataFrame(results)
    
    def predict_proba(self, X): 
        check_is_fitted(self)
        X = check_array(X, accept_sparse= True)
        probability_positive_class = self.model.predict_proba(X)[:, 1] 
        probability_positive_class_scaled = (probability_positive_class - probability_positive_class.min()) / (
        probability_positive_class.max() - probability_positive_class.min() + 1e-10)
        return np.vstack([1 - probability_positive_class_scaled, probability_positive_class_scaled]).T

    def decision_function(self, X): 
        proba = self.predict_proba(X)
        decision = np.log((proba[:, 1] + 1e-10) / (proba[:, 0] + 1e-10))
        return decision
    
    def save_results(self, filename):
        if self.result_df is not None: 
            self.result_df.to_csv(filename, index= False)
        