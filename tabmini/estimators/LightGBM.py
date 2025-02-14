from pathlib import Path

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_is_fitted, check_array


class LightGBM(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that uses LightGBM to fit and predict data."""

    def __init__(
            self,
            time_limit: int = 3600,
            device: str = "cpu",
            seed: int = 0,
            kwargs: dict = {},
            small_dataset: bool = False
    ):
        self.time_limit = time_limit
        self.device = device
        self.seed = seed
        self.kwargs = kwargs
        self.result_df = None

        # specify that this is a binary classifier
        # self.n_classes_ = 2
        # self.classes_ = [0, 1]

        self.param_grid = {
            "n_estimators": [10, 20, 30] if small_dataset else [50, 100, 200],
            "max_depth": [3, 5] if small_dataset else [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1] if small_dataset else [0.01, 0.1, 0.3],
        }

    def fit(self, X, y, X_test, y_test) -> 'LightGBM':
        """

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X_train, y_train = check_X_y(X, y, accept_sparse=True)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

        results = []
        best_f1 = -1
        best_model = None

        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in np.array(np.meshgrid(*self.param_grid.values())).T.reshape(
                -1, len(self.param_grid.keys())
            )
        ]

        for param in param_combinations:
            param["n_estimators"] = int(param["n_estimators"])
            param["max_depth"] = int(param["max_depth"])

            current_model = LGBMClassifier(
                **param,
                objective="binary",
                random_state=self.seed
            )
            current_model.fit(X_train, y_train)

            # make predictions
            y_pred = current_model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="binary")
            acc = accuracy_score(y_test, y_pred)

            results.append({**param, "accuracy": acc, "f1_score": f1})

            # Update best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = current_model

        self.result_df = pd.DataFrame(results).sort_values(
            by="f1_score", ascending=False
        )
        self.model = best_model
        self.best_params_ = best_model.get_params() if best_model else None

        # return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict the probability of each sample belonging to each class.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like, shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the model.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse=True)

        probability_positive_class = self.model.predict(X)
        probability_positive_class_scaled = (probability_positive_class - probability_positive_class.min()) / (
                probability_positive_class.max() - probability_positive_class.min() + 1e-10)

        # if this contains a NaN, replace it with 0.5
        probability_positive_class_scaled[np.isnan(probability_positive_class_scaled)] = 0.5

        # Create a 2D array with probabilities of both classes
        return np.vstack([1 - probability_positive_class_scaled, probability_positive_class_scaled]).T

    def decision_function(self, X):
        # Get the probabilities from predict_proba
        proba = self.predict_proba(X)

        # Calculate the log of ratios for binary classification
        decision = np.log((proba[:, 1] + 1e-10) / (proba[:, 0] + 1e-10))

        return decision

    def save_results(self, filename):
        if self.result_df is not None:
            self.result_df.to_csv(filename, index=False)