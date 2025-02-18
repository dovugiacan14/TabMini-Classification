import torch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from pytorch_tabnet.tab_model import TabNetClassifier


class TabNet(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        time_limit: int = 3600,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 0,
        kwargs: dict = {},
        small_dataset: bool = False,
    ):
        self.time_limit = time_limit
        self.device = device
        self.seed = seed
        self.kwargs = kwargs
        self.result_df = None
        
        self.param_grid = {
            "learning_rate": [0.001, 0.05, 0.1],
            "epochs": [10, 50, 100, 150],
        }

    def fit(self, X, y, X_test, y_test) -> "TabNet":
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
            custom_lr = param["lr"]
            custom_epochs = param["epochs"]
            print(f"\nTraining with parameters: learning_rate = {custom_lr}; epochs= {custom_epochs}")
            current_model = TabNetClassifier(
                n_d=8,
                n_a=8,
                scheduler_params={"is_batch_level": True, "epochs": custom_epochs},
                optimizer_params=dict(lr=custom_lr),
                gamma=1.3,
            )
            current_model.fit(
                X_train=X_train,
                y_train=y_train,
                patience= 1000,
                eval_set=[(X_train, y_train)],
                eval_name=["train"],
            )

            y_pred = current_model.predict(X_test.values)
            f1 = f1_score(y_test.values, y_pred, average="binary")
            acc = accuracy_score(y_test.values, y_pred)

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

    def save_results(self, filename):
        if self.result_df is not None:
            self.result_df.to_csv(filename, index=False)


if __name__ == "__main__":
    x_df = pd.read_csv("dataset/400/bupa/X.csv")
    y_df = pd.read_csv("dataset/400/bupa/y.csv")
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, train_size= 0.8)
    model =TabNet()
    model.fit(X_train, y_train, X_test, y_test)
    model.save_results(filename= "tmp.csv")

