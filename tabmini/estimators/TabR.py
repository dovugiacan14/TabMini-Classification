# import lib
import delu
import math
import numpy as np
import pandas as pd
import faiss
import faiss.contrib.torch_utils
from itertools import product
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def get_d_out(n_classes: Optional[int]) -> int:
    return 1 if n_classes is None or n_classes == 2 else n_classes


class TabR(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        num_embeddings: Optional[dict],
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal["auto"]],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal["dropout0"]],
        normalization: str,
        activation: str,
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == "auto":
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == "dropout0":
            dropout1 = dropout0

        self.one_hot_encoder = None
        self.num_embeddings = None

        # >>> Encoder
        d_in = (
            n_num_features
            * (1 if num_embeddings is None else num_embeddings["d_embedding"])
            + n_bin_features
            + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> Retrieval
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> Predict
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, get_d_out(n_classes)),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get("num")
        x_bin = x_.get("bin")
        x_cat = x_.get("cat")
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
        self,
        *,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        # >>>
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                            candidate_x_, self.candidate_encoding_batch_size
                        )
                    ]
                )
            )

        x, k = self._encode(x_)
        if is_train:
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        # else:
        #     assert y is None

        # >>>
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = faiss.IndexFlatL2(d_main)
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            candidate_k = candidate_k.cpu().numpy()
            self.search_index.add(candidate_k)  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            k_np = k.detach().cpu().numpy()
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k_np, context_size + (1 if is_train else 0)
            )

            context_idx = torch.tensor(context_idx, device=device)
            distances = torch.tensor(distances, device=device)

            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        context_k = torch.from_numpy(context_k).float()
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        context_k = context_k.to(k.dtype).to(k.device)

        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x
    

class TabRTrainer:
    def __init__(
        self,
        time_limit: int = 3600,
        device: str = "cpu",
        seed: int = 42,
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

        self.default_model_params = {
            "d_main": 128,
            "d_multiplier": 2,
            "encoder_n_blocks": 3,
            "predictor_n_blocks": 2,
            "mixer_normalization": "auto",
            "context_dropout": 0.1,
            "dropout0": 0.2,
            "dropout1": "dropout0",
            "normalization": "BatchNorm1d",
            "activation": "ReLU",
            "memory_efficient": False,
        }

    def fit(self, X, y, X_val=None, y_val=None) -> "TabRTrainer":
        X_train, y_train = check_X_y(X, y, accept_sparse=True)
        if X_val is None or y_val is None:
            X_val, y_val = X_train, y_train

        # convert input data to Tensor
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(self.device)
        self.y_train_tensor = (
            torch.tensor(y_train, dtype=torch.float32).squeeze().long().to(self.device)
        )
        self.y_val_tensor = (
            torch.tensor(y_val.values, dtype=torch.float32).squeeze().long().to(self.device)
        )

        self.X_train_dict = {"num": self.X_train_tensor}
        self.X_val_dict = {"num": self.X_val_tensor}

        results = []
        best_model = None

        # Create the parameter combinations
        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in np.array(np.meshgrid(*self.param_grid.values())).T.reshape(
                -1, len(self.param_grid.keys())
            )
        ]

        for param in param_combinations:
            print(f"\nTraining with parameters: {param}")
            current_model = TabR(
                n_num_features=self.X_train_tensor.shape[1],
                n_bin_features=0,
                cat_cardinalities=[],
                n_classes=len(np.unique(self.y_train_tensor.cpu().numpy())),
                num_embeddings=None,
                **self.default_model_params,
            ).to(self.device)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(
                current_model.parameters(), lr=param["learning_rate"]
            )

            # Training loop
            for epoch in range(int(param["epochs"])):
                current_model.train()
                optimizer.zero_grad()

                outputs = current_model(
                    x_=self.X_train_dict,
                    y=self.y_train_tensor,
                    candidate_x_=self.X_train_dict,
                    candidate_y=self.y_train_tensor,
                    context_size=5,
                    is_train=True,
                )

                loss = criterion(outputs.squeeze(), self.y_train_tensor.float())
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{param['epochs']}] - Loss: {loss.item():.4f}"
                    )

            # Evaluation
            current_model.eval()
            with torch.no_grad():
                y_pred = current_model(
                    x_=self.X_val_dict,
                    y=None,
                    candidate_x_=self.X_val_dict,
                    candidate_y=self.y_val_tensor,
                    context_size=5,
                    is_train=False,
                )
                # y_pred = y_pred.argmax(dim=1)
                y_pred = torch.sigmoid(y_pred)  
                y_pred = (y_pred > 0.5).long() 

            # Calculate metrics
            y_val_np = self.y_val_tensor.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()

            acc = accuracy_score(y_val_np, y_pred_np)
            f1 = f1_score(y_val_np, y_pred_np, average="weighted")

            # Store results
            results.append(
                {
                    **param,
                    "accuracy": acc,
                    "f1_score": f1,
                }
            )

        # Save results to DataFrame
        self.result_df = pd.DataFrame(results).sort_values(
            by="f1_score", ascending=False
        )
        self.model = best_model
        self.best_params_ = {k: param[k] for k in self.param_grid.keys()}

        return self

    def predict(self, X):
        check_is_fitted(self, ["model", "best_params_"])
        X = check_array(X, accept_sparse=True)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        X_dict = {"num": X_tensor}

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                x_=X_dict,
                y=None,
                candidate_x_=X_dict,
                candidate_y=None,
                context_size=5,
                is_train=False,
            )
            predictions = predictions.argmax(dim=1)

        return predictions.cpu().numpy()

    def save_results(self, filename):
        if self.result_df is not None:
            self.result_df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")


if __name__ == "__main__":
    x_df = pd.read_csv("dataset/400/bupa/X.csv")
    y_df = pd.read_csv("dataset/400/bupa/y.csv")
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, train_size= 0.8)
    model = TabRTrainer()
    model.fit(X_train, y_train, X_test, y_test)
    model.save_results(filename= "tmp.csv")