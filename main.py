import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.optim as optim 

import tabmini
import argparse
from pathlib import Path
from tabmini.data.data_processing import DataProcessor
from tabmini.estimators import XGBoost
from tabmini.estimators import LightGBM
from tabmini.estimators.RF import RandomForest
from tabmini.estimators.TabR import TabR
from tabmini.types import TabminiDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="TabMini-Classification Options.")
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 4, 8, 10],
        default= 8,
        help="Type of model (1: XGBoost, 2: LightGBM, 4: Random Forest, 8: TabR, 10: TabNet)",
    )
    parser.add_argument(
        "--selection", 
        type = bool, 
        default= False, 
        help= "Implement feature selections or not."
    )
    parser.add_argument(
        "--scale", 
        type= bool, 
        default= False, 
        help= "Apply Standard Scaler or not."
    )
    parser.add_argument(
        "--save_dir", type=str, default="result", help="Folder to save result."
    )
    return parser.parse_args()


def main(args):

    working_directory = Path.cwd() / args.save_dir
    working_directory.mkdir(parents=True, exist_ok=True)

    # load dataset
    data_processor = DataProcessor()
    dataset: TabminiDataset = tabmini.load_dataset()
    dataset_name_lst = list(dataset.keys())

    # process
    for dt_name in dataset_name_lst:
        X, y = dataset[dt_name]
        if 2 in y.values:
            y = (y == 2).astype(int)
        num_records = len(X)

        # preprocessing data 
        if args.selection: 
            X = data_processor.feature_selection(X, y) 
        if args.scale: 
            X = data_processor.normalize_data(X)
        X_train, X_test, y_train, y_test = data_processor.split_train_test(X, y)

        # train and predict        
        if args.model == 1:
            model = XGBoost(small_dataset=True)
        elif args.model == 2:
            model = LightGBM(small_dataset=True)
        elif args.model == 4:
            model = RandomForest(small_dataset=True)
        elif args.model == 8:
            # convert data to Tensor 
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to("cpu")
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to("cpu")
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).squeeze().long().to("cpu")
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze().long().to("cpu")

            # convert X_train_tensor to dictionary to fit with TabR 
            X_train_dict = {"num": X_train_tensor}
            X_test_dict = {"num": X_test_tensor}

            model = TabR(
                n_num_features= X_train.shape[1], 
                n_bin_features= 0,          # assume there is no binary feature
                cat_cardinalities= [],      # assume there is no category feature 
                n_classes= len(np.unique(y_train)), 
                num_embeddings= None,
                d_main= 128, 
                d_multiplier= 2, 
                encoder_n_blocks= 3, 
                predictor_n_blocks= 2, 
                mixer_normalization= "auto",
                context_dropout= 0.1,
                dropout0= 0.2, 
                dropout1= "dropout0",
                normalization= "BatchNorm1d",
                activation= "ReLU", 
                memory_efficient= False
            ).to("cpu")

            citerion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr= 0.001)
            epochs = 50
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                outputs = model(
                    x_=X_train_dict,
                    y=y_train_tensor,
                    candidate_x_=X_train_dict,
                    candidate_y=y_train_tensor,
                    context_size=5,
                    is_train=True
                )

                loss = citerion(outputs.squeeze(), y_train_tensor.float())
                loss.backward()
                optimizer.step()
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

            # evaluate 
            acc, f1 = model.evaluate(X_test_dict, y_test_tensor)
            result = pd.DataFrame({"accuracy": acc, "f1_score": f1})

            # export result
            model.save_results(
                result= result, 
                filename=working_directory / f"{dt_name}_{num_records}.csv"
            )
            continue 

        model.fit(X_train, y_train, X_test, y_test)

        model.save_results(filename=working_directory / f"{dt_name}_{num_records}.csv")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
