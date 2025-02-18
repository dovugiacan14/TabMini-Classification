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
from tabmini.estimators.TabR import TabR, TabRTrainer 
from tabmini.estimators.TabNet import TabNet
from tabmini.types import TabminiDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="TabMini-Classification Options.")
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 4, 8, 10],
        default= 10,
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
            model = TabRTrainer(small_dataset= True) 
        elif args.model == 10: 
            model = TabNet(small_dataset = True)

        model.fit(X_train, y_train, X_test, y_test)
        model.save_results(filename=working_directory / f"{dt_name}_{num_records}.csv")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
