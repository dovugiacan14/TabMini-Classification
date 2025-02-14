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
        default=1,
        help="Type of model (1: XGBoost, 2: LightGBM, 4: Random Forest, 8: TabR, 10: TabNet)",
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
        if args.model == 1:
            X = data_processor.feature_selection(X, y)
            X_train, X_test, y_train, y_test = data_processor.split_train_test(X, y)
            model = XGBoost(small_dataset=True)
        elif args.model == 2:
            X = data_processor.feature_selection(X, y)
            X_train, X_test, y_train, y_test = data_processor.split_train_test(X, y)
            model = LightGBM(small_dataset=True)
        elif args.model == 4:
            X = data_processor.feature_selection(X, y)
            X_train, X_test, y_train, y_test = data_processor.split_train_test(X, y)
            model = RandomForest(small_dataset=True)
        elif args.model == 8:
            X = data_processor.feature_selection(X, y)
            X = data_processor.normalize_data(X) # need to apply StandardScaler to apply TabR
            model = TabR()
        model.fit(X_train, y_train, X_test, y_test)

        model.save_results(filename=working_directory / f"{dt_name}_{num_records}.csv")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
