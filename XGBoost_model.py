import tabmini
from pathlib import Path
from tabmini.estimators import XGBoost
from tabmini.types import TabminiDataset


def main(folder):
    working_directory = Path.cwd() / folder
    working_directory.mkdir(parents=True, exist_ok=True)

    # load dataset
    dataset: TabminiDataset = tabmini.load_dataset()
    dataset_name_lst = list(dataset.keys())

    # process
    for dt_name in dataset_name_lst:
        X, y = dataset[dt_name]
        if 2 in y.values: 
            y = (y == 2).astype(int)
        num_records = len(X)
        model = XGBoost(small_dataset=True)
        model.fit(X, y)

        model.save_results(filename= working_directory / f"{dt_name}_{num_records}.csv")

if __name__ == "__main__":
    main("result")