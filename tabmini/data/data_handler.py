import os 
import requests
import pandas as pd

from tabmini.data import data_info
from tabmini.types import TabminiDataset

GITHUB_URL = 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets'
suffix = '.tsv.gz'


def get_dataset_url(GITHUB_URL, dataset_name, suffix):
    dataset_url = '{GITHUB_URL}/{DATASET_NAME}/{DATASET_NAME}{SUFFIX}'.format(
                                GITHUB_URL=GITHUB_URL,
                                DATASET_NAME=dataset_name,
                                SUFFIX=suffix
                                )

    re = requests.get(dataset_url)
    if re.status_code != 200:
        raise ValueError('Dataset not found in PMLB.')
    return dataset_url


def fetch_data(dataset_name, return_X_y=False, local_cache_dir=None, dropna=True):
    """Download a data set from the PMLB, (optionally) store it locally, and return the data set.

    You must be connected to the internet if you are fetching a data set that is not cached locally.

    Parameters
    ----------
    dataset_name: str
        The name of the data set to load from PMLB.
    return_X_y: bool (default: False)
        Whether to return the data in scikit-learn format, with the features 
        and labels stored in separate NumPy arrays.
    local_cache_dir: str (default: None)
        The directory on your local machine to store the data files.
        If None, then the local data cache will not be used.
    dropna: bool
        If True, pmlb will drop NAs in exported dataset.

    Returns
    ----------
    dataset: pd.DataFrame or (array-like, array-like)
        if return_X_y == False: A pandas DataFrame containing the fetched data set.
        if return_X_y == True: A tuple of NumPy arrays containing (features, labels)

    """

    if local_cache_dir is None:
        dataset_url = get_dataset_url(GITHUB_URL,
                                        dataset_name, suffix)
        dataset = pd.read_csv(dataset_url, sep='\t', compression='gzip')
    else:
        dataset_path = os.path.join(local_cache_dir, dataset_name,
                                    dataset_name+suffix)

        # Use the local cache if the file already exists there
        if os.path.exists(dataset_path):
            dataset = pd.read_csv(dataset_path, sep='\t', compression='gzip')
        # Download the data to the local cache if it is not already there
        else:
            dataset_url = get_dataset_url(GITHUB_URL,
                                            dataset_name, suffix)
            dataset = pd.read_csv(dataset_url, sep='\t', compression='gzip')
            dataset_dir = os.path.split(dataset_path)[0]
            if not os.path.isdir(dataset_dir):
                os.makedirs(dataset_dir)
            dataset.to_csv(dataset_path, sep='\t', compression='gzip',
                    index=False)

    if dropna:
        dataset.dropna(inplace=True)
    if return_X_y:
        X = dataset.drop('target', axis=1).values
        y = dataset['target'].values
        return (X, y)
    else:
        return dataset

def load_dataset(reduced: bool = False) -> TabminiDataset:
    """
    Load the dataset for AutoML. The datasets are loaded from the PMLB library.
    :param reduced: Whether to exclude the datasets that have been used to train TabPFN. Default is False.
    :return: A dictionary containing the loaded dataset. The key is the dataset name and the value is a tuple containing
    the input features and the target variable.
    """
    dataset = {}

    print("Loading dataset...")
    for idx, _datasets in enumerate(data_info.files):
        datasets = _datasets if not reduced else [file for file in _datasets if data_info.is_not_excluded(file)]

        for dataset_name in datasets:
            try: 
                fetched_data = fetch_data(dataset_name)
            except: 
                new_dataset_name = "_deprecated_" + dataset_name
                fetched_data = fetch_data(new_dataset_name)

            if not isinstance(fetched_data, pd.DataFrame):
                print(f"Dataset {dataset_name} is not an instance of DataFrame. Skipping...")
                continue

            data = fetched_data.sample(frac=1, random_state=42).reset_index(drop=True)

            dataset[dataset_name] = (data.drop(columns=["target"]), data["target"])

    # Print on the same line
    print("Dataset loaded.")

    return dataset


def load_dummy_dataset() -> TabminiDataset:
    """
    Load a smaller subset of the dataset for AutoML. The datasets are loaded from the PMLB library.
    This is for testing purposes only.
    :return: A dictionary containing the loaded dataset. The key is the dataset name and the value is a tuple containing
    the input features and the target variable.
    """
    print("YOU ARE USING THE DUMMY DATASET LOADER. THIS IS FOR TESTING PURPOSES ONLY.")

    dataset = {}

    # We want to load the first ten rows of every dataset
    for idx, _datasets in enumerate(data_info.files[0:2]):
        for dataset_name in _datasets[0:2]:
            fetched_data = fetch_data(dataset_name)
            if not isinstance(fetched_data, pd.DataFrame):
                print(f"Dataset {dataset_name} is not an instance of DataFrame. Skipping...")
                continue

            data = fetched_data.sample(frac=1, random_state=42).reset_index(drop=True)

            dataset[dataset_name] = (data.drop(columns=["target"]).head(20), data["target"].head(20))

    return dataset
