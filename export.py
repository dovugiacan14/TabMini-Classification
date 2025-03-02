import tabmini
from tabmini.data.data_info import data_groups
from tabmini.types import TabminiDataset
from tabmini.data.data_processing import DataProcessor

data_processor = DataProcessor()
dataset: TabminiDataset = tabmini.load_dataset() 

data_processor.export(dataset, data_groups, "dataset2")
