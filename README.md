# TabMini-Classification

This repository presents a performance comparison of various classification models on TabMini, a tabular benchmark suite designed specifically for the low-data regime. TabMini consists of 44 binary classification datasets, offering a robust platform for evaluating model effectiveness in scenarios with limited data.

The models implemented in this study include:

- XGBoost
- LightGBM
- Random Forest
- TabR
- TableNet 

## Requirements
 - Python 3.10+ (Recommend: 3.10)
 - pip 24.0

## Installation
1. **Install the required packages from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run**

   Example, we want to train XGBoost model
   ```bash
   python main.py --model 1 --selection False --scale False --save_dir result/
   ```
   The result will be saved in **./result** folder. 

## 🚀 Command Line Arguments

| 🏷 Argument Command | 🔢 Type | 📝 Description |
|--------------------|--------|-------------|
| **--model**       | `int`  | Type of model:  <br> **1** - XGBoost  <br> **2** - LightGBM  <br> **4** - Random Forest  <br> **8** - TabR  <br> **10** - TabNet |
| **--selection**    | `bool`  | Implement feature selections or not.
| **--scale**    | `bool`  | Apply Standard Scaler or not..
| **--save_dir**    | `str`  | The directory to save result.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg