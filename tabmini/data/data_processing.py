import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class DataProcessor:
    def __init__(self):
        pass

    def load_data(self, x_path, y_path):
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        return X, y

    def split_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return  X_train, X_test, y_train, y_test

    def feature_selection(self, X, y, threshold=0.07):
        try:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            # get feature importances
            importances = model.feature_importances_
            feature_names = X.columns

            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            )
            selected_features = importance_df[importance_df["Importance"] > threshold][
                "Feature"
            ].tolist()

            if selected_features:
                return X[selected_features]
            else:
                return X
        except Exception as e:
            print(f"Error occurred when selecting features: {e}")
            return X

    def normalize_data(self, X: pd.DataFrame):
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            return pd.DataFrame(X_scaled, columns=X.columns)
        except Exception as e:
            print(f"Error occurred when applying normalization: {e}")
            return X

    def export(self, dataset, data_info: dict, folder_path: str):
        working_directory = Path.cwd() / folder_path
        working_directory.mkdir(parents=True, exist_ok=True)
        for num_records, dataset_name_lst in data_info.items():
            sub_folder = os.path.join(working_directory, num_records)
            os.makedirs(sub_folder, exist_ok=True)
            for dt_name in dataset_name_lst:
                data_folder = os.path.join(sub_folder, dt_name)
                os.makedirs(data_folder, exist_ok=True)
                feature_filename = os.path.join(data_folder, "X.csv")
                target_filename = os.path.join(data_folder, "y.csv")

                # write dataset to corresponding file
                dataset[dt_name][0].to_csv(feature_filename, index=False)
                dataset[dt_name][1].to_csv(target_filename, index=False)
        print("Done.! ✅")
