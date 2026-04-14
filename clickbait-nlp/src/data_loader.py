import pandas as pd
import os

def load_and_combine_data():

    base_dir = os.path.dirname(os.path.dirname(__file__))
    folder_path = os.path.join(base_dir, "data", "annotated", "csv")

    all_dfs = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print("Folder path:", folder_path)
    print("Files:", os.listdir(folder_path))

    return combined_df