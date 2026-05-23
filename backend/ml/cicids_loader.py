import pandas as pd
from pathlib import Path
import numpy as np


DATASET_DIR = Path("data/cicids2017")


def load_cicids():
    csv_files = DATASET_DIR.glob("*.csv")

    frames = []

    for file in csv_files:
        print(f"Loading {file.name}")

        df = pd.read_csv(file)

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    data.columns = data.columns.str.strip()

    # Convert labels
    data["Label"] = data["Label"].apply(
        lambda x: 0 if x == "BENIGN" else 1
    )

    # Keep only numeric features
    X = data.drop(columns=["Label"]).select_dtypes(include=["number"])

    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN
    X = X.fillna(0)

    X = X.astype(np.float32)

    y = data["Label"]

    print(f"Dataset Loaded: {len(X)} samples")

    return X.values, y.values