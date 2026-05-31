import pandas as pd
import numpy as np
import os


def load_data(raw_data_dir):
    """
    Loads raw PPG data from the specified directory.
    Assumes CSV files containing PPG and BPM/label columns.
    """
    # Placeholder for actual data loading logic
    print(f"Loading data from {raw_data_dir}...")
    # Example: all_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    return []


def clean_data(data):
    """
    Performs basic cleaning: handling NaNs, normalization.
    """
    print("Cleaning data...")
    # Placeholder for actual cleaning logic
    return data


def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"

    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist. Please place Kaggle data here.")
        return

    data = load_data(raw_dir)
    cleaned_data = clean_data(data)

    print("Data ingestion and cleaning complete (placeholder).")


if __name__ == "__main__":
    main()
