"""
Script: prepare_flood_data.py
Description: Download and prepare a fixed-seed sample (30,000 rows) of Kaggle Playground Series s4e5
Destination: data/processed/flood_sample_30k.csv
Target: FloodProbability (continuous 0-1)
"""

import os
import sys
import pandas as pd
import requests

FLOOD_RAW_URL = "https://raw.githubusercontent.com/Devaaldo/flood-probability-prediction/main/data/flood/train.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "flood_sample_30k.csv")
SAMPLE_SIZE = 30000
RANDOM_STATE = 42


def prepare_flood_data(sample_size: int = SAMPLE_SIZE, random_state: int = RANDOM_STATE):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"[OK] flood_sample_30k.csv already exists at {OUTPUT_FILE} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df

    print(f"Downloading and sampling Flood dataset ({sample_size} rows, random_state={random_state})...")
    print(f"Reading from {FLOOD_RAW_URL} (this may take ~20-30 seconds)...")

    # Read the dataset
    df_raw = pd.read_csv(FLOOD_RAW_URL)
    raw_rows = len(df_raw)
    print(f"Loaded raw dataset with {raw_rows} rows and {df_raw.shape[1]} columns.")

    # Validate target and id
    if "FloodProbability" not in df_raw.columns:
        raise ValueError("Missing FloodProbability column!")

    # Fixed-seed random sample
    if raw_rows > sample_size:
        df_sample = df_raw.sample(n=sample_size, random_state=random_state)
    else:
        df_sample = df_raw

    # Store metadata about original size
    df_sample.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Saved {len(df_sample)} sampled rows (from {raw_rows} raw) to {OUTPUT_FILE}")
    return df_sample


if __name__ == "__main__":
    prepare_flood_data()
