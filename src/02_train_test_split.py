"""
Phase 2: Train / Validation / Test Splitting
Splits preeclampsia.csv into:
  - data/train.csv (70%, N=280)
  - data/validation.csv (15%, N=60)
  - data/test.csv (15%, N=60)
Uses random_state = 42. No stratification by disease target.
Preserves original preeclampsia.csv and copies to data/preeclampsia.csv.
"""

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

def run_train_test_split(
    raw_csv="preeclampsia.csv",
    data_dir="data",
    config_path="config/risk_thresholds.yaml"
):
    os.makedirs(data_dir, exist_ok=True)

    # Load configuration
    seed = 42
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            seed = cfg.get("seed", 42)
            split_cfg = cfg.get("split", {})
            train_ratio = split_cfg.get("train_ratio", 0.70)
            val_ratio = split_cfg.get("val_ratio", 0.15)
            test_ratio = split_cfg.get("test_ratio", 0.15)

    print("=" * 60)
    print("PHASE 2: TRAIN / VALIDATION / TEST DATA SPLITTING")
    print("=" * 60)
    print(f"Random Seed: {seed}")
    print(f"Target Ratios: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")

    df = pd.read_csv(raw_csv)
    n_total = len(df)
    print(f"Total Cohort Size: {n_total} records")

    # Copy original to data/ directory for completeness
    dest_raw = os.path.join(data_dir, "preeclampsia.csv")
    shutil.copyfile(raw_csv, dest_raw)
    print(f"Original dataset mirrored to: {dest_raw}")

    # First split: Train vs Temp (Val + Test)
    # Temp is (val_ratio + test_ratio) = 0.30
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        shuffle=True
    )

    # Second split: Val vs Test from Temp (each gets 0.15 of total, so 50% of temp)
    val_frac_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_frac_of_temp),
        random_state=seed,
        shuffle=True
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path} -> {len(train_df)} samples ({len(train_df)/n_total*100:.1f}%)")
    print(f"Saved: {val_path} -> {len(val_df)} samples ({len(val_df)/n_total*100:.1f}%)")
    print(f"Saved: {test_path} -> {len(test_df)} samples ({len(test_df)/n_total*100:.1f}%)")
    print("Zero Data Leakage Rule: Test set locked and untouched until final evaluation.")
    print("=" * 60)

if __name__ == "__main__":
    run_train_test_split()
