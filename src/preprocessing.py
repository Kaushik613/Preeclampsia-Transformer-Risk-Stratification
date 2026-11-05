"""
Core Preprocessing Module: ClinicalDataPreprocessor
Defines the clinical preprocessor class for strict training-only fitting and zero-leakage inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ClinicalDataPreprocessor:
    def __init__(self):
        # 17 continuous features (clean_sysbp and clean_diabp physiologically mapped)
        self.continuous_cols = [
            'age', 'gest_age', 'height', 'weight', 'bmi',
            'clean_sysbp', 'clean_diabp',
            'hb', 'pcv', 'tsh', 'platelet', 'creatinine',
            'plgf:sflt', 'SEng', 'cysC', 'pp_13', 'glycerides'
        ]
        
        # 5 binary features
        self.binary_cols = ['htn', 'diabetes', 'fam_htn', 'sp_art', 'diet']
        
        # 3 categorical features
        self.categorical_cols = ['occupation', 'activity', 'sleep']
        
        self.scaler = StandardScaler()
        self.cat_vocab = {}          # {col: {orig_val: cat_id}}
        self.cat_cardinalities = {}  # {col: num_categories_including_unk}
        self.train_stats = {}
        self.is_fitted = False

    def _remap_hemodynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps raw swapped headers:
        - raw 'diabp' (120-139 mmHg) -> true physiological systolic BP ('clean_sysbp')
        - raw 'sysbp' (70-89 mmHg)   -> true physiological diastolic BP ('clean_diabp')
        """
        df_clean = df.copy()
        if 'clean_sysbp' not in df_clean.columns:
            if (df_clean['sysbp'] < df_clean['diabp']).all():
                df_clean['clean_sysbp'] = df_clean['diabp'].astype(float)
                df_clean['clean_diabp'] = df_clean['sysbp'].astype(float)
            else:
                df_clean['clean_sysbp'] = df_clean['sysbp'].astype(float)
                df_clean['clean_diabp'] = df_clean['diabp'].astype(float)
        return df_clean

    def fit(self, train_df: pd.DataFrame):
        """Fit scalers and categorical dictionaries on training partition only."""
        df = self._remap_hemodynamics(train_df)

        # 1. Fit Continuous Scaler
        cont_data = df[self.continuous_cols].values
        self.scaler.fit(cont_data)
        
        self.train_stats['cont_mean'] = dict(zip(self.continuous_cols, self.scaler.mean_))
        self.train_stats['cont_scale'] = dict(zip(self.continuous_cols, self.scaler.scale_))

        # 2. Build Categorical Vocabularies (index 0 reserved for UNK)
        for col in self.categorical_cols:
            unique_vals = sorted(df[col].dropna().unique())
            # Map values starting at 1, keeping 0 for unknown/unseen
            mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            self.cat_vocab[col] = mapping
            self.cat_cardinalities[col] = len(mapping) + 1  # +1 for UNK (0)

        # 3. Verify Binary features
        for col in self.binary_cols:
            vals = set(df[col].unique())
            if not vals.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(f"Binary column {col} contains unexpected values: {vals}")

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> dict:
        """Transform dataframe into numpy arrays for continuous, binary, and categorical tokens."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming.")

        df_clean = self._remap_hemodynamics(df)

        # Continuous transformation
        cont_raw = df_clean[self.continuous_cols].values
        cont_scaled = self.scaler.transform(cont_raw).astype(np.float32)

        # Binary transformation (strictly 0 or 1)
        bin_vals = df_clean[self.binary_cols].values.astype(np.int64)

        # Categorical transformation with UNK mapping
        cat_encoded_list = []
        for col in self.categorical_cols:
            vocab = self.cat_vocab[col]
            encoded_col = df_clean[col].map(lambda x: vocab.get(x, 0)).values.astype(np.int64)
            cat_encoded_list.append(encoded_col.reshape(-1, 1))
        cat_vals = np.hstack(cat_encoded_list)

        return {
            "continuous": cont_scaled,
            "binary": bin_vals,
            "categorical": cat_vals,
            "raw_continuous": cont_raw,
            "continuous_cols": self.continuous_cols,
            "binary_cols": self.binary_cols,
            "categorical_cols": self.categorical_cols,
            "df_clean": df_clean
        }

    def inverse_transform_continuous(self, cont_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(cont_scaled)
