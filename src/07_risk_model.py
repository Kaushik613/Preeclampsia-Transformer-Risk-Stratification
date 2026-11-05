"""
Phase 7: Final Clinical Risk Stratification Model
Integrates Preprocessing + FT-Transformer Patient Embedding + Neural Risk Head
Provides end-to-end inference producing:
  - Patient 32-D Representation
  - Continuous Research Risk Index [0, 1]
  - 3-Level Risk Category: LOW / MODERATE / HIGH
  - Patient-level feature contribution breakdown
"""

import os
import sys
import pickle
import yaml
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import ClinicalDataPreprocessor
import importlib
model_module = importlib.import_module("src.04_model")
FTTransformer = model_module.FTTransformer

class PreeclampsiaRiskModel:
    def __init__(
        self,
        checkpoint_path: str = "outputs/checkpoints/best_model.pt",
        preprocessor_path: str = "outputs/preprocessor.pkl",
        config_path: str = "config/risk_thresholds.yaml"
    ):
        self.device = torch.device("cpu")
        
        # Load configuration
        self.cfg = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        
        thresholds = self.cfg.get("thresholds", {})
        self.low_upper = thresholds.get("low_risk_upper", 0.35)
        self.mod_upper = thresholds.get("moderate_risk_upper", 0.65)

        # Load preprocessor
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        # Initialize and load model
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        num_cont = len(self.preprocessor.continuous_cols)
        bin_cards = [2] * len(self.preprocessor.binary_cols)
        cat_cards = [self.preprocessor.cat_cardinalities[col] for col in self.preprocessor.categorical_cols]

        self.model = FTTransformer(
            num_continuous=num_cont,
            binary_cardinalities=bin_cards,
            categorical_cardinalities=cat_cards,
            token_dim=ckpt.get("token_dim", 32),
            num_heads=ckpt.get("num_heads", 4),
            num_layers=ckpt.get("num_layers", 2)
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, df: pd.DataFrame, compute_contributions: bool = True) -> pd.DataFrame:
        """
        Runs inference on patient dataframe.
        Returns dataframe with risk_score, risk_category, patient_embedding, and top features.
        """
        feats = self.preprocessor.transform(df)
        cont = torch.tensor(feats["continuous"], dtype=torch.float32, device=self.device)
        bin_x = torch.tensor(feats["binary"], dtype=torch.long, device=self.device)
        cat_x = torch.tensor(feats["categorical"], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model.forward_reconstruction(cont, bin_x, cat_x)
            risk_scores = out["predicted_risk"].cpu().numpy()
            embeddings = out["patient_embedding"].cpu().numpy()

        categories = []
        for s in risk_scores:
            if s < self.low_upper:
                categories.append("LOW")
            elif s < self.mod_upper:
                categories.append("MODERATE")
            else:
                categories.append("HIGH")

        results = df.copy()
        results["risk_score"] = np.round(risk_scores, 4)
        results["risk_category"] = categories

        # Feature contribution analysis via baseline feature deviation
        if compute_contributions:
            top_feats_list = self._compute_feature_contributions(df, feats, risk_scores)
            results["top_feature_1"] = [tf[0] if len(tf) > 0 else "N/A" for tf in top_feats_list]
            results["top_feature_2"] = [tf[1] if len(tf) > 1 else "N/A" for tf in top_feats_list]
            results["top_feature_3"] = [tf[2] if len(tf) > 2 else "N/A" for tf in top_feats_list]

        return results, embeddings

    def _compute_feature_contributions(self, df: pd.DataFrame, feats: dict, baseline_scores: np.ndarray) -> list:
        """
        Computes patient-specific top contributing features by masking each feature
        and measuring sensitivity shift on the patient's risk profile.
        """
        n_samples = len(df)
        all_feature_names = self.preprocessor.continuous_cols + self.preprocessor.binary_cols + self.preprocessor.categorical_cols
        top_features_per_patient = []

        cont_orig = torch.tensor(feats["continuous"], dtype=torch.float32, device=self.device)
        bin_orig = torch.tensor(feats["binary"], dtype=torch.long, device=self.device)
        cat_orig = torch.tensor(feats["categorical"], dtype=torch.long, device=self.device)

        n_cont = cont_orig.shape[1]
        n_bin = bin_orig.shape[1]
        n_cat = cat_orig.shape[1]

        # Compute perturbation deltas
        deltas = np.zeros((n_samples, len(all_feature_names)))

        with torch.no_grad():
            # Continuous perturbation
            for i in range(n_cont):
                cont_masked = cont_orig.clone()
                cont_masked[:, i] = 0.0
                out_m = self.model.forward_reconstruction(cont_masked, bin_orig, cat_orig)
                deltas[:, i] = np.abs(baseline_scores - out_m["predicted_risk"].cpu().numpy())

            # Binary perturbation
            for j in range(n_bin):
                bin_masked = bin_orig.clone()
                bin_masked[:, j] = 0
                out_m = self.model.forward_reconstruction(cont_orig, bin_masked, cat_orig)
                deltas[:, n_cont + j] = np.abs(baseline_scores - out_m["predicted_risk"].cpu().numpy())

            # Categorical perturbation
            for k in range(n_cat):
                cat_masked = cat_orig.clone()
                cat_masked[:, k] = 0
                out_m = self.model.forward_reconstruction(cont_orig, bin_orig, cat_masked)
                deltas[:, n_cont + n_bin + k] = np.abs(baseline_scores - out_m["predicted_risk"].cpu().numpy())

        # Clean names for presentation
        name_map = {
            "plgf:sflt": "Angiogenic ratio (plgf:sflt)",
            "SEng": "Soluble endoglin (SEng)",
            "clean_sysbp": "Systolic blood pressure",
            "clean_diabp": "Diastolic blood pressure",
            "htn": "Chronic hypertension",
            "diabetes": "Pre-existing diabetes",
            "fam_htn": "Family history of preeclampsia",
            "sp_art": "Uterine/spiral artery resistance",
            "creatinine": "Serum creatinine",
            "cysC": "Cystatin C",
            "platelet": "Platelet count",
            "bmi": "Maternal BMI",
            "age": "Maternal age",
            "pp_13": "Placental protein 13",
            "glycerides": "Triglycerides",
            "sleep": "Sleep quality",
            "activity": "Physical activity",
            "diet": "Dietary profile",
            "hb": "Hemoglobin",
            "pcv": "Hematocrit (PCV)",
            "tsh": "Thyroid stimulating hormone",
            "gest_age": "Gestational age"
        }

        for idx in range(n_samples):
            sorted_indices = np.argsort(-deltas[idx])
            top_3 = [name_map.get(all_feature_names[i], all_feature_names[i]) for i in sorted_indices[:3]]
            top_features_per_patient.append(top_3)

        return top_features_per_patient

def run_risk_model(output_dir="outputs"):
    print("=" * 60)
    print("PHASE 7: FINAL CLINICAL RISK STRATIFICATION MODEL")
    print("=" * 60)
    model = PreeclampsiaRiskModel()

    # Generate Model Architecture Summary
    summary_path = os.path.join(output_dir, "model_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# FT-Transformer Model Architecture & Design\n\n")
        f.write("## 1. Network Overview\n")
        f.write("- **Model Class:** Feature Tokenizer + Transformer (FT-Transformer)\n")
        f.write("- **Token Dimension:** 32\n")
        f.write("- **Encoder Layers:** 2 Transformer blocks\n")
        f.write("- **Multi-Head Attention:** 4 heads per layer\n")
        f.write("- **Feed-Forward Dimension:** 64 with GELU activation\n")
        f.write("- **Dropout Rate:** 0.15 with Pre-LayerNorm\n")
        f.write("- **Patient Representation:** 32-D embedding from `[CLS]` token\n\n")

        f.write("## 2. Feature Tokenization\n")
        f.write("- **Continuous Features (17):** Per-feature learned linear projection ($W_i x_i + b_i \\in \\mathbb{R}^{32}$)\n")
        f.write("- **Binary Features (5):** Learned 2-class embeddings $\\mathbb{R}^{32}$\n")
        f.write("- **Categorical Features (3):** Learned multi-class embeddings with dedicated `UNK` token ID\n")
        f.write("- **Sequence Structure:** `[CLS]` (1) + Continuous (17) + Binary (5) + Categorical (3) = 26 tokens\n\n")

        f.write("## 3. Dual-Head Design\n")
        f.write("- **Self-Supervised Denoising Head:** Feature-level reconstruction heads predicting uncorrupted clinical values.\n")
        f.write("- **Neural Clinical Risk Head:** Multi-layer perceptron with LayerNorm and Sigmoid mapping the 32-D patient embedding to the continuous Research Risk Index ($[0, 1]$).\n\n")

        f.write("## 4. Clinical Context & Risk Boundaries\n")
        f.write("- **LOW RISK:** Score < 0.35\n")
        f.write("- **MODERATE RISK:** 0.35 ≤ Score < 0.65\n")
        f.write("- **HIGH RISK:** Score ≥ 0.65\n")
        f.write("- *Mandatory Clinical Note:* These categories represent an estimated preeclampsia-related risk profile based on clinical anchors; they are NOT a confirmed medical diagnosis.\n")

    print(f"Model summary written to: {summary_path}")
    print("Risk model initialized successfully.")
    print("=" * 60)
    return model

if __name__ == "__main__":
    run_risk_model()
