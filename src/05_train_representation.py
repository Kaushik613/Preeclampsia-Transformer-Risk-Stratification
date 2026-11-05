"""
Phase 5: Self-Supervised Denoising Representation Pretraining
Trains the FT-Transformer on tabular patient records using a denoising autoencoder objective.
Randomly masks 15% of input features and trains the network to reconstruct the original clean values.
Saves the best model checkpoint to outputs/checkpoints/best_model.pt and records training loss curves.
"""

import os
import sys
import pickle
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import ClinicalDataPreprocessor
import importlib
model_module = importlib.import_module("src.04_model")
FTTransformer = model_module.FTTransformer

class ClinicalPatientDataset(Dataset):
    def __init__(self, features: dict, ref_scores: np.ndarray = None):
        self.cont = torch.tensor(features["continuous"], dtype=torch.float32)
        self.bin = torch.tensor(features["binary"], dtype=torch.long)
        self.cat = torch.tensor(features["categorical"], dtype=torch.long)
        if ref_scores is not None:
            self.ref_risk = torch.tensor(ref_scores, dtype=torch.float32)
        else:
            self.ref_risk = torch.zeros(len(self.cont), dtype=torch.float32)

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        return {
            "cont": self.cont[idx],
            "bin": self.bin[idx],
            "cat": self.cat[idx],
            "ref_risk": self.ref_risk[idx]
        }

def corrupt_features(cont: torch.Tensor, bin_x: torch.Tensor, cat_x: torch.Tensor, mask_prob: float = 0.15):
    """
    Applies random masking/corruption:
    - Continuous features masked to 0.0 (mean in standard scaling)
    - Binary features masked to 0
    - Categorical features masked to 0 (UNK token)
    """
    b, n_cont = cont.shape
    _, n_bin = bin_x.shape
    _, n_cat = cat_x.shape

    # Continuous mask
    mask_cont = (torch.rand_like(cont) < mask_prob)
    corrupted_cont = cont.clone()
    corrupted_cont[mask_cont] = 0.0

    # Binary mask
    mask_bin = (torch.rand(b, n_bin, device=bin_x.device) < mask_prob)
    corrupted_bin = bin_x.clone()
    corrupted_bin[mask_bin] = 0

    # Categorical mask (set to 0 = UNK)
    mask_cat = (torch.rand(b, n_cat, device=cat_x.device) < mask_prob)
    corrupted_cat = cat_x.clone()
    corrupted_cat[mask_cat] = 0

    return corrupted_cont, corrupted_bin, corrupted_cat

def train_representation_model(
    config_path: str = "config/risk_thresholds.yaml",
    preprocessor_path: str = "outputs/preprocessor.pkl",
    data_dir: str = "data",
    output_dir: str = "outputs"
):
    print("=" * 60)
    print("PHASE 5: SELF-SUPERVISED REPRESENTATION LEARNING")
    print("=" * 60)

    # Load configuration
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    token_dim = model_cfg.get("token_dim", 32)
    num_heads = model_cfg.get("num_heads", 4)
    num_layers = model_cfg.get("num_layers", 2)
    ff_dim = model_cfg.get("ff_dim", 64)
    dropout = model_cfg.get("dropout", 0.15)
    mask_prob = model_cfg.get("mask_prob", 0.15)
    lr = model_cfg.get("learning_rate", 0.001)
    batch_size = model_cfg.get("batch_size", 32)
    epochs = model_cfg.get("epochs", 100)
    patience = model_cfg.get("patience", 15)
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load preprocessor
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    # Load data splits with reference risk
    train_df = pd.read_csv(os.path.join(data_dir, "train_with_ref_risk.csv") if os.path.exists(os.path.join(data_dir, "train_with_ref_risk.csv")) else os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation_with_ref_risk.csv") if os.path.exists(os.path.join(data_dir, "validation_with_ref_risk.csv")) else os.path.join(data_dir, "validation.csv"))

    train_feats = preprocessor.transform(train_df)
    val_feats = preprocessor.transform(val_df)

    train_ref = train_df.get('reference_risk_score', pd.Series(np.zeros(len(train_df)))).values
    val_ref = val_df.get('reference_risk_score', pd.Series(np.zeros(len(val_df)))).values

    train_dataset = ClinicalPatientDataset(train_feats, train_ref)
    val_dataset = ClinicalPatientDataset(val_feats, val_ref)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize FT-Transformer
    num_cont = len(preprocessor.continuous_cols)
    bin_cards = [2] * len(preprocessor.binary_cols)
    cat_cards = [preprocessor.cat_cardinalities[col] for col in preprocessor.categorical_cols]

    model = FTTransformer(
        num_continuous=num_cont,
        binary_cardinalities=bin_cards,
        categorical_cardinalities=cat_cards,
        token_dim=token_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_cont_loss": [],
        "val_cont_loss": []
    }

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    # Stage 1: Self-Supervised Denoising Representation Pretraining
    print("\n--- STAGE 1: Self-Supervised Representation Pretraining ---")
    recon_epochs = 40
    best_recon_loss = float("inf")

    for epoch in range(1, recon_epochs + 1):
        model.train()
        train_recon_loss = 0.0

        for batch in train_loader:
            clean_cont = batch["cont"]
            clean_bin = batch["bin"]
            clean_cat = batch["cat"]

            corrupted_cont, corrupted_bin, corrupted_cat = corrupt_features(
                clean_cont, clean_bin, clean_cat, mask_prob=mask_prob
            )

            optimizer.zero_grad()
            out = model.forward_reconstruction(corrupted_cont, corrupted_bin, corrupted_cat)

            l_cont = mse_loss_fn(out["cont_recon"], clean_cont)
            l_bin = bce_loss_fn(out["bin_recon"], clean_bin.float())
            l_cat = sum(ce_loss_fn(out["cat_recon"][k], clean_cat[:, k]) for k in range(len(cat_cards)))
            l_recon = l_cont + 0.5 * l_bin + 0.5 * l_cat

            l_recon.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_recon_loss += l_recon.item() * len(clean_cont)

        train_recon_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_recon_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                clean_cont = batch["cont"]
                clean_bin = batch["bin"]
                clean_cat = batch["cat"]
                corrupted_cont, corrupted_bin, corrupted_cat = corrupt_features(
                    clean_cont, clean_bin, clean_cat, mask_prob=mask_prob
                )
                out = model.forward_reconstruction(corrupted_cont, corrupted_bin, corrupted_cat)
                l_cont = mse_loss_fn(out["cont_recon"], clean_cont)
                l_bin = bce_loss_fn(out["bin_recon"], clean_bin.float())
                l_cat = sum(ce_loss_fn(out["cat_recon"][k], clean_cat[:, k]) for k in range(len(cat_cards)))
                val_recon_loss += (l_cont + 0.5 * l_bin + 0.5 * l_cat).item() * len(clean_cont)
        val_recon_loss /= len(val_dataset)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Pretrain Epoch {epoch:02d}/{recon_epochs} | Recon Train Loss: {train_recon_loss:.4f} | Recon Val Loss: {val_recon_loss:.4f}")

    # Stage 2: Reference-Guided Neural Risk Head Training (Option B in MD Section 14)
    print("\n--- STAGE 2: Clinical Reference-Guided Risk Head Training ---")
    risk_optimizer = optim.AdamW([
        {"params": model.transformer.parameters(), "lr": lr * 0.2},
        {"params": model.norm.parameters(), "lr": lr * 0.2},
        {"params": model.risk_head.parameters(), "lr": lr * 1.5}
    ], weight_decay=1e-4)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_cont_loss": [],
        "val_cont_loss": []
    }

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        train_cont_loss_acc = 0.0

        for batch in train_loader:
            clean_cont = batch["cont"]
            clean_bin = batch["bin"]
            clean_cat = batch["cat"]
            ref_risk = batch["ref_risk"]

            corrupted_cont, corrupted_bin, corrupted_cat = corrupt_features(
                clean_cont, clean_bin, clean_cat, mask_prob=mask_prob
            )

            risk_optimizer.zero_grad()
            out = model.forward_reconstruction(corrupted_cont, clean_bin, clean_cat)

            l_cont = mse_loss_fn(out["cont_recon"], clean_cont)
            l_bin = bce_loss_fn(out["bin_recon"], clean_bin.float())
            l_cat = sum(ce_loss_fn(out["cat_recon"][k], clean_cat[:, k]) for k in range(len(cat_cards)))
            l_recon = l_cont + 0.5 * l_bin + 0.5 * l_cat

            # Risk head loss with amplified gradient to capture full dynamic range
            l_risk = mse_loss_fn(out["predicted_risk"], ref_risk)
            total_loss = 0.3 * l_recon + 2.0 * l_risk

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            risk_optimizer.step()

            train_loss_acc += total_loss.item() * len(clean_cont)
            train_cont_loss_acc += l_cont.item() * len(clean_cont)

        train_loss = train_loss_acc / len(train_dataset)
        train_cont_loss = train_cont_loss_acc / len(train_dataset)

        # Validation
        model.eval()
        val_loss_acc = 0.0
        val_cont_loss_acc = 0.0

        with torch.no_grad():
            for batch in val_loader:
                clean_cont = batch["cont"]
                clean_bin = batch["bin"]
                clean_cat = batch["cat"]
                ref_risk = batch["ref_risk"]

                out = model.forward_reconstruction(clean_cont, clean_bin, clean_cat)

                l_cont = mse_loss_fn(out["cont_recon"], clean_cont)
                l_bin = bce_loss_fn(out["bin_recon"], clean_bin.float())
                l_cat = sum(ce_loss_fn(out["cat_recon"][k], clean_cat[:, k]) for k in range(len(cat_cards)))
                l_recon = l_cont + 0.5 * l_bin + 0.5 * l_cat
                l_risk = mse_loss_fn(out["predicted_risk"], ref_risk)
                total_loss = 0.3 * l_recon + 2.0 * l_risk

                val_loss_acc += total_loss.item() * len(clean_cont)
                val_cont_loss_acc += l_cont.item() * len(clean_cont)

        val_loss = val_loss_acc / len(val_dataset)
        val_cont_loss = val_cont_loss_acc / len(val_dataset)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cont_loss"].append(train_cont_loss)
        history["val_cont_loss"].append(val_cont_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Alignment Epoch {epoch:03d}/{epochs} | Total Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "token_dim": token_dim,
                "num_heads": num_heads,
                "num_layers": num_layers
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss:.4f}")
                break

    # Save training history JSON
    history_path = os.path.join(output_dir, "tables", "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best model checkpoint saved to: {best_checkpoint_path}")
    print(f"Training history saved to: {history_path}")
    print("=" * 60)
    return best_checkpoint_path

if __name__ == "__main__":
    train_representation_model()
