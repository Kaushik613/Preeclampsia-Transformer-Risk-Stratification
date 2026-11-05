"""
Phase 4: FT-Transformer Architecture for Tabular Clinical Data
Implements Feature Tokenizer + Transformer (FT-Transformer) with:
  - Numerical Feature Tokenizer (learned linear projection per scalar)
  - Discrete Feature Tokenizers (learned embedding per binary / categorical feature)
  - Learnable [CLS] patient token
  - Multi-Head Self-Attention Blocks with Residual Connections & Pre-LayerNorm
  - Self-Supervised Reconstruction Heads
  - Clinical Neural Risk Head (predicting continuous Research Risk Index)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NumericalFeatureTokenizer(nn.Module):
    """Tokenizes continuous features via dedicated per-feature weight and bias vectors."""
    def __init__(self, num_features: int, token_dim: int):
        super().__init__()
        self.num_features = num_features
        self.token_dim = token_dim
        # Shape: (num_features, token_dim)
        self.weights = nn.Parameter(torch.empty(num_features, token_dim))
        self.biases = nn.Parameter(torch.empty(num_features, token_dim))
        self.reset_parameters()

    def reset_parameters(self):
        d = self.token_dim
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.zeros_(self.biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_features)
        # Output: (batch_size, num_features, token_dim)
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    """Tokenizes discrete categorical and binary features using embedding layers."""
    def __init__(self, cardinalities: list, token_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, token_dim) for card in cardinalities
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_discrete_features)
        tokens = [emb(x[:, i]).unsqueeze(1) for i, emb in enumerate(self.embeddings)]
        return torch.cat(tokens, dim=1)  # (batch_size, num_discrete_features, token_dim)


class FTTransformer(nn.Module):
    def __init__(
        self,
        num_continuous: int = 17,
        binary_cardinalities: list = None,      # e.g. [2, 2, 2, 2, 2]
        categorical_cardinalities: list = None, # e.g. [4, 4, 4] (including UNK)
        token_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.15
    ):
        super().__init__()
        if binary_cardinalities is None:
            binary_cardinalities = [2] * 5
        if categorical_cardinalities is None:
            categorical_cardinalities = [4, 4, 4]

        self.token_dim = token_dim
        self.num_continuous = num_continuous
        self.num_binary = len(binary_cardinalities)
        self.num_categorical = len(categorical_cardinalities)
        self.total_features = num_continuous + self.num_binary + self.num_categorical

        # 1. Feature Tokenizers
        self.num_tokenizer = NumericalFeatureTokenizer(num_continuous, token_dim)
        self.bin_tokenizer = CategoricalFeatureTokenizer(binary_cardinalities, token_dim)
        self.cat_tokenizer = CategoricalFeatureTokenizer(categorical_cardinalities, token_dim)

        # 2. Learnable [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)

        # 4. Self-Supervised Denoising Reconstruction Heads
        # Continuous: predict continuous scalar from each continuous token
        self.cont_recon_head = nn.ModuleList([
            nn.Linear(token_dim, 1) for _ in range(num_continuous)
        ])

        # Binary: predict logit from each binary token
        self.bin_recon_head = nn.ModuleList([
            nn.Linear(token_dim, 1) for _ in range(self.num_binary)
        ])

        # Categorical: predict logits over classes from each categorical token
        self.cat_recon_head = nn.ModuleList([
            nn.Linear(token_dim, card) for card in categorical_cardinalities
        ])

        # 5. Neural Clinical Risk Head (Option B in MD)
        # Takes 32-D patient embedding [CLS] and estimates continuous Research Risk Index
        self.risk_head = nn.Sequential(
            nn.Linear(token_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Bound risk score in [0, 1]
        )

    def forward_tokens(
        self,
        cont_x: torch.Tensor,
        bin_x: torch.Tensor,
        cat_x: torch.Tensor
    ) -> torch.Tensor:
        """Construct sequence of tokens: [CLS, cont_1..17, bin_1..5, cat_1..3]"""
        b = cont_x.shape[0]

        # Tokenize features
        cont_tokens = self.num_tokenizer(cont_x)  # (B, 17, 32)
        bin_tokens = self.bin_tokenizer(bin_x)    # (B, 5, 32)
        cat_tokens = self.cat_tokenizer(cat_x)    # (B, 3, 32)

        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, 32)

        all_tokens = torch.cat([cls_tokens, cont_tokens, bin_tokens, cat_tokens], dim=1)  # (B, 26, 32)
        encoded = self.transformer(all_tokens)
        encoded = self.norm(encoded)
        return encoded

    def get_patient_embedding(
        self,
        cont_x: torch.Tensor,
        bin_x: torch.Tensor,
        cat_x: torch.Tensor
    ) -> torch.Tensor:
        """Returns 32-D patient embedding extracted from the [CLS] token."""
        encoded = self.forward_tokens(cont_x, bin_x, cat_x)
        return encoded[:, 0, :]  # [CLS] token embedding

    def forward_reconstruction(
        self,
        cont_x: torch.Tensor,
        bin_x: torch.Tensor,
        cat_x: torch.Tensor
    ) -> dict:
        """Reconstructs features from their respective output tokens."""
        encoded = self.forward_tokens(cont_x, bin_x, cat_x)
        
        # Token layout:
        # Index 0: [CLS]
        # Index 1 to 17: continuous tokens
        # Index 18 to 22: binary tokens
        # Index 23 to 25: categorical tokens

        # Reconstruct continuous
        cont_preds = []
        for i in range(self.num_continuous):
            tok = encoded[:, 1 + i, :]
            pred = self.cont_recon_head[i](tok)
            cont_preds.append(pred)
        cont_recon = torch.cat(cont_preds, dim=1)  # (B, 17)

        # Reconstruct binary
        bin_preds = []
        offset_bin = 1 + self.num_continuous
        for j in range(self.num_binary):
            tok = encoded[:, offset_bin + j, :]
            pred = self.bin_recon_head[j](tok)
            bin_preds.append(pred)
        bin_recon = torch.cat(bin_preds, dim=1)  # (B, 5) logits

        # Reconstruct categorical
        cat_preds = []
        offset_cat = offset_bin + self.num_binary
        for k in range(self.num_categorical):
            tok = encoded[:, offset_cat + k, :]
            pred = self.cat_recon_head[k](tok)
            cat_preds.append(pred)  # list of (B, card_k) logits

        patient_embedding = encoded[:, 0, :]
        predicted_risk = self.risk_head(patient_embedding).squeeze(-1)

        return {
            "cont_recon": cont_recon,
            "bin_recon": bin_recon,
            "cat_recon": cat_preds,
            "patient_embedding": patient_embedding,
            "predicted_risk": predicted_risk
        }
