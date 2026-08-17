"""
SequenceFeature — Card-level transaction sequence embeddings via LSTM autoencoder.

Architecture:
    card1 → sort by TransactionDT → build sliding windows → LSTM encode
    → fixed-dim embedding → new columns seq_emb_{0..dim-1}

Anti-leakage design:
    1. The LSTM autoencoder is trained on TRAINING data only (no target).
    2. Card-level embeddings are pre-computed during fit() from training
       sequences only, stored as a lookup map {card1: embedding_vector}.
    3. During transform(), each row receives the embedding of its card1
       via map lookup.  Unseen cards fall back to a zero vector — no
       information from the validation / test set ever influences
       training statistics.
    4. No target column is read at any point (unsupervised reconstruction).

Integrates as a standard FeatureBase subclass — auto-discovered by
FeatureRegistry, serializable via joblib, and produces dense float
features that LightGBM / XGBoost / CatBoost can consume directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import FeatureBase

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _LSTMEncoder(nn.Module):
    """LSTM sequence encoder → fixed-dim embedding.

    Input shape: (batch, seq_len, input_dim) with padding mask.
    Output shape: (batch, embed_dim) — last non-pad hidden state.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, self.hidden_dim)
        last = out.gather(1, idx).squeeze(1)

        emb = self.proj(last)
        emb = self.layer_norm(emb)
        return emb


class _LSTMAutoEncoder(nn.Module):
    """LSTM autoencoder — encodes sequence → embeds → reconstructs.

    Used during fit() to learn a compact, smooth embedding space
    via unsupervised reconstruction.  Only the encoder part is kept
    after training; the decoder is discarded.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoder = _LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.recon_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x, lengths)

    def decode(
        self, emb: "torch.Tensor", lengths: "torch.Tensor"
    ) -> "torch.Tensor":
        max_len = int(lengths.max().item())
        emb_seq = emb.unsqueeze(1).expand(-1, max_len, -1)
        out, _ = self.decoder(emb_seq)
        recon = self.recon_proj(out)
        return recon

    def forward(
        self, x: "torch.Tensor", lengths: "torch.Tensor"
    ) -> Dict[str, "torch.Tensor"]:
        emb = self.encode(x, lengths)
        recon = self.decode(emb, lengths)
        return {"embedding": emb, "reconstruction": recon}



class _CNNEncoder(nn.Module):
    """1-D dilated CNN encoder → fixed-dim embedding."""

    def __init__(self, input_dim, hidden_dim=64, embed_dim=32, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = hidden_dim

        self.conv_net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, lengths):
        x_t = x.transpose(1, 2)
        feat = self.conv_net(x_t)
        feat_t = feat.transpose(1, 2)
        mask = (torch.arange(feat_t.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)
        masked = feat_t * mask
        summed = masked.sum(dim=1)
        lengths_f = lengths.float().clamp(min=1).unsqueeze(-1)
        mean_pooled = summed / lengths_f
        emb = self.proj(mean_pooled)
        emb = self.layer_norm(emb)
        return emb


class _CNNAutoEncoder(nn.Module):
    """CNN autoencoder — encodes sequence → embeds → reconstructs."""

    def __init__(self, input_dim, hidden_dim=64, embed_dim=32, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = _CNNEncoder(input_dim, hidden_dim, embed_dim, num_layers, dropout)
        layers = []
        in_ch = embed_dim
        for i in range(num_layers):
            dilation = 2 ** (num_layers - 1 - i)
            layers.append(nn.ConvTranspose1d(in_ch, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = hidden_dim
        layers.append(nn.Conv1d(hidden_dim, input_dim, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x, lengths):
        return self.encoder(x, lengths)

    def decode(self, emb, lengths):
        max_len = int(lengths.max().item())
        emb_seq = emb.unsqueeze(1).expand(-1, max_len, -1)
        emb_t = emb_seq.transpose(1, 2)
        out = self.decoder(emb_t)
        return out.transpose(1, 2)

    def forward(self, x, lengths):
        emb = self.encode(x, lengths)
        recon = self.decode(emb, lengths)
        return {"embedding": emb, "reconstruction": recon}





def _prepare_sequence_data(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    feature_cols: List[str],
    max_seq_len: int,
) -> List[Dict[str, Any]]:
    """Group by *group_col*, sort by *time_col*, build per-card sequences.

    Returns list of dicts: {card, features: np.array(seq_len, n_feat), length: int}
    """
    rows: List[Dict[str, Any]] = []
    grouped = df.groupby(group_col, sort=False)

    for card, grp in grouped:
        if len(grp) < 1:
            continue
        grp_sorted = grp.sort_values(time_col)
        feats = grp_sorted[feature_cols].fillna(0).values.astype(np.float32)

        if len(feats) > max_seq_len:
            feats = feats[-max_seq_len:]

        rows.append(
            {
                "card": card,
                "features": feats,
                "length": len(feats),
            }
        )
    return rows


def _build_batches(
    sequences: List[Dict[str, Any]],
    batch_size: int,
    input_dim: int,
) -> List[Dict[str, Any]]:
    """Collate sequences into padded batches for PyTorch."""
    batches: List[Dict[str, Any]] = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i : i + batch_size]
        lengths = np.array([s["length"] for s in chunk], dtype=np.int64)
        max_len = int(lengths.max())
        padded = np.zeros((len(chunk), max_len, input_dim), dtype=np.float32)
        for j, s in enumerate(chunk):
            seq_len = s["length"]
            padded[j, :seq_len] = s["features"][:seq_len]
        batches.append(
            {
                "cards": [s["card"] for s in chunk],
                "padded": padded,
                "lengths": lengths,
            }
        )
    return batches


class SequenceFeature(FeatureBase):
    """Card-level sequence embeddings via LSTM autoencoder.

    Groups transactions by ``card1``, sorts by ``TransactionDT``,
    encodes each card's behavioral sequence into a fixed-size
    embedding vector, and exposes the vector components as new
    numeric features (``seq_emb_0`` … ``seq_emb_{dim-1}``).

    **Stateful** — learns the LSTM weights and stores a
    ``{card1: embedding_vector}`` lookup map during ``fit()``.

    Parameters
    ----------
    group_col : str
        Column to group sequences by. Default ``"card1"``.
    time_col : str
        Column to sort sequences by. Default ``"TransactionDT"``.
    feature_cols : list of str
        Numerical columns used as sequence features.
        Default: ``["TransactionAmt", "dist1", "dist2"]``.
    max_seq_len : int
        Maximum sequence length per card (truncated from the tail).
    embed_dim : int
        Dimensionality of the output embedding vector.
    hidden_dim : int
        LSTM hidden size.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate (applied only when ``num_layers > 1``).
    epochs : int
        Number of training epochs for the autoencoder.
    batch_size : int
        Mini-batch size for autoencoder training + inference.
    lr : float
        Adam learning rate.
    random_state : int
        Seed for reproducibility.
    """

    _DEFAULT_FEATURE_COLS: List[str] = [
        "TransactionAmt",
        "dist1",
        "dist2",
    ]

    def __init__(
        self,
        name: str = "SequenceFeature",
        group_col: str = "card1",
        time_col: str = "TransactionDT",
        feature_cols: Optional[List[str]] = None,
        max_seq_len: int = 20,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        random_state: int = 42,
        device: str = "auto",
        model_type: str = "lstm",
    ) -> None:
        super().__init__(name=name)
        self._group_col = group_col
        self._time_col = time_col
        self._feature_cols = list(feature_cols or self._DEFAULT_FEATURE_COLS)
        self._max_seq_len = max_seq_len
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._random_state = random_state
        self._device_spec = device
        self._model_type = model_type

        self._embed_map: Dict[Any, np.ndarray] = {}
        self._anomaly_map: Dict[Any, float] = {}
        self._input_dim: int = 0
        self._global_mean_embedding: Optional[np.ndarray] = None
        self._global_mean_anomaly: float = 0.0

    def _resolve_device(self) -> "torch.device":
        if not _TORCH_AVAILABLE:
            return None
        if self._device_spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device_spec)

    @property
    def is_stateful(self) -> bool:
        return True

    @property
    def _col_suffix(self) -> str:
        if self.name == self.__class__.__name__:
            return ""
        return f"_{self.name}"

    def _get_state(self) -> Dict[str, Any]:
        return {
            "_group_col": self._group_col,
            "_time_col": self._time_col,
            "_feature_cols": self._feature_cols,
            "_max_seq_len": self._max_seq_len,
            "_embed_dim": self._embed_dim,
            "_hidden_dim": self._hidden_dim,
            "_num_layers": self._num_layers,
            "_dropout": self._dropout,
            "_embed_map": self._embed_map,
            "_anomaly_map": self._anomaly_map,
            "_input_dim": self._input_dim,
            "_global_mean_embedding": self._global_mean_embedding,
            "_global_mean_anomaly": self._global_mean_anomaly,
            "_model_type": self._model_type,
            "_device_spec": self._device_spec,
        }

    def fit(self, df: pd.DataFrame) -> "SequenceFeature":
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SequenceFeature. "
                "Install it via: pip install torch"
            )

        available = [c for c in self._feature_cols if c in df.columns]
        if not available:
            raise ValueError(
                f"{self.name}: none of feature_cols {self._feature_cols} "
                f"found in DataFrame columns."
            )
        self._feature_cols = available
        self._input_dim = len(available)

        device = self._resolve_device()
        torch.manual_seed(self._random_state)
        np.random.seed(self._random_state)

        sequences = _prepare_sequence_data(
            df=df,
            group_col=self._group_col,
            time_col=self._time_col,
            feature_cols=self._feature_cols,
            max_seq_len=self._max_seq_len,
        )
        if not sequences:
            self._fitted = True
            return self

        if self._model_type == "lstm":
            model = _LSTMAutoEncoder(
                input_dim=self._input_dim,
                hidden_dim=self._hidden_dim,
                embed_dim=self._embed_dim,
                num_layers=self._num_layers,
                dropout=self._dropout,
            ).to(device)
        elif self._model_type == "cnn":
            model = _CNNAutoEncoder(
                input_dim=self._input_dim,
                hidden_dim=self._hidden_dim,
                embed_dim=self._embed_dim,
                num_layers=self._num_layers,
                dropout=self._dropout,
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        batches = _build_batches(
            sequences, self._batch_size, self._input_dim
        )

        model.train()
        for _ in range(self._epochs):
            total_loss = 0.0
            n_samples = 0
            for batch in batches:
                x = torch.from_numpy(batch["padded"]).to(device)
                lengths = torch.from_numpy(batch["lengths"]).to(device)

                optimizer.zero_grad()
                out = model(x, lengths)
                recon = out["reconstruction"]

                max_len = int(lengths.max().item())
                mask = (
                    torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                ).float().unsqueeze(-1) # 从 (batch, max_len) 变成 (batch, max_len, 1)
                loss = (loss_fn(recon * mask, x * mask) * mask.sum() / self._input_dim).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * len(batch["cards"])
                n_samples += len(batch["cards"])

            if n_samples > 0:
                _ = total_loss / n_samples

        model.eval()
        self._embed_map = {}
        self._anomaly_map = {}
        all_embs: List[np.ndarray] = []
        all_anomalies: List[float] = []

        with torch.no_grad():
            for batch in batches:
                x = torch.from_numpy(batch["padded"]).to(device)
                lengths = torch.from_numpy(batch["lengths"]).to(device)

                emb = model.encode(x, lengths).cpu().numpy()

                recon = model(x, lengths)["reconstruction"]
                max_len = int(lengths.max().item())
                mask = (
                    torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                ).float().unsqueeze(-1)
                sq_err = ((recon - x) * mask) ** 2
                mse_per_sample = sq_err.sum(dim=(1, 2)) / (lengths.float() * self._input_dim)
                anomaly_scores = mse_per_sample.cpu().numpy()

                for card, e, a in zip(batch["cards"], emb, anomaly_scores):
                    self._embed_map[card] = e.astype(np.float32)
                    self._anomaly_map[card] = float(a)
                    all_embs.append(e.astype(np.float32))
                    all_anomalies.append(float(a))

        if all_embs:
            self._global_mean_embedding = np.mean(
                np.stack(all_embs, axis=0), axis=0
            ).astype(np.float32)
        if all_anomalies:
            self._global_mean_anomaly = float(np.mean(all_anomalies))

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()
        suffix = self._col_suffix
        n_embed = self._embed_dim

        default_vec = (
            self._global_mean_embedding
            if self._global_mean_embedding is not None
            else np.zeros(n_embed, dtype=np.float32)
        )

        embeddings = np.tile(default_vec, (len(df), 1)).astype(np.float32)

        if self._group_col in df.columns:
            cards = df[self._group_col].values
            for i, card in enumerate(cards):
                emb = self._embed_map.get(card)
                if emb is not None:
                    embeddings[i] = emb

        for k in range(n_embed):
            col_name = f"seq_emb_{k}{suffix}"
            df[col_name] = embeddings[:, k].astype(np.float32)

        anomaly_default = self._global_mean_anomaly
        anomalies = np.full(len(df), anomaly_default, dtype=np.float32)
        if self._group_col in df.columns:
            cards = df[self._group_col].values
            for i, card in enumerate(cards):
                a = self._anomaly_map.get(card)
                if a is not None:
                    anomalies[i] = a
        df[f"seq_anomaly_score{suffix}"] = anomalies

        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "SequenceFeature",
            "layer": "business-domain",
            "is_stateful": True,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "SequenceFeature",
                    "description": "Instance name.",
                },
                {
                    "name": "group_col",
                    "type": "str",
                    "default": "card1",
                    "description": "Column to group sequences by.",
                },
                {
                    "name": "time_col",
                    "type": "str",
                    "default": "TransactionDT",
                    "description": "Column to sort sequences by.",
                },
                {
                    "name": "feature_cols",
                    "type": "list[str]",
                    "default": '["TransactionAmt", "dist1", "dist2"]',
                    "description": "Numerical columns forming the sequence input.",
                },
                {
                    "name": "max_seq_len",
                    "type": "int",
                    "default": 20,
                    "description": "Maximum sequence length per card (tail-truncated).",
                },
                {
                    "name": "embed_dim",
                    "type": "int",
                    "default": 32,
                    "description": "Dimensionality of output embedding.",
                },
                {
                    "name": "hidden_dim",
                    "type": "int",
                    "default": 64,
                    "description": "LSTM hidden size.",
                },
                {
                    "name": "epochs",
                    "type": "int",
                    "default": 10,
                    "description": "Training epochs for the autoencoder.",
                },
            ],
            "example": """# Basic (default params):
- SequenceFeature

# Customized:
- SequenceFeature:
    embed_dim: 64
    max_seq_len: 30
    epochs: 20""",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        suffix = self._col_suffix
        return {
            "feature_names": [
                f"seq_emb_{k}{suffix}" for k in range(self._embed_dim)
            ] + [f"seq_anomaly_score{suffix}"],
            "physical_meaning": "LSTM autoencoder embeddings of per-card transaction sequence",
            "unit": "float32",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_sequence_basic():
        if not _TORCH_AVAILABLE:
            print("PyTorch not installed — skipping smoke test.")
            return

        df = pd.DataFrame(
            {
                "card1": [1, 1, 1, 2, 2, 3],
                "TransactionDT": [1000, 2000, 3000, 1500, 2500, 1800],
                "TransactionAmt": [10.0, 20.0, 30.0, 5.0, 15.0, 8.0],
                "dist1": [1.0, 2.0, 3.0, 0.5, 1.5, 1.0],
                "dist2": [0.1, 0.2, 0.3, 0.05, 0.15, 0.1],
            }
        )

        for mt in ["lstm", "cnn"]:
            feat = SequenceFeature(embed_dim=4, epochs=3, batch_size=4, model_type=mt)
            feat.fit(df)
            result = feat.transform(df)

            for k in range(4):
                col = f"seq_emb_{k}"
                assert col in result.columns, f"Missing column: {col}"
                assert result[col].dtype == np.float32

            assert len(feat._embed_map) == 3
            print(f"  [{mt}] basic OK")

            assert f"seq_anomaly_score" in result.columns
            assert result["seq_anomaly_score"].dtype == np.float32
            assert not result["seq_anomaly_score"].isna().any()


    def test_sequence_unseen_card():
        if not _TORCH_AVAILABLE:
            print("PyTorch not installed — skipping smoke test.")
            return

        train_df = pd.DataFrame(
            {
                "card1": [1, 1, 2],
                "TransactionDT": [1000, 2000, 1500],
                "TransactionAmt": [10.0, 20.0, 5.0],
                "dist1": [1.0, 2.0, 0.5],
                "dist2": [0.1, 0.2, 0.05],
            }
        )

        for mt in ["lstm", "cnn"]:
            feat = SequenceFeature(embed_dim=4, epochs=3, model_type=mt)
            feat.fit(train_df)

        test_df = pd.DataFrame(
            {
                "card1": [1, 999, 2],
                "TransactionDT": [1000, 2000, 1500],
                "TransactionAmt": [10.0, 20.0, 5.0],
                "dist1": [1.0, 2.0, 0.5],
                "dist2": [0.1, 0.2, 0.05],
            }
        )
        result = feat.transform(test_df)

        unseen_row = result[result["card1"] == 999].iloc[0]
        for k in range(4):
            col = f"seq_emb_{k}"
            assert col in result.columns
            assert pd.notna(unseen_row[col])
            assert f"seq_anomaly_score" in result.columns
            assert pd.notna(unseen_row["seq_anomaly_score"])

            
    test_sequence_basic()
    test_sequence_unseen_card()
    print("All SequenceFeature tests passed!")