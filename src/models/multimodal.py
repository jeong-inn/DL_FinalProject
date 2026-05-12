"""
Multimodal SLI classifier.

Three input streams:
  ① Text     — DistilBERT [CLS] embedding        (768-D)
  ② Acoustic — wav2vec2 mean-pooled embedding     (768-D)
  ③ Linguistic — handcrafted features             (12-D)
  ④ Prosodic — handcrafted acoustic features      (13-D)

Fusion: concat → projection → ReLU → Dropout → classifier

          text_emb (768)  ──┐
       acoustic_emb (768) ──┤
          ling_feat  (12) ──┼─ concat → Linear(hidden) → ReLU → Dropout → Linear(2)
        prosodic_feat(13) ──┘
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.bert_classifier import DistilBertClassifier, _get_device
from src.features.acoustic import WAV2VEC2_DIM, PROSODIC_FEATURE_NAMES
from src.features.linguistic import FEATURE_NAMES as LING_FEATURE_NAMES


TEXT_DIM = 768
LING_DIM = len(LING_FEATURE_NAMES)     # 12
PROSODIC_DIM = len(PROSODIC_FEATURE_NAMES)  # 13
FUSED_DIM = TEXT_DIM + WAV2VEC2_DIM + LING_DIM + PROSODIC_DIM


class _FusionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalClassifier:
    """
    Training flow:
      1. Fine-tune DistilBERT (or reuse existing checkpoint).
      2. Extract DistilBERT [CLS] embeddings + wav2vec2 embeddings.
      3. Concatenate with linguistic & prosodic features.
      4. Train fusion MLP.
    """

    def __init__(
        self,
        bert_epochs: int = 10,
        mlp_epochs: int = 80,
        batch_size: int = 8,
        lr_bert: float = 2e-5,
        lr_mlp: float = 5e-4,
        patience: int = 5,
    ):
        self.bert_epochs = bert_epochs
        self.mlp_epochs = mlp_epochs
        self.batch_size = batch_size
        self.device = _get_device()

        self.bert = DistilBertClassifier(
            epochs=bert_epochs, batch_size=batch_size, lr=lr_bert, patience=patience
        )
        self.mlp: _FusionMLP | None = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}
        self.lr_mlp = lr_mlp
        self.patience = patience

    # ── public interface ────────────────────────────────────────────────────

    def fit(
        self,
        train_texts: list[str],
        train_wav2vec: np.ndarray,   # (N, 768)
        train_ling: np.ndarray,       # (N, 12)
        train_prosodic: np.ndarray,   # (N, 13)
        train_labels: list[str],
        val_texts: list[str] | None = None,
        val_wav2vec: np.ndarray | None = None,
        val_ling: np.ndarray | None = None,
        val_prosodic: np.ndarray | None = None,
        val_labels: list[str] | None = None,
        bert_checkpoint: Path | None = None,
    ) -> "MultimodalClassifier":
        unique = sorted(set(train_labels))
        self.label2id = {l: i for i, l in enumerate(unique)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.bert.label2id = self.label2id
        self.bert.id2label = self.id2label

        # step 1: fine-tune or load BERT
        if bert_checkpoint and bert_checkpoint.exists():
            print(f"Loading BERT checkpoint from {bert_checkpoint}")
            self.bert.load(bert_checkpoint)
        else:
            print("Fine-tuning DistilBERT …")
            self.bert.fit(
                train_texts, train_labels,
                val_texts=val_texts, val_labels=val_labels,
            )

        # step 2: extract text embeddings
        print("Extracting DistilBERT [CLS] embeddings …")
        train_cls = self.bert.get_cls_embeddings(train_texts)
        train_X = self._concat(train_cls, train_wav2vec, train_ling, train_prosodic)
        train_y = np.array([self.label2id[l] for l in train_labels])

        val_X = val_y = None
        if val_texts is not None:
            val_cls = self.bert.get_cls_embeddings(val_texts)
            val_X = self._concat(val_cls, val_wav2vec, val_ling, val_prosodic)
            val_y = np.array([self.label2id[l] for l in val_labels])

        # step 3: train fusion MLP
        print("Training fusion MLP …")
        self._train_mlp(train_X, train_y, val_X, val_y)
        return self

    def predict(
        self,
        texts: list[str],
        wav2vec: np.ndarray,
        ling: np.ndarray,
        prosodic: np.ndarray,
    ) -> np.ndarray:
        ids = self._predict_ids(texts, wav2vec, ling, prosodic)
        return np.array([self.id2label[i] for i in ids])

    def predict_proba(
        self,
        texts: list[str],
        wav2vec: np.ndarray,
        ling: np.ndarray,
        prosodic: np.ndarray,
    ) -> np.ndarray:
        self.mlp.eval()
        cls = self.bert.get_cls_embeddings(texts)
        X = torch.tensor(self._concat(cls, wav2vec, ling, prosodic)).to(self.device)
        sli_idx = self.label2id["SLI"]
        with torch.no_grad():
            probs = torch.softmax(self.mlp(X), dim=-1)
        return probs[:, sli_idx].cpu().numpy()

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _concat(*arrays: np.ndarray) -> np.ndarray:
        return np.hstack([a.astype(np.float32) for a in arrays])

    def _predict_ids(self, texts, wav2vec, ling, prosodic) -> np.ndarray:
        self.mlp.eval()
        cls = self.bert.get_cls_embeddings(texts)
        X = torch.tensor(self._concat(cls, wav2vec, ling, prosodic)).to(self.device)
        with torch.no_grad():
            return self.mlp(X).argmax(dim=-1).cpu().numpy()

    def _train_mlp(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> None:
        counts = np.bincount(train_y)
        weights = torch.tensor(len(train_y) / (len(counts) * counts), dtype=torch.float).to(self.device)

        self.mlp = _FusionMLP(in_dim=train_X.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=self.lr_mlp, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.mlp_epochs)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        X_t = torch.tensor(train_X).to(self.device)
        y_t = torch.tensor(train_y, dtype=torch.long).to(self.device)
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        best_val_f1, patience_count, best_state = -1.0, 0, None

        for epoch in range(1, self.mlp_epochs + 1):
            self.mlp.train()
            for xb, yb in dl:
                optimizer.zero_grad()
                loss_fn(self.mlp(xb), yb).backward()
                optimizer.step()
            scheduler.step()

            if val_X is not None:
                from sklearn.metrics import f1_score
                preds = self._infer_mlp(val_X)
                val_f1 = f1_score(val_y, preds, average='macro')
                if epoch % 20 == 0:
                    print(f"  epoch {epoch:3d}  val_macro_f1={val_f1:.4f}")
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_count = 0
                    best_state = {k: v.clone() for k, v in self.mlp.state_dict().items()}
                else:
                    patience_count += 1
                    if patience_count >= self.patience * 8:
                        print("  Early stopping fusion MLP.")
                        break

        if best_state:
            self.mlp.load_state_dict(best_state)

    def _infer_mlp(self, X: np.ndarray) -> np.ndarray:
        self.mlp.eval()
        with torch.no_grad():
            return self.mlp(torch.tensor(X).to(self.device)).argmax(dim=-1).cpu().numpy()
