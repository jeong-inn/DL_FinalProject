"""
Hybrid classifier: DistilBERT [CLS] embeddings + linguistic features → MLP.

Architecture:
  [CLS] (768) ──┐
                 ├─ concat (768 + n_ling) → Linear → ReLU → Dropout → Linear(2)
  ling_feats ───┘
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.bert_classifier import DistilBertClassifier, _get_device


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridClassifier:
    """
    Step 1 – fine-tune DistilBERT (or load a pre-trained checkpoint).
    Step 2 – freeze BERT, train a lightweight MLP on top of
             [CLS] embeddings + linguistic features.
    """

    def __init__(
        self,
        bert_epochs: int = 10,
        mlp_epochs: int = 50,
        batch_size: int = 8,
        lr_bert: float = 2e-5,
        lr_mlp: float = 1e-3,
        patience: int = 3,
    ):
        self.bert_epochs = bert_epochs
        self.mlp_epochs = mlp_epochs
        self.batch_size = batch_size
        self.lr_bert = lr_bert
        self.lr_mlp = lr_mlp
        self.patience = patience
        self.device = _get_device()

        self.bert = DistilBertClassifier(
            epochs=bert_epochs, batch_size=batch_size, lr=lr_bert, patience=patience
        )
        # Embedding extraction must run on CPU — MPS→CPU transfer hangs on Rosetta
        self.bert.device = torch.device("cpu")
        self.mlp: _MLP | None = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}

    def fit(
        self,
        train_texts: list[str],
        train_ling: np.ndarray,
        train_labels: list[str],
        val_texts: list[str] | None = None,
        val_ling: np.ndarray | None = None,
        val_labels: list[str] | None = None,
        bert_checkpoint: Path | None = None,
    ) -> "HybridClassifier":
        unique = sorted(set(train_labels))
        self.label2id = {l: i for i, l in enumerate(unique)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.bert.label2id = self.label2id
        self.bert.id2label = self.id2label

        if bert_checkpoint and bert_checkpoint.exists():
            print(f"Loading BERT from {bert_checkpoint}")
            self.bert.load(bert_checkpoint)
        else:
            print("Fine-tuning DistilBERT …")
            self.bert.fit(train_texts, train_labels, val_texts, val_labels)

        print("Extracting [CLS] embeddings …")
        train_cls = self.bert.get_cls_embeddings(train_texts)
        train_X = np.hstack([train_cls, train_ling]).astype(np.float32)
        train_y = np.array([self.label2id[l] for l in train_labels])

        # class weights
        counts = np.bincount(train_y)
        weights = torch.tensor(len(train_y) / (len(counts) * counts), dtype=torch.float).to(self.device)

        self.mlp = _MLP(in_dim=train_X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr_mlp)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        X_t = torch.tensor(train_X).to(self.device)
        y_t = torch.tensor(train_y, dtype=torch.long).to(self.device)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        best_val_f1 = -1.0
        patience_count = 0
        best_state = None

        for epoch in range(1, self.mlp_epochs + 1):
            self.mlp.train()
            for xb, yb in dl:
                optimizer.zero_grad()
                loss = loss_fn(self.mlp(xb), yb)
                loss.backward()
                optimizer.step()

            if val_texts is not None and val_ling is not None and val_labels is not None:
                from sklearn.metrics import f1_score
                val_preds = self._predict_ids(val_texts, val_ling)
                val_ids = [self.label2id[l] for l in val_labels]
                val_f1 = f1_score(val_ids, val_preds, average="macro")
                if epoch % 10 == 0:
                    print(f"MLP epoch {epoch}  val_macro_f1={val_f1:.4f}")
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_count = 0
                    best_state = {k: v.clone() for k, v in self.mlp.state_dict().items()}
                else:
                    patience_count += 1
                    if patience_count >= self.patience * 5:
                        print("Early stopping MLP.")
                        break

        if best_state:
            self.mlp.load_state_dict(best_state)

        return self

    def _predict_ids(self, texts: list[str], ling: np.ndarray) -> np.ndarray:
        self.mlp.eval()
        cls = self.bert.get_cls_embeddings(texts)
        X = torch.tensor(np.hstack([cls, ling]).astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.mlp(X)
        return logits.argmax(dim=-1).cpu().numpy()

    def predict(self, texts: list[str], ling: np.ndarray) -> np.ndarray:
        ids = self._predict_ids(texts, ling)
        return np.array([self.id2label[i] for i in ids])

    def predict_proba(self, texts: list[str], ling: np.ndarray) -> np.ndarray:
        self.mlp.eval()
        cls = self.bert.get_cls_embeddings(texts)
        X = torch.tensor(np.hstack([cls, ling]).astype(np.float32)).to(self.device)
        sli_idx = self.label2id["SLI"]
        with torch.no_grad():
            probs = torch.softmax(self.mlp(X), dim=-1)
        return probs[:, sli_idx].cpu().numpy()
