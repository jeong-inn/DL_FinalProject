from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, get_cosine_schedule_with_warmup

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512


class SpeechDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]


def _get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DistilBertClassifier:
    """Fine-tuned DistilBERT for SLI/TD binary classification."""

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        patience: int = 3,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.patience = patience
        self.device = _get_device()
        print(f"Using device: {self.device}")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        self.model: DistilBertForSequenceClassification | None = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}

    def _encode_labels(self, labels: list[str]) -> list[int]:
        if not self.label2id:
            unique = sorted(set(labels))
            self.label2id = {l: i for i, l in enumerate(unique)}
            self.id2label = {i: l for l, i in self.label2id.items()}
        return [self.label2id[l] for l in labels]

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str] | None = None,
        val_labels: list[str] | None = None,
        save_dir: Path | None = None,
    ) -> "DistilBertClassifier":
        train_ids = self._encode_labels(train_labels)

        # class weights to handle SLI/TD imbalance
        counts = np.bincount(train_ids)
        weights = torch.tensor(len(train_ids) / (len(counts) * counts), dtype=torch.float).to(self.device)

        self.model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(self.device)

        train_ds = SpeechDataset(train_texts, train_ids, self.tokenizer)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_dl) * self.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        best_val_f1 = -1.0
        patience_count = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch, batch_labels in train_dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(**batch).logits
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dl)
            print(f"Epoch {epoch}/{self.epochs}  loss={avg_loss:.4f}", end="")

            if val_texts and val_labels:
                from sklearn.metrics import f1_score
                val_preds = self._predict_ids(val_texts)
                val_ids = self._encode_labels(val_labels)
                val_f1 = f1_score(val_ids, val_preds, average="macro")
                print(f"  val_macro_f1={val_f1:.4f}", end="")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_count = 0
                    if save_dir:
                        self.save(save_dir)
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        print("\nEarly stopping.")
                        break
            print()

        return self

    def _predict_ids(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        ds = SpeechDataset(texts, [0] * len(texts), self.tokenizer)
        dl = DataLoader(ds, batch_size=self.batch_size)
        all_preds = []
        with torch.no_grad():
            for batch, _ in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        return np.array(all_preds)

    def predict(self, texts: list[str]) -> np.ndarray:
        ids = self._predict_ids(texts)
        return np.array([self.id2label[i] for i in ids])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        ds = SpeechDataset(texts, [0] * len(texts), self.tokenizer)
        dl = DataLoader(ds, batch_size=self.batch_size)
        all_probs = []
        sli_idx = self.label2id["SLI"]
        with torch.no_grad():
            for batch, _ in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                probs = torch.softmax(logits, dim=-1)
                all_probs.extend(probs[:, sli_idx].cpu().numpy())
        return np.array(all_probs)

    def get_cls_embeddings(self, texts: list[str]) -> np.ndarray:
        """Return [CLS] token embeddings (768-D) for the hybrid model.

        Reloads the model fresh from disk directly to CPU to avoid the MPS→CPU
        transfer hang that occurs on Rosetta when the model was trained on MPS.
        """
        checkpoint_dir = getattr(self, "_checkpoint_dir", None)
        if checkpoint_dir is not None:
            model_cpu = DistilBertForSequenceClassification.from_pretrained(
                str(checkpoint_dir)
            )
        else:
            # fallback: copy weights to a new CPU model
            model_cpu = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
            cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            model_cpu.load_state_dict(cpu_state)

        model_cpu.eval()
        ds = SpeechDataset(texts, [0] * len(texts), self.tokenizer)
        dl = DataLoader(ds, batch_size=32)
        embeddings = []
        with torch.no_grad():
            for batch, _ in dl:
                outputs = model_cpu(**batch, output_hidden_states=True)
                cls = outputs.hidden_states[-1][:, 0, :].numpy()
                embeddings.append(cls)
        del model_cpu
        return np.vstack(embeddings)

    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        import json
        (save_dir / "label_map.json").write_text(json.dumps(self.label2id))

    def load(self, load_dir: Path) -> "DistilBertClassifier":
        import json
        load_dir = Path(load_dir)
        self._checkpoint_dir = load_dir   # stored for CPU reload in get_cls_embeddings
        self.label2id = json.loads((load_dir / "label_map.json").read_text())
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.model = DistilBertForSequenceClassification.from_pretrained(load_dir).to(self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(load_dir)
        return self
