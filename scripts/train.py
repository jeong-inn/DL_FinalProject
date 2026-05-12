"""
Unified training script for all models.

Usage:
    python scripts/train.py --model baseline
    python scripts/train.py --model svm
    python scripts/train.py --model bert       --epochs 10
    python scripts/train.py --model hybrid     --epochs 10
    python scripts/train.py --model multimodal --epochs 10
    python scripts/train.py --model all        --epochs 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import compute_metrics, plot_confusion_matrix, print_metrics
from src.features.linguistic import extract_features


def load_split(csv_path: Path) -> tuple[list[str], list[list[str]], list[str], list[str]]:
    df = pd.read_csv(csv_path)
    texts = df["text"].fillna("").tolist()
    utterances = [json.loads(u) for u in df["utterances_json"].fillna("[]")]
    labels = df["group"].tolist()
    subject_ids = df["subject_id"].astype(str).tolist()
    return texts, utterances, labels, subject_ids


def build_ling_features(utterance_lists: list[list[str]]) -> np.ndarray:
    return np.vstack([extract_features(utts) for utts in utterance_lists])


def load_audio_features(subject_ids: list[str], audio_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-extracted wav2vec2 and prosodic features. Returns zero arrays if missing."""
    from src.features.acoustic import WAV2VEC2_DIM, PROSODIC_FEATURE_NAMES

    wav2vec_list, prosodic_list = [], []
    for sid in subject_ids:
        w2v_path = audio_dir / f"{sid}_wav2vec2.npy"
        pro_path = audio_dir / f"{sid}_prosodic.npy"
        wav2vec_list.append(np.load(w2v_path) if w2v_path.exists() else np.zeros(WAV2VEC2_DIM))
        prosodic_list.append(np.load(pro_path) if pro_path.exists() else np.zeros(len(PROSODIC_FEATURE_NAMES)))

    return np.vstack(wav2vec_list), np.vstack(prosodic_list)


# ── individual runners ──────────────────────────────────────────────────────

def run_baseline(splits_dir: Path, results_dir: Path) -> dict:
    from src.models.baseline import TfidfBaseline

    train_texts, _, train_labels, _ = load_split(splits_dir / "ENNI_train_ready.csv")
    dev_texts,   _, dev_labels,   _ = load_split(splits_dir / "ENNI_dev_ready.csv")
    test_texts,  _, test_labels,  _ = load_split(splits_dir / "ENNI_test_ready.csv")

    model = TfidfBaseline()
    model.fit(train_texts, train_labels)

    print_metrics(compute_metrics(dev_labels, model.predict(dev_texts), model.predict_proba(dev_texts)), "Baseline — Dev")
    test_metrics = compute_metrics(test_labels, model.predict(test_texts), model.predict_proba(test_texts))
    print_metrics(test_metrics, "Baseline — Test")
    plot_confusion_matrix(test_labels, model.predict(test_texts), "Baseline", results_dir / "cm_baseline.png")
    return test_metrics


def run_svm(splits_dir: Path, results_dir: Path) -> dict:
    from src.models.svm_classifier import LinguisticSVM

    train_texts, train_utts, train_labels, _ = load_split(splits_dir / "ENNI_train_ready.csv")
    dev_texts,   dev_utts,   dev_labels,   _ = load_split(splits_dir / "ENNI_dev_ready.csv")
    test_texts,  test_utts,  test_labels,  _ = load_split(splits_dir / "ENNI_test_ready.csv")

    train_X = build_ling_features(train_utts)
    dev_X   = build_ling_features(dev_utts)
    test_X  = build_ling_features(test_utts)

    model = LinguisticSVM()
    model.fit(train_X, train_labels)

    print_metrics(compute_metrics(dev_labels, model.predict(dev_X), model.predict_proba(dev_X)), "SVM — Dev")
    test_metrics = compute_metrics(test_labels, model.predict(test_X), model.predict_proba(test_X))
    print_metrics(test_metrics, "SVM — Test")
    plot_confusion_matrix(test_labels, model.predict(test_X), "Linguistic SVM", results_dir / "cm_svm.png")
    return test_metrics


def run_bert(splits_dir: Path, results_dir: Path, epochs: int) -> dict:
    from src.models.bert_classifier import DistilBertClassifier

    train_texts, _, train_labels, _ = load_split(splits_dir / "ENNI_train_ready.csv")
    dev_texts,   _, dev_labels,   _ = load_split(splits_dir / "ENNI_dev_ready.csv")
    test_texts,  _, test_labels,  _ = load_split(splits_dir / "ENNI_test_ready.csv")

    model = DistilBertClassifier(epochs=epochs)
    model.fit(train_texts, train_labels, val_texts=dev_texts, val_labels=dev_labels,
              save_dir=results_dir / "bert_checkpoint")

    print_metrics(compute_metrics(dev_labels, model.predict(dev_texts), model.predict_proba(dev_texts)), "BERT — Dev")
    test_metrics = compute_metrics(test_labels, model.predict(test_texts), model.predict_proba(test_texts))
    print_metrics(test_metrics, "BERT — Test")
    plot_confusion_matrix(test_labels, model.predict(test_texts), "DistilBERT", results_dir / "cm_bert.png")
    return test_metrics


def run_hybrid(splits_dir: Path, results_dir: Path, epochs: int) -> dict:
    from src.models.hybrid import HybridClassifier

    train_texts, train_utts, train_labels, _ = load_split(splits_dir / "ENNI_train_ready.csv")
    dev_texts,   dev_utts,   dev_labels,   _ = load_split(splits_dir / "ENNI_dev_ready.csv")
    test_texts,  test_utts,  test_labels,  _ = load_split(splits_dir / "ENNI_test_ready.csv")

    train_ling = build_ling_features(train_utts)
    dev_ling   = build_ling_features(dev_utts)
    test_ling  = build_ling_features(test_utts)

    model = HybridClassifier(bert_epochs=epochs)
    model.fit(train_texts, train_ling, train_labels,
              val_texts=dev_texts, val_ling=dev_ling, val_labels=dev_labels,
              bert_checkpoint=results_dir / "bert_checkpoint")

    print_metrics(compute_metrics(dev_labels, model.predict(dev_texts, dev_ling), model.predict_proba(dev_texts, dev_ling)), "Hybrid — Dev")
    test_metrics = compute_metrics(test_labels, model.predict(test_texts, test_ling), model.predict_proba(test_texts, test_ling))
    print_metrics(test_metrics, "Hybrid — Test")
    plot_confusion_matrix(test_labels, model.predict(test_texts, test_ling), "Hybrid", results_dir / "cm_hybrid.png")
    return test_metrics


def run_multimodal(splits_dir: Path, results_dir: Path, epochs: int, audio_dir: Path) -> dict:
    from src.models.multimodal import MultimodalClassifier

    train_texts, train_utts, train_labels, train_ids = load_split(splits_dir / "ENNI_train_ready.csv")
    dev_texts,   dev_utts,   dev_labels,   dev_ids   = load_split(splits_dir / "ENNI_dev_ready.csv")
    test_texts,  test_utts,  test_labels,  test_ids  = load_split(splits_dir / "ENNI_test_ready.csv")

    train_ling = build_ling_features(train_utts)
    dev_ling   = build_ling_features(dev_utts)
    test_ling  = build_ling_features(test_utts)

    train_w2v, train_pro = load_audio_features(train_ids, audio_dir)
    dev_w2v,   dev_pro   = load_audio_features(dev_ids,   audio_dir)
    test_w2v,  test_pro  = load_audio_features(test_ids,  audio_dir)

    model = MultimodalClassifier(bert_epochs=epochs)
    model.fit(
        train_texts, train_w2v, train_ling, train_pro, train_labels,
        val_texts=dev_texts, val_wav2vec=dev_w2v, val_ling=dev_ling,
        val_prosodic=dev_pro, val_labels=dev_labels,
        bert_checkpoint=results_dir / "bert_checkpoint",
    )

    dev_preds = model.predict(dev_texts, dev_w2v, dev_ling, dev_pro)
    test_preds = model.predict(test_texts, test_w2v, test_ling, test_pro)
    test_proba = model.predict_proba(test_texts, test_w2v, test_ling, test_pro)

    print_metrics(compute_metrics(dev_labels, dev_preds), "Multimodal — Dev")
    test_metrics = compute_metrics(test_labels, test_preds, test_proba)
    print_metrics(test_metrics, "Multimodal — Test")
    plot_confusion_matrix(test_labels, test_preds, "Multimodal", results_dir / "cm_multimodal.png")
    return test_metrics


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "svm", "bert", "hybrid", "multimodal", "all"], default="all")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    splits_dir = Path(args.data_dir) / "splits"
    audio_dir  = Path(args.data_dir) / "audio_features"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        "baseline":   lambda: run_baseline(splits_dir, results_dir),
        "svm":        lambda: run_svm(splits_dir, results_dir),
        "bert":       lambda: run_bert(splits_dir, results_dir, args.epochs),
        "hybrid":     lambda: run_hybrid(splits_dir, results_dir, args.epochs),
        "multimodal": lambda: run_multimodal(splits_dir, results_dir, args.epochs, audio_dir),
    }

    to_run = list(runners.keys()) if args.model == "all" else [args.model]
    all_metrics: dict[str, dict] = {}

    for name in to_run:
        print(f"\n{'='*50}\nRunning: {name}\n{'='*50}")
        all_metrics[name] = runners[name]()

    if len(all_metrics) > 1:
        from src.evaluate import plot_model_comparison
        plot_model_comparison(all_metrics, results_dir / "model_comparison.png")

    import json as _json
    (results_dir / "summary.json").write_text(_json.dumps(all_metrics, indent=2))
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
