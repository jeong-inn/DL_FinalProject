# SpeechDev-Classifier

Binary classification of children's speech transcripts to detect **Specific Language Impairment (SLI)** vs. **Typically Developing (TD)** using the CHILDES ENNI corpus.

## Overview

| Model | Input | Key technique |
|---|---|---|
| TF-IDF Baseline | Raw text | TF-IDF (1-2 gram) + Logistic Regression |
| Linguistic SVM | Linguistic features | MLU, TTR, MATTR, NDW + SVM + SMOTE |
| DistilBERT | Raw text | Fine-tuned `distilbert-base-uncased` |
| Hybrid | Text + Linguistic | DistilBERT [CLS] + linguistic features → MLP |
| **Multimodal** *(best)* | Text + Audio + Linguistic + Prosodic | DistilBERT + wav2vec2 + handcrafted features → MLP |

### Linguistic Features (SLI-specific indicators)

| Feature | Description | SLI pattern |
|---|---|---|
| MLU-w | Mean Length of Utterance (words) | ↓ shorter utterances |
| TTR | Type-Token Ratio | ↓ less vocabulary diversity |
| MATTR | Moving Average TTR (window=50) | ↓ more stable diversity measure |
| CTTR | Corrected TTR | ↓ |
| NDW | Number of Different Words | ↓ smaller vocabulary |
| TNW | Total Number of Words | ↓ |
| Utterance length std | Variability in sentence length | varies |
| Proportion short utterances | % of 1-2 word utterances | ↑ more fragments |

### Acoustic / Prosodic Features

| Feature | Description | SLI pattern |
|---|---|---|
| Speech rate | Words per second | ↓ slower speech |
| Pause rate | Pauses per utterance | ↑ more hesitations |
| Mean pause duration | Average pause length (s) | ↑ longer pauses |
| F0 mean / std / range | Fundamental frequency statistics | varies |
| Energy mean / std | RMS energy statistics | ↓ lower energy |
| Speaking ratio | Fraction of voiced frames | ↓ more silence |
| MFCC (1-4) | Spectral shape summary | articulation differences |
| **wav2vec2** | facebook/wav2vec2-base embedding (768-D) | learned audio representation |

## Setup

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/SpeechDev-Classifier
cd SpeechDev-Classifier
pip install -r requirements.txt

# 2. Download ENNI dataset
#    Sign up at https://talkbank.org/childes/
#    Download ENNI and place at: data/raw/ENNI/
#    Expected layout: data/raw/ENNI/SLI/A/*.cha, SLI/B/, TD/A/, TD/B/

# 3. Preprocess + split (stratified 8:1:1 by group × sub_group)
python scripts/prepare_data.py

# 4. Pre-extract audio features (wav2vec2 + prosodic)
#    Skip this step if ENNI has no audio files — text-only models still work
python scripts/extract_audio_features.py

# 5. Train
python scripts/train.py --model all --epochs 10
```

## Project Structure

```
SpeechDev-Classifier/
├── src/
│   ├── data/
│   │   └── cha_parser.py       # .cha file parser
│   ├── features/
│   │   └── linguistic.py       # MLU, TTR, MATTR, NDW, CTTR, …
│   ├── models/
│   │   ├── baseline.py         # TF-IDF + Logistic Regression
│   │   ├── svm_classifier.py   # Linguistic features + SVM + SMOTE
│   │   ├── bert_classifier.py  # Fine-tuned DistilBERT
│   │   └── hybrid.py           # DistilBERT [CLS] + linguistic → MLP
│   └── evaluate.py             # Metrics, confusion matrix, comparison plots
├── scripts/
│   ├── prepare_data.py             # Scan ENNI dir → stratified split → CSV
│   ├── extract_audio_features.py   # Pre-extract wav2vec2 + prosodic features
│   └── train.py                    # Unified training & evaluation (5 models)
├── data/
│   └── splits/                 # Generated CSVs (not committed)
└── results/                    # Saved plots and metrics
```

## Running Individual Models

```bash
python scripts/train.py --model baseline
python scripts/train.py --model svm
python scripts/train.py --model bert        --epochs 10
python scripts/train.py --model hybrid      --epochs 10
python scripts/train.py --model multimodal  --epochs 10   # requires audio
```

> **Note**: `multimodal` falls back to zero-vectors for subjects without audio,
> so it runs even if only partial audio is available.

## Dataset

**ENNI** (Edmonton Narrative Norms Instrument) from the CHILDES TalkBank database.  
Registration required at https://talkbank.org/childes/

- Children labeled **SLI** (Specific Language Impairment) or **TD** (Typically Developing)
- Stratified split 8:1:1 by group × sub_group

## Background

SLI is characterized by difficulties acquiring language in the absence of neurological, hearing, or cognitive deficits. Key diagnostic markers include reduced MLU, limited vocabulary diversity (TTR), and shorter, more fragmented utterances — all captured by the linguistic features in this project.

## References

- Rice, M. L., & Wexler, K. (1996). Toward tense as a clinical marker of SLI.
- Leadholm, B. J., & Miller, J. F. (1992). Language sample analysis.
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: the moving-average type-token ratio (MATTR).
