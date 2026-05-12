"""
Pre-extract wav2vec2 embeddings and prosodic features from audio files.

Run this once after prepare_data.py. Results are saved as .npy files so
training doesn't need to reload audio every run.

Output layout:
    data/audio_features/
        {subject_id}_wav2vec2.npy    (768,)
        {subject_id}_prosodic.npy    (13,)
        missing_audio.txt            subjects with no audio file

Usage:
    python scripts/extract_audio_features.py --enni_dir data/raw/ENNI
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cha_parser import extract_utterances
from src.data.audio_parser import extract_child_segments
from src.features.acoustic import Wav2Vec2Encoder, extract_prosodic, PROSODIC_FEATURE_NAMES


def process_subject(
    cha_path: Path,
    encoder: Wav2Vec2Encoder,
    out_dir: Path,
    subject_id: str,
) -> bool:
    """Returns True if audio was found and processed."""
    utterances = extract_utterances(cha_path)
    segments = extract_child_segments(cha_path, utterances)

    if segments is None:
        return False

    chi_texts = [u.clean_text for u in utterances if u.speaker == 'CHI' and u.clean_text]

    wav2vec_emb = encoder.encode(segments)
    prosodic_feat = extract_prosodic(segments, chi_texts)

    np.save(out_dir / f"{subject_id}_wav2vec2.npy", wav2vec_emb)
    np.save(out_dir / f"{subject_id}_prosodic.npy", prosodic_feat)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enni_dir", default="data/raw/ENNI")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--out_dir", default="data/audio_features")
    args = parser.parse_args()

    enni_dir = Path(args.enni_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect all subjects from all splits
    all_rows = []
    for split in ("train", "dev", "test"):
        csv_path = splits_dir / f"ENNI_{split}_ready.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_rows.append(df)
    if not all_rows:
        print("No ready CSVs found. Run prepare_data.py first.")
        return

    subjects = pd.concat(all_rows).drop_duplicates("subject_id")
    print(f"Loading wav2vec2 model …")
    encoder = Wav2Vec2Encoder()

    found, missing = 0, []
    for _, row in subjects.iterrows():
        sid = row["subject_id"]
        # find .cha file by scanning enni_dir
        matches = list(enni_dir.rglob(f"{sid}.cha"))
        if not matches:
            missing.append(sid)
            continue

        cha_path = matches[0]
        print(f"  Processing {sid} …", end=" ")
        ok = process_subject(cha_path, encoder, out_dir, sid)
        if ok:
            found += 1
            print("done")
        else:
            missing.append(sid)
            print("no audio")

    (out_dir / "missing_audio.txt").write_text("\n".join(missing))
    print(f"\nExtracted audio features for {found}/{len(subjects)} subjects.")
    print(f"Missing audio: {len(missing)} subjects (see {out_dir}/missing_audio.txt)")


if __name__ == "__main__":
    main()
