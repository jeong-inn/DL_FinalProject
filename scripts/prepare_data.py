"""
Scan ENNI directory, perform stratified split, and save ready-to-use CSVs.

Expected ENNI directory layout:
    data/raw/ENNI/
        SLI/A/*.cha
        SLI/B/*.cha
        TD/A/*.cha
        TD/B/*.cha

Output (data/splits/):
    ENNI_train_ready.csv
    ENNI_dev_ready.csv
    ENNI_test_ready.csv

Usage:
    python scripts/prepare_data.py --enni_dir data/raw/ENNI
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cha_parser import extract_utterances


def scan_enni(enni_dir: Path) -> pd.DataFrame:
    records = []
    for cha_path in sorted(enni_dir.rglob("*.cha")):
        parts = cha_path.relative_to(enni_dir).parts
        if len(parts) < 3:
            continue
        group, sub_group = parts[0], parts[1]   # e.g. SLI, A
        records.append({
            "filepath": str(cha_path),
            "group": group,
            "sub_group": sub_group,
            "subject_id": cha_path.stem,
            "strat_key": f"{group}-{sub_group}",
        })
    return pd.DataFrame(records)


def extract_text(row: pd.Series) -> tuple[str, list[str]]:
    utts = extract_utterances(row["filepath"], speakers=["CHI"])
    clean = [u.clean_text for u in utts if u.clean_text]
    return " ".join(clean), clean


def stratified_split(df: pd.DataFrame, dev_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=df["strat_key"], random_state=seed
    )
    val_ratio_adjusted = dev_ratio / (1 - test_ratio)
    train, dev = train_test_split(
        train_val, test_size=val_ratio_adjusted, stratify=train_val["strat_key"], random_state=seed
    )
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)


def build_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        text, utts = extract_text(row)
        rows.append({
            "subject_id": row["subject_id"],
            "group": row["group"],
            "sub_group": row["sub_group"],
            "text": text,
            "utterances_json": json.dumps(utts),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enni_dir", default="data/raw/ENNI")
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    enni_dir = Path(args.enni_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {enni_dir} …")
    df = scan_enni(enni_dir)
    if df.empty:
        print("No .cha files found. Check --enni_dir path.")
        return

    print(f"Found {len(df)} files  |  {df['group'].value_counts().to_dict()}")

    train_df, dev_df, test_df = stratified_split(df, seed=args.seed)
    print(f"Split → train={len(train_df)}  dev={len(dev_df)}  test={len(test_df)}")

    for name, split_df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        print(f"Extracting utterances for {name} …")
        ready = build_ready_df(split_df)
        out_path = out_dir / f"ENNI_{name}_ready.csv"
        ready.to_csv(out_path, index=False)
        print(f"  → {out_path}  ({len(ready)} samples)")

    print("Done.")


if __name__ == "__main__":
    main()
