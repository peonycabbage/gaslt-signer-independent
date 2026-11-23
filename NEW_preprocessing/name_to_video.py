import argparse
import json
from pathlib import Path

import pandas as pd


def load_corpus(annotations_dir):
    splits = []
    for split in ("train", "dev", "test"):
        path = annotations_dir / f"PHOENIX-2014-T.{split}.corpus.csv"
        splits.append(pd.read_csv(path, sep="|", dtype=str).rename(columns=str.strip))

    data = pd.concat(splits, ignore_index=True)
    required = ["name", "speaker", "translation"]
    if data[required].isna().any().any():
        raise ValueError("Corpus contains missing names, signers, or translations")
    data["name"] = data["name"].str.strip()
    if data["name"].duplicated().any():
        raise ValueError("Video names must be unique across the corpus")
    return data

signer_folds = {
    1: {"dev": "Signer08", "test": "Signer01"},
    2: {"dev": "Signer01", "test": "Signer02"},
    3: {"dev": "Signer05", "test": "Signer03"},
    4: {"dev": "Signer03", "test": "Signer04"},
    5: {"dev": "Signer07", "test": "Signer05"},
    6: {"dev": "Signer04", "test": "Signer06"},
    7: {"dev": "Signer06", "test": "Signer07"},
    8: {"dev": "Signer09", "test": "Signer08"},
    9: {"dev": "Signer02", "test": "Signer09"},
}


def main():
    parser = argparse.ArgumentParser(description="Index PHOENIX14T videos for one fold")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=signer_folds, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pht/sim"))
    args = parser.parse_args()

    data = load_corpus(args.annotations_dir)
    signers = signer_folds[args.fold]
    dev = data["speaker"] == signers["dev"]
    test = data["speaker"] == signers["test"]
    names = [name for split in (~(dev | test), dev, test) for name in data.loc[split, "name"]]
    mapping = {name: index for index, name in enumerate(names)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"fold_{args.fold}.json"
    path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
