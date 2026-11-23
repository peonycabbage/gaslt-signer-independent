import argparse
import gzip
import pickle
from pathlib import Path

import torch

from name_to_video import load_corpus, signer_folds


def main():
    parser = argparse.ArgumentParser(description="Pack one PHOENIX14T signer-independent fold")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=signer_folds, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pht/data/signer_folds"))
    args = parser.parse_args()

    data = load_corpus(args.annotations_dir)
    signers = signer_folds[args.fold]
    splits = {"train": [], "dev": [], "test": []}
    for row in data.itertuples(index=False):
        features = torch.load(args.features_dir / f"{row.name}.pt", map_location="cpu")
        if isinstance(features, list):
            features = torch.cat(features, dim=0)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != 1024:
            raise ValueError(f"Expected nonempty [time, 1024] features for {row.name}")
        split = "train"
        if row.speaker == signers["dev"]:
            split = "dev"
        elif row.speaker == signers["test"]:
            split = "test"
        splits[split].append({
            "name": row.name,
            "video": row.video,
            "signer": row.speaker,
            "gloss": row.orth,
            "text": row.translation,
            "sign": features,
        })

    if any(not entries for entries in splits.values()):
        raise ValueError("Each split must contain at least one sample")
    output = args.output_dir / f"fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    for split, entries in splits.items():
        path = output / f"phoenix14t.pami0.{split}"
        with gzip.open(path, "wb") as stream:
            pickle.dump(entries, stream, protocol=4)
        print(f"{path}: {len(entries)} samples")


if __name__ == "__main__":
    main()
