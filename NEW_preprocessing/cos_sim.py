import argparse
import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer

from name_to_video import load_corpus, signer_folds


def main():
    parser = argparse.ArgumentParser(description="Compute GASLT similarity labels for one fold")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=signer_folds, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/pht/sim"))
    args = parser.parse_args()

    mapping_path = args.output_dir / f"fold_{args.fold}.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    translations = load_corpus(args.annotations_dir).set_index("name")["translation"]
    if set(mapping) != set(translations.index) or sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Regenerate the zero-based fold mapping with name_to_video.py")
    names = sorted(mapping, key=mapping.get)
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(translations.loc[names].tolist(), normalize_embeddings=True)
    similarities = (embeddings @ embeddings.T).astype("float64")

    path = args.output_dir / f"fold_{args.fold}_cos_sim.pkl"
    with path.open("wb") as stream:
        pickle.dump(similarities, stream, protocol=4)
    print(f"{path}: {similarities.shape}")


if __name__ == "__main__":
    main()
