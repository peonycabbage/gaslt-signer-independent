import os
import torch
import pickle
import gzip
import pandas as pd
from collections import defaultdict

# Define signer-fold mapping
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


import os
import pandas as pd
from collections import Counter




# def load_all_data(train_corpus, dev_corpus, test_corpus, pt_files_path):
#     """Load all corpus data from CSV files and ensure we store full dictionaries."""
#     data_entries = []
#     loaded_files = {}

#     for corpus_name, corpus_path in zip(["train", "dev", "test"], [train_corpus, dev_corpus, test_corpus]):
#         df = pd.read_csv(corpus_path, delimiter="|", dtype=str)
#         df.columns = df.columns.str.strip()

#         print(f"Loaded {df.shape[0]} samples from {corpus_name}")

#         for _, row in df.iterrows():
#             video_name = row["name"].strip()
#             pt_file_path = os.path.join(pt_files_path, video_name + ".pt")

#             # Load the .pt file
#             sign_tensor = torch.load(pt_file_path, map_location=torch.device('cpu')) if os.path.exists(pt_file_path) else None

#             # Store full dictionary (not just the name!)
#             entry = {
#                 "name": row["name"],
#                 "video": row["video"],
#                 "signer": row["speaker"],  # Ensure this column exists
#                 "gloss": row["orth"],
#                 "text": row["translation"],
#                 "sign": sign_tensor
#             }
#             data_entries.append(entry)  # Append dictionary, not just name

#             # Track file origins
#             loaded_files[video_name] = corpus_name

#     print(f"Total Samples Loaded: {len(data_entries)}")

#     return data_entries, loaded_files  # List of dictionaries, not just names


import os
import pandas as pd
import torch

def load_all_data(train_corpus, dev_corpus, test_corpus, pt_files_path):
    """Load all corpus data from CSV files and ensure we store full dictionaries."""
    data_entries = []
    loaded_files = {}

    for corpus_name, corpus_path in zip(["train", "dev", "test"], [train_corpus, dev_corpus, test_corpus]):
        df = pd.read_csv(corpus_path, delimiter="|", dtype=str)
        df.columns = df.columns.str.strip()

        print(f"Loaded {df.shape[0]} samples from {corpus_name}")

        for _, row in df.iterrows():
            video_name = row["name"].strip()
            pt_file_path = os.path.join(pt_files_path, video_name + ".pt")

            # Load the .pt file
            if os.path.exists(pt_file_path):
                sign_tensor = torch.load(pt_file_path, map_location=torch.device('cpu'))
                
                # Ensure sign_tensor is a single tensor, not a list of tensors
                if isinstance(sign_tensor, list):
                    sign_tensor = torch.cat(sign_tensor, dim=0)  # Concatenate along first axis

            else:
                sign_tensor = torch.empty(0)  # Default to empty tensor

            # Store full dictionary (not just the name!)
            entry = {
                "name": row["name"],
                "video": row["video"],
                "signer": row["speaker"],  # Ensure this column exists
                "gloss": row["orth"],
                "text": row["translation"],
                "sign": sign_tensor  # Ensure it's a single tensor
            }
            data_entries.append(entry)

            # Track file origins
            loaded_files[video_name] = corpus_name

    print(f"Total Samples Loaded: {len(data_entries)}")

    return data_entries, loaded_files







def save_splits_by_signer(data, save_dir, fold):
    """Splits dataset based on signer-fold mapping and saves train/dev/test sets correctly."""
    os.makedirs(save_dir, exist_ok=True)

    # Get signers for this fold
    dev_signer = signer_folds[fold]["dev"]
    test_signer = signer_folds[fold]["test"]

    # Correctly filter based on signers
    dev_data = [entry for entry in data if entry["signer"] == dev_signer]
    test_data = [entry for entry in data if entry["signer"] == test_signer]
    train_data = [entry for entry in data if entry["signer"] not in [dev_signer, test_signer]]

    # Save datasets
    with gzip.open(os.path.join(save_dir, "phoenix14t.pami0.train"), "wb") as f:
        pickle.dump(train_data, f)
    with gzip.open(os.path.join(save_dir, "phoenix14t.pami0.dev"), "wb") as f:
        pickle.dump(dev_data, f)
    with gzip.open(os.path.join(save_dir, "phoenix14t.pami0.test"), "wb") as f:
        pickle.dump(test_data, f)

    print(f" Fold {fold} Saved: {len(train_data)} Train | {len(dev_data)} Dev | {len(test_data)} Test")


# File paths
train_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
dev_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
test_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv"
pt_files_path = "/mnt/c/Research/GASLT/data/span=8_stride=2"
save_directory_base = "/mnt/c/Research/GASLT/data/pht/data/signer_folds"

# Load ALL data
all_data, _ = load_all_data(train_corpus, dev_corpus, test_corpus, pt_files_path)


# Process each fold separately
for fold in signer_folds.keys():
    save_directory = os.path.join(save_directory_base, f"fold_{fold}")
    print(f"Type of all_data: {type(all_data)}")
    print(f"Sample entry: {all_data[0] if all_data else 'Empty list!'}")

    save_splits_by_signer(all_data, save_directory, fold)


# Save the official dev set separately (without modification)
official_dev_data = load_all_data(train_corpus, dev_corpus, test_corpus, pt_files_path)
official_dev_path = os.path.join(save_directory_base, "official_dev", "phoenix14t.pami0.dev")
os.makedirs(os.path.dirname(official_dev_path), exist_ok=True)
with gzip.open(official_dev_path, "wb") as f:
    pickle.dump(official_dev_data, f)
print(f"Saved official dev set: {official_dev_path}")

# Save the official test set separately (without modification)
official_test_data = load_all_data(train_corpus, dev_corpus, test_corpus, pt_files_path)
official_test_path = os.path.join(save_directory_base, "official_test", "phoenix14t.pami0.test")
os.makedirs(os.path.dirname(official_test_path), exist_ok=True)
with gzip.open(official_test_path, "wb") as f:
    pickle.dump(official_test_data, f)
print(f"Saved official test set: {official_test_path}")


