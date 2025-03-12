import json
import os
import pandas as pd

# Paths to CSV corpus files
train_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
dev_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
test_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv"

# Directory to save JSON files
save_dir = "/mnt/c/Research/GASLT/data/pht/sim"
os.makedirs(save_dir, exist_ok=True)

# Signer-fold mapping
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

# Load CSV files into Pandas DataFrames
def load_corpus(corpus_path):
    df = pd.read_csv(corpus_path, delimiter="|", dtype=str)
    df.columns = df.columns.str.strip()  # Remove spaces from column names
    return df

train_df = load_corpus(train_corpus)
dev_df = load_corpus(dev_corpus)
test_df = load_corpus(test_corpus)

# Combine all datasets
all_data = pd.concat([train_df, dev_df, test_df], ignore_index=True)

# Generate JSON for each fold
for fold, signers in signer_folds.items():
    fold_dict = {}
    count = 1  # Start numbering from 1

    # Split into Train, Dev, and Test based on signer mapping
    dev_signer = signers["dev"]
    test_signer = signers["test"]

    dev_data = all_data[all_data["speaker"] == dev_signer]
    test_data = all_data[all_data["speaker"] == test_signer]
    train_data = all_data[(all_data["speaker"] != dev_signer) & (all_data["speaker"] != test_signer)]

    # Add videos to fold_dict with unique integer values
    for dataset in [train_data, dev_data, test_data]:
        for _, row in dataset.iterrows():
            video_name = row["name"]  # Only store the video name
            fold_dict[video_name] = count
            count += 1  # Increment count for uniqueness

    # Save the JSON file for this fold
    json_filename = os.path.join(save_dir, f"fold_{fold}.json")
    with open(json_filename, "w") as outfile:
        json.dump(fold_dict, outfile, indent=4)

    print(f"Saved JSON for Fold {fold}: {json_filename}")


# import json
# import os
# import pandas as pd

# # Paths to CSV corpus files
# train_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
# dev_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
# test_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv"

# # Directory to save JSON files
# save_dir = "/mnt/c/Research/GASLT/data/pht/sim"
# os.makedirs(save_dir, exist_ok=True)

# # Signer-fold mapping
# signer_folds = {
#     1: {"dev": "Signer08", "test": "Signer01"},
#     2: {"dev": "Signer01", "test": "Signer02"},
#     3: {"dev": "Signer05", "test": "Signer03"},
#     4: {"dev": "Signer03", "test": "Signer04"},
#     5: {"dev": "Signer07", "test": "Signer05"},
#     6: {"dev": "Signer04", "test": "Signer06"},
#     7: {"dev": "Signer06", "test": "Signer07"},
#     8: {"dev": "Signer09", "test": "Signer08"},
#     9: {"dev": "Signer02", "test": "Signer09"},
# }

# # Load CSV files into Pandas DataFrames
# def load_corpus(corpus_path):
#     df = pd.read_csv(corpus_path, delimiter="|", dtype=str)
#     df.columns = df.columns.str.strip()  # Remove spaces from column names
#     return df

# train_df = load_corpus(train_corpus)
# dev_df = load_corpus(dev_corpus)
# test_df = load_corpus(test_corpus)

# # Combine all datasets
# all_data = pd.concat([train_df, dev_df, test_df], ignore_index=True)

# # Generate JSON for each fold
# for fold, signers in signer_folds.items():
#     fold_dict = {}

#     # Split into Train, Dev, and Test based on signer mapping
#     dev_signer = signers["dev"]
#     test_signer = signers["test"]

#     dev_data = all_data[all_data["speaker"] == dev_signer]
#     test_data = all_data[all_data["speaker"] == test_signer]
#     train_data = all_data[(all_data["speaker"] != dev_signer) & (all_data["speaker"] != test_signer)]

#     # Reset numbering for each dataset
#     count_train = 1
#     count_dev = 1
#     count_test = 1

#     # Assign numbers separately for train, dev, and test
#     for _, row in train_data.iterrows():
#         video_name = row["name"]
#         fold_dict[video_name] = count_train
#         count_train += 1

#     for _, row in dev_data.iterrows():
#         video_name = row["name"]
#         fold_dict[video_name] = count_dev
#         count_dev += 1

#     for _, row in test_data.iterrows():
#         video_name = row["name"]
#         fold_dict[video_name] = count_test
#         count_test += 1

#     # Save the JSON file for this fold
#     json_filename = os.path.join(save_dir, f"fold_{fold}.json")
#     with open(json_filename, "w") as outfile:
#         json.dump(fold_dict, outfile, indent=4)

#     print(f"Saved JSON for Fold {fold}: {json_filename}")
