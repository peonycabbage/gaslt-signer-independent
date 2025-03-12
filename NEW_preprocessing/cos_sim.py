from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import os

# Paths to CSV corpus files
train_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
dev_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
test_corpus = "/mnt/c/Research/Dataset/phoenix-2014_o/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv"

# Load all datasets
def load_translations(corpus_path):
    df = pd.read_csv(corpus_path, delimiter="|", dtype=str)
    return df["translation"].dropna().tolist()  # Remove NaN values

train_texts = load_translations(train_corpus)
dev_texts = load_translations(dev_corpus)
test_texts = load_translations(test_corpus)

# Combine all sentences
all_sentences = train_texts + dev_texts + test_texts

print(f"Loaded {len(train_texts)} train, {len(dev_texts)} dev, {len(test_texts)} test sentences.")
print(f"Total sentences: {len(all_sentences)}")

# Load the Sentence-Transformer model
model = SentenceTransformer("all-mpnet-base-v2")

# Compute embeddings
print("Generating embeddings...")
embeddings = model.encode(all_sentences)
print("Embeddings generated!")
print(f"Debug: Embeddings shape before saving: {embeddings.shape}")  # Should be (N, 512)

# Compute cosine similarity matrix
print("Computing cosine similarity matrix...")
similarities = model.similarity(embeddings, embeddings)

print(f"Cosine similarity matrix shape: {similarities.shape}")  # Should be (N, N)

# Save the cosine similarity matrix to `cos_sim.pkl`
save_dir = "/mnt/c/Research/GASLT/data/pht/sim"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
save_path = os.path.join(save_dir, "cos_sim.pkl")  # Define full save path

with open(save_path, 'wb') as file:
    pickle.dump(similarities, file)

print(f"Saved cosine similarity matrix to: {save_path}")
