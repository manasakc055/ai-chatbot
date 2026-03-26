import faiss
import pickle
import os

INDEX_PATH = "rag_index.faiss"
CHUNKS_PATH = "rag_chunks.pkl"

dimension = 384

# Load FAISS index if exists
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(dimension)

# Load stored chunks
if os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, "rb") as f:
        stored_chunks = pickle.load(f)
else:
    stored_chunks = []


def add_vectors(vectors, chunks):
    global index, stored_chunks

    print("Adding vectors to FAISS:", vectors.shape)

    index.add(vectors)
    stored_chunks.extend(chunks)

    save_index()


def save_index():
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(stored_chunks, f)

    print("✅ FAISS index saved")