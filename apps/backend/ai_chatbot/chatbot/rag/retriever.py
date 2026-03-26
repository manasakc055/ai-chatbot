from .embeddings import model
from .vector_store import index, stored_chunks


def retrieve_chunks(query, k=3):

    if index is None or len(stored_chunks) == 0:
        print("⚠️ No indexed documents available")
        return []

    query_vector = model.encode([query])

    distances, indices = index.search(query_vector, k)

    results = []

    for i in indices[0]:

        if i < len(stored_chunks):
            results.append(stored_chunks[i])

    return results