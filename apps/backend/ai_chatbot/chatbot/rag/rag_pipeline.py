from .document_loader import load_pdf
from .text_splitter import split_text
from .embeddings import create_embeddings
from .vector_store import add_vectors


def process_document(file_path):

    print("🚀 Processing document:", file_path)

    text = load_pdf(file_path)

    print("📄 Extracted text length:", len(text))

    chunks = split_text(text)

    print("📄 Total chunks created:", len(chunks))

    embeddings = create_embeddings(chunks)

    print("🧠 Embeddings created:", len(embeddings))

    add_vectors(embeddings, chunks)

    print("✅ Document indexed successfully")