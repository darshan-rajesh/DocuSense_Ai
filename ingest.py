import os
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


def ingest_pdf(pdf_path):
    # 1️⃣ Read PDF
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        raise ValueError("No text extracted from PDF.")

    # 2️⃣ Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(text)

    if len(chunks) == 0:
        raise ValueError("No chunks created.")

    print("Chunks created:", len(chunks))

    # 3️⃣ Load embedding model (GPU)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embed_model.encode(
        chunks,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    dimension = embeddings.shape[1]

    # 4️⃣ Create FAISS index (Inner Product = faster with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs("vector_store", exist_ok=True)

    faiss.write_index(index, "vector_store/index.faiss")

    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("PDF indexed successfully.")