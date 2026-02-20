import pickle
import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ==========================
# DEVICE SETUP
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INDEX = 0 if torch.cuda.is_available() else -1

print("Using device:", DEVICE)


# ==========================
# LOAD EMBEDDING MODEL (GPU)
# ==========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)


# ==========================
# LOAD GENERATOR MODEL (FAST + FP16)
# ==========================
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

if DEVICE == "cuda":
    model = model.to("cuda")

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE_INDEX
)


# ==========================
# ANSWER FUNCTION
# ==========================
def answer_query(query):
    # 1️⃣ Load FAISS index
    index = faiss.read_index("vector_store/index.faiss")

    # 2️⃣ Load stored chunks
    with open("vector_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # 3️⃣ Embed query
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")

    # 4️⃣ Retrieve TOP 1 chunk (faster)
    distances, indices = index.search(query_embedding, 1)

    retrieved_chunk = chunks[indices[0][0]]

    # 5️⃣ Build compact prompt (smaller = faster)
    prompt = f"""
Answer the question based only on the context below.

Context:
{retrieved_chunk}

Question:
{query}
"""

    # 6️⃣ Generate answer (shorter max_length)
    result = generator(
        prompt,
        max_length=128,
        do_sample=False
    )

    return result[0]["generated_text"]