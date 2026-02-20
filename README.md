# DocuSense AI

A GPU-accelerated Local PDF RAG Assistant built using:

- Sentence Transformers
- FAISS
- HuggingFace Transformers
- Streamlit
- PyTorch (CUDA)

## Features

- Upload PDF
- Automatic chunking
- Vector embedding
- Semantic search
- Context-aware answer generation
- GPU acceleration (RTX supported)

## Tech Stack

- Python 3.10
- FAISS (Vector DB)
- flan-t5-small
- all-MiniLM-L6-v2
- Streamlit UI

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py