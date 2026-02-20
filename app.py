import streamlit as st
from ingest import ingest_pdf
from rag_engine import answer_query
import os

st.set_page_config(page_title="DocuSense Ai")

st.title("DocuSense Ai")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    pdf_path = "data/uploaded.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Process PDF"):
        try:
            ingest_pdf(pdf_path)
            st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

query = st.text_input("Ask a question about the PDF")

if st.button("Get Answer"):
    if query:
        try:
            answer = answer_query(query)
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")