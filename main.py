import os
import tempfile
import warnings
from typing import List

import PyPDF2
import google.generativeai as genai
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Model Configuration
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "gemini-1.5-pro-latest"


# Load PDF

def load_pdf(path: str) -> List[str]:
    try:
        reader = PyPDF2.PdfReader(path)
        return [p.extract_text() or "" for p in reader.pages]
    except Exception as e:
        return [f"error : {e}"]


# Dynamic Token-Based Splitting

def chunk_text(text: str) -> List[str]:
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    n = len(tokens)
    size = 800 if n > 2000 else (n // 2 if n > 1000 else n)
    overlap = int(size * 0.2)
    splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)


# Embedding and Vector Store

def create_embeddings(chunks: List[str]):
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={"device": "cpu"})
    return FAISS.from_texts(chunks, emb)


def get_relevant_chunks(db, query: str, k=8):
    docs = db.similarity_search(query, k=k)
    return [d.page_content for d in docs]


# Answer Generation (Gemini)

def generate_answer(context: List[str], query: str) -> str:
    joined = "\n---\n".join(context)
    prompt = (
        f"You are a helpful PDF assistant. Based on the context provided below, answer the user's question in a detailed and clear manner. "
        f"If there are multiple context chunks, consider all of them. Do not go beyond the content of the PDF.\n\n"
        f"Context:\n{joined}\n\nQuestion: {query}\n\nAnswer:"
    )
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(GEN_MODEL_NAME)
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"Failed to generate response: {e}"


# PDF Processing

def process_pdf(path: str, query: str) -> str:
    pages = load_pdf(path)
    if "error" in pages[0].lower():
        return pages[0]
    text = " ".join(pages)
    chunks = chunk_text(text)
    if not chunks:
        return "Text could not be extracted."
    db = create_embeddings(chunks)
    ctx = get_relevant_chunks(db, query)
    if not ctx:
        return "No relevant information found."
    return generate_answer(ctx, query)


# Streamlit Interface

def main():
    st.set_page_config(page_title="PDF QA", page_icon="📄", layout="wide")
    st.title("📄 PDF Question-Answer Assistant")
    st.markdown("Talk to your PDF — extract knowledge directly from it.")

    uploaded = st.file_uploader("Upload your PDF file", type="pdf")
    q = st.text_input("Type your question…", placeholder="Ask anything related to the PDF content")

    if uploaded and q:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        st.info("Generating answer…", icon="⏳")
        ans = process_pdf(tmp_path, q)
        st.success("Answer", icon="✅")
        st.write(ans)
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
