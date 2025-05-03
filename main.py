import os
import tempfile
import warnings
from typing import List

import PyPDF2
import google.generativeai as genai
import streamlit as st
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

warnings.filterwarnings("ignore")
# — API Anahtarları
os.environ["GOOGLE_API_KEY"] = "AIzaSyBAncSXs0GozEJf9IoqbMMLZ6hIbPQCymY"

# — Model Seçimi
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Aşağıdakilerden birini deneyin: gemini-1.5-pro-latest, gemini-1.5-flash
GEN_MODEL_NAME = "gemini-1.5-pro-latest"


# — PDF Yükleme
def load_pdf(path: str) -> List[str]:
    try:
        reader = PyPDF2.PdfReader(path)
        return [p.extract_text() or "" for p in reader.pages]
    except Exception as e:
        return [f"PDF yüklenirken hata: {e}"]


# — Dinamik token‑tabanlı chunking
def chunk_text(text: str) -> List[str]:
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    n = len(tokens)
    size = 1000 if n > 2000 else (n // 2 if n > 1000 else n)
    overlap = int(size * 0.1)
    splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)


# — Embedding ve vektör DB
def create_embeddings(chunks: List[str]):
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={"device": "cpu"})
    return FAISS.from_texts(chunks, emb)


def get_relevant_chunks(db, query: str, k=4):
    docs = db.similarity_search(query, k=k)
    return [d.page_content for d in docs]


# — Cevap üretme (Gemini)
def generate_answer(context: List[str], query: str) -> str:
    joined = "\n".join(context)
    prompt = f"Bağlam:\n{joined}\n\nSoru: {query}\n\nYanıt:"
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(GEN_MODEL_NAME)  # ← artık sadece model adı
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"Cevap oluşturulamadı: {e}"


# — PDF→Cevap
def process_pdf(path: str, query: str) -> str:
    pages = load_pdf(path)
    if "hata" in pages[0].lower():
        return pages[0]
    text = " ".join(pages)
    chunks = chunk_text(text)
    if not chunks:
        return "Metin çıkarılamadı."
    db = create_embeddings(chunks)
    ctx = get_relevant_chunks(db, query)
    if not ctx:
        return "İlgili bilgi bulunamadı."
    return generate_answer(ctx, query)


# — Streamlit UI
def main():
    st.set_page_config(page_title="PDF QA", page_icon="📄")
    st.title("PDF Soru–Cevap (Gemini)")
    uploaded = st.file_uploader("PDF yükleyin", type="pdf")
    q = st.text_input("Sorunuz")
    if uploaded and q:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        st.info("İşleniyor…")
        ans = process_pdf(tmp_path, q)
        st.success("Cevap:")
        st.write(ans)
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
