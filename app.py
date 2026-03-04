import os
from pathlib import Path
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

DATA_DIR = Path(__file__).resolve().parent / "data"

def load_corpus(folder: Path) -> str:
    parts = []
    for p in sorted(folder.glob("*.txt")):
        parts.append(f"\n\n===== FILE: {p.name} =====\n\n")
        parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "".join(parts)

def chunk_text(text: str, chunk_size: int, overlap: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >=0 and < chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

def embed_texts(texts):
    embs = []
    for t in texts:
        r = client.embeddings.create(model="text-embedding-3-small", input=t)
        embs.append(r.data[0].embedding)
    return np.array(embs, dtype="float32")

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def top_k_chunks(query, chunks, index, k=3):
    q = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    D, I = index.search(np.array([q], dtype="float32"), k=k)
    return [(int(rank), float(D[0][rank]), chunks[int(I[0][rank])]) for rank in range(len(I[0]))]

def preview_chunk(s: str, n=220):
    s = " ".join(s.split())
    return (s[:n] + "…") if len(s) > n else s

def run_setting(corpus, query, chunk_size, overlap, k=3):
    chunks = chunk_text(corpus, chunk_size=chunk_size, overlap=overlap)
    embeddings = embed_texts(chunks)
    index = build_index(embeddings)
    results = top_k_chunks(query, chunks, index, k=k)

    print(f"\n=== Setting: chunk_size={chunk_size}, overlap={overlap} | chunks={len(chunks)} ===")
    for rank, dist, chunk in results:
        print(f"\n#{rank+1}  (L2 distance: {dist:.4f})")
        print(preview_chunk(chunk))

def main():
    if not DATA_DIR.exists():
        raise RuntimeError(f"Missing folder: {DATA_DIR} (add .txt files)")

    corpus = load_corpus(DATA_DIR)
    if not corpus.strip():
        raise RuntimeError("No text found in data/")

    query = input("Query: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return

    settings = [
        (300, 50),
        (600, 100),
        (900, 150),
    ]

    for chunk_size, overlap in settings:
        run_setting(corpus, query, chunk_size, overlap, k=3)

    print("\nDone. Tip: change settings list to test more combos.")

if __name__ == "__main__":

    main()

