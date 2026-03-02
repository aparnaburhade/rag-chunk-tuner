# RAG Chunk Tuner

Small experiment tool to see how chunk size + overlap affects retrieval results in RAG.

## What it does
- Loads `.txt` files from `/data`
- Builds chunks using different `(chunk_size, overlap)` settings
- Embeds chunks + stores in FAISS
- Prints Top-K retrieved chunks for the same query for each setting

## Run
pip install -r requirements.txt
python app.py