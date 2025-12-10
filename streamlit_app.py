# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import os
from model_utils import get_model_and_transform, image_to_embedding
from db import load_all_embeddings
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="ePill Visual Search", layout="wide")
st.title("ePill — Busca por imagens similares (demo)")

# Paths
DB_PATH = Path("sqlite_db/epill.db")
EMBED_PATH = Path("data/embeddings.npy")

@st.cache_resource
def load_model():
    model, transform = get_model_and_transform(device="cpu")
    return model, transform

@st.cache_data(show_spinner=False)
def load_dataset_embeddings():
    if DB_PATH.exists():
        ids, paths, embs = load_all_embeddings(DB_PATH)
        if embs is None:
            return [], [], None
        return ids, paths, embs
    elif EMBED_PATH.exists():
        embs = np.load(EMBED_PATH)
        # if you saved paths separately, you'd load them — for simplicity require DB
        return [], [], embs
    else:
        return [], [], None

model, transform = load_model()
ids, paths, embs = load_dataset_embeddings()

if embs is None:
    st.warning("Dataset/embeddings não encontrados. Rode `python prepare_dataset.py` antes.")
    st.stop()

# Build NearestNeighbors index
nn = NearestNeighbors(n_neighbors=6, metric="cosine")
nn.fit(embs)

col1, col2 = st.columns([1,2])
with col1:
    st.header("Envie uma imagem")
    uploaded = st.file_uploader("Faça upload de uma foto de comprimido/imagem do dataset", type=["jpg","jpeg","png"])
    k = st.slider("Quantas imagens similares mostrar?", min_value=1, max_value=10, value=5)
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Imagem enviada", use_column_width=True)
        query_emb = image_to_embedding(image, model, transform, device="cpu")
        # Busca
        dists, idxs = nn.kneighbors([query_emb], n_neighbors=k+1, return_distance=True)
        dists = dists[0]
        idxs = idxs[0]
        # se o índice 0 for exatamente a mesma imagem do dataset (no caso de fotos idênticas), ignorar
        results = []
        for dist, idx in zip(dists, idxs):
            results.append((dist, idx))

with col2:
    st.header("Resultados similares")
    if uploaded:
        # Mostrar imagens
        cols = st.columns(3)
        for i, (dist, idx) in enumerate(results[1: k+1]):  # ignorar o primeiro se for a própria imagem
            pos = i % 3
            with cols[pos]:
                img_path = paths[idx] if paths else None
                if img_path and os.path.exists(img_path):
                    st.image(Image.open(img_path), caption=f"Rank {i+1} — dist {dist:.3f}")
                else:
                    st.write(f"Imagem {idx} (arquivo ausente)")