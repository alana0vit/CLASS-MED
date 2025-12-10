# prepare_dataset.py
import os
import sys
import shutil
from pathlib import Path
import subprocess
import numpy as np
from tqdm import tqdm
from model_utils import get_model_and_transform, image_to_embedding
from db import init_db, insert_image_record
from PIL import Image
import json
from pathlib import Path

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
EMBED_PATH = DATA_DIR / "embeddings.npy"
DB_DIR = Path("sqlite_db")
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "epill.db"

def write_kaggle_json_from_env_or_secrets(kaggle_username=None, kaggle_key=None):
    """
    Cria ~/.kaggle/kaggle.json a partir de variáveis de ambiente
    (KAGGLE_USERNAME, KAGGLE_KEY) ou parâmetros passados.
    """
    user = kaggle_username or os.environ.get("KAGGLE_USERNAME")
    key = kaggle_key or os.environ.get("KAGGLE_KEY")
    if not user or not key:
        return False
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    cred = {"username": user, "key": key}
    kf = kaggle_dir / "kaggle.json"
    with open(kf, "w") as f:
        json.dump(cred, f)
    os.chmod(kf, 0o600)
    print("kaggle.json escrito em", str(kf))
    return True

def download_kaggle_dataset(slug="tommyngx/epillid-data-v1", max_images=500):
    # Tenta criar kaggle.json
    write_kaggle_json_from_env_or_secrets()
    # comando kaggle (precisa do pacote 'kaggle' instalado)
    try:
        cmd = f"kaggle datasets download -d {slug} -p {DATA_DIR} --unzip"
        print("Executando:", cmd)
        subprocess.check_call(cmd, shell=True)
    except Exception as e:
        st.error("Falha ao baixar o dataset automaticamente. Verifique o slug do Kaggle e seus secrets.")
        raise e
    # opcional: cortar o número de imagens para max_images (implemente depois em prepare_dataset)

def download_dataset_if_needed():
    # Tenta usar kaggle package para baixar dataset: substitua o slug conforme necessário
    # Exemplo slug (coloque o slug correto do ePillID_data_v1.0 no Kaggle)
    KAGGLE_SLUG = "/tommyngx/epillid-data-v1"  # <<< ATENÇÃO: substitua pelo slug real se souber
    if not IMAGES_DIR.exists():
        print("Imagens não encontradas localmente. Tentando baixar via Kaggle...")
        try:
            import kaggle
            cmd = f"kaggle datasets download -d {KAGGLE_SLUG} -p {DATA_DIR} --unzip"
            print("Executando:", cmd)
            subprocess.check_call(cmd, shell=True)
        except Exception as e:
            print("Falha ao baixar automaticamente via Kaggle. Por favor baixe manualmente e coloque as imagens em data/images.")
            print("Erro:", e)
            sys.exit(1)
    else:
        print("Imagens já presentes em data/images.")

def collect_image_paths():
    MAX_IMAGES_TO_PROCESS = int(os.environ.get("MAX_IMAGES", 500))
    image_paths = collect_image_paths()[:MAX_IMAGES_TO_PROCESS]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    for root, _, files in os.walk(IMAGES_DIR):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(str(Path(root) / f))
    return sorted(paths)

def main():
    download_dataset_if_needed()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths()
    if not image_paths:
        print("Nenhuma imagem encontrada em data/images. Coloque as imagens do dataset lá e rode novamente.")
        sys.exit(1)

    # ✅ Correção: não muda sua chamada, a função que será ajustada no model_utils
    model, transform = get_model_and_transform(device="cpu")

    embeddings = []
    init_db(DB_PATH)  # cria DB e tabela
    print(f"Processando {len(image_paths)} imagens e gerando embeddings...")

    for p in tqdm(image_paths):
        try:
            emb = image_to_embedding(p, model, transform, device="cpu")
            embeddings.append(emb)
            insert_image_record(DB_PATH, image_path=p, embedding=emb)
        except Exception as e:
            print("Erro processando", p, e)

    embeddings = np.vstack(embeddings)
    np.save(EMBED_PATH, embeddings)
    print("Embeddings salvos em", EMBED_PATH)
    print("Banco de dados criado em", DB_PATH)

if __name__ == "__main__":
    main()