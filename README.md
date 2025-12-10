# ePill Streamlit Demo (classificação e imagens similares)

Resumo: Aplicação Streamlit que usa um modelo CNN pré-treinado (ResNet) para extrair embeddings de imagens do dataset ePillID_data_v1.0 e, dada uma imagem enviada pelo usuário, retorna as imagens mais similares do dataset.

## Requisitos
- Python 3.11
- Conta no Kaggle com API token (kaggle.json)

## Passos (para iniciantes)
1. Clone o repositório:
   `git clone <repo-url> && cd epill-streamlit`
2. Crie e ative venv:
   - Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
3. Instale dependências:
   `pip install -r requirements.txt`
4. Configure o Kaggle API:
   - Crie/entre no Kaggle, vá em "Account" → "Create API token" para obter `kaggle.json`.
   - Coloque `kaggle.json` em `~/.kaggle/kaggle.json` (Linux/macOS) ou `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
5. Rode o script de preparação (irá baixar dataset, extrair e calcular embeddings):
   `python prepare_dataset.py`
6. Rode o app:
   `streamlit run streamlit_app.py`
7. Acesse `http://localhost:8501` e faça upload de uma imagem para testar.

## Deploy no Streamlit Cloud
- No repositório, não comite `data/` nem o DB.
- No Streamlit Cloud, crie secrets com sua chave do Kaggle (KAGGLE_USERNAME e KAGGLE_KEY) e ajuste o arquivo `scripts/download_from_kaggle.sh` se necessário.