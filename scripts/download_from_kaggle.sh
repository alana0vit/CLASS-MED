#!/bin/bash
# Uso: scripts/download_from_kaggle.sh
mkdir -p data
# exemplo: o nome do dataset pode variar; substitua 'username/dataset-name' se necessário
kaggle datasets download -d <KAGGLE_DATASET_SLUG> -p data/ --unzip