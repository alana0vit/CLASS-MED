#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python prepare_dataset.py
streamlit run streamlit_app.py