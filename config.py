"""CONFIGURATION PROJET CLV AUCHAN"""
import os
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# -------------------------------------------------------
# 🔹 Configuration BigQuery via secrets
# -------------------------------------------------------
service_account_info = st.secrets["bigquery"]

# Création des credentials
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Client BigQuery
client = bigquery.Client(credentials=credentials, project=service_account_info["project_id"])

# Définir les variables pour les datasets
PROJECT_ID = service_account_info["project_id"]
DATASET_ENRICHIE = "dataset_reel"  # Remplace par ton dataset réel

# -------------------------------------------------------
# 🔹 Configuration générale
# -------------------------------------------------------
PHASE_DEVELOPPEMENT = False
SAMPLE_SIZE = None
N_PERSONAS = 6
RANDOM_STATE = 42
HORIZONS_ROI = {
    'court_terme': 30,
    'moyen_terme': 90,
    'long_terme': 180
}

OUTPUT_DIR = 'outputs'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'
for dir_path in [OUTPUT_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
