"""CONFIGURATION PROJET CLV AUCHAN"""
import os
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

print("="*70)
print(" CONFIGURATION BIGQUERY")
print("="*70)
print(f"\n:dossier: Répertoire : {os.getcwd()}")

# -------------------------------------------------------
# 🔹 Configuration BigQuery via secrets Streamlit
# -------------------------------------------------------
service_account_info = st.secrets["bigquery"]

# Création des credentials
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Variables projet / dataset
PROJECT_ID = service_account_info["project_id"]
DATASET_ENRICHIE = "data_enrichie"  # Nom exact du dataset dans BigQuery

# Client BigQuery avec localisation EU
client = bigquery.Client(
    credentials=credentials,
    project=PROJECT_ID,
    location="EU"
)

# Vérification tables disponibles (optionnel)
try:
    tables = list(client.list_tables(DATASET_ENRICHIE))
    print("Tables disponibles :", [t.table_id for t in tables])
except Exception as e:
    print("⚠️ Attention, impossible de lister les tables :", e)

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

print(f"\n:coche_blanche: Configuration OK")
print("="*70)
