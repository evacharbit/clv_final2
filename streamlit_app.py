import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

# -------------------------------------------------------
# 🔹 Configuration BigQuery avec secrets Streamlit
# -------------------------------------------------------
credentials_info = st.secrets["bigquery"]
credentials = service_account.Credentials.from_service_account_info(credentials_info)
client = bigquery.Client(credentials=credentials, project=credentials_info["project_id"])

# Exemple de requête test (optionnel, tu peux adapter)
query = "SELECT * FROM `mon_dataset.ma_table` LIMIT 10"
try:
    df_test = client.query(query).to_dataframe()
    st.write("✅ Connexion BigQuery OK, aperçu des données :")
    st.dataframe(df_test)
except Exception as e:
    st.error(f"Erreur BigQuery : {e}")

# -------------------------------------------------------
# 🔹 Configuration générale de la page
# -------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Personae",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# 🔹 Chargement des données centralisé
# -------------------------------------------------------
from utils.data_loader import (
    load_personas_profiles,
    load_clusters,
    load_ticket
)

@st.cache_data
def load_data():
    df_personas = load_personas_profiles()
    df_clusters = load_clusters()
    ticket = load_ticket()
    return df_personas, df_clusters, ticket

df_personas, df_clusters, ticket = load_data()

# -------------------------------------------------------
# 🔹 Onglets principaux (navigation en haut)
# -------------------------------------------------------
tabs = [
    "🏠 Accueil",
    "😃 Team & Project",
    "🌳 Dashboard",
    "🤖 ML Process",
    "📊 Personae",
    "🧑 Recommandations",
    "💰 ROI Marketing",
    "📈 Prédictions",
    "🎯 Simulateur"
]

selected_tab = st.tabs(tabs)

# -------------------------------------------------------
# 🔹 Page Accueil
# -------------------------------------------------------
with selected_tab[0]:
    from pages import _acceuil
    _acceuil.run()

# -------------------------------------------------------
# 🔹 Page Team & Project
# -------------------------------------------------------
with selected_tab[1]:
    from pages import _0_team
    _0_team.run()

# -------------------------------------------------------
# 🔹 Page Dashboard
# -------------------------------------------------------
with selected_tab[2]:
    from pages import _7_contexte
    _7_contexte.run()

# -------------------------------------------------------
# 🔹 Page ML Process
# -------------------------------------------------------
with selected_tab[3]:
    from pages import _1_ML_Process
    _1_ML_Process.run(df_personas)

# -------------------------------------------------------
# 🔹 Page Personae
# -------------------------------------------------------
with selected_tab[4]:
    from pages import _2_histogramme_Vue_Ensemble
    _2_histogramme_Vue_Ensemble.run(df_personas, df_clusters)

# -------------------------------------------------------
# 🔹 Page Recommandations
# -------------------------------------------------------
with selected_tab[5]:
    from pages import _3_bustes_silhouettes_Personas
    _3_bustes_silhouettes_Personas.run(df_personas, df_clusters)

# -------------------------------------------------------
# 🔹 Page ROI Marketing
# -------------------------------------------------------
with selected_tab[6]:
    from pages import _5_ROI_Marketing
    _5_ROI_Marketing.run(df_personas, df_clusters)

# -------------------------------------------------------
# 🔹 Page Prédictions Transfrontaliers
# -------------------------------------------------------
with selected_tab[7]:
    from pages import _6_Predictions
    _6_Predictions.run(df_personas, df_clusters, ticket)

# -------------------------------------------------------
# 🔹 Page Simulateur
# -------------------------------------------------------
with selected_tab[8]:
    from pages import _4_flechette_Simulateur
    _4_flechette_Simulateur.run(df_personas, df_clusters)
