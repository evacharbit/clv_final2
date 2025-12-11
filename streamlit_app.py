import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

# -------------------------------------------------------
# 🔹 Test connexion BigQuery
# -------------------------------------------------------
service_account_info = st.secrets["bigquery"]

# Création des credentials
credentials = service_account.Credentials.from_service_account_info(service_account_info)
PROJECT_ID = service_account_info["project_id"]
DATASET_ENRICHIE = "data_enrichie"  # Vérifie que c'est exactement le nom dans BigQuery

# Création du client avec localisation EU
client = bigquery.Client(
    credentials=credentials,
    project=PROJECT_ID,
    location="EU"
)

# Lister les tables du dataset pour vérifier la connexion
try:
    tables = list(client.list_tables(DATASET_ENRICHIE))
    st.success(f"✅ Connexion OK, tables disponibles : {[t.table_id for t in tables]}")
except Exception as e:
    st.error(f"❌ Erreur de connexion BigQuery : {e}")


import streamlit as st
from utils.data_loader import load_data

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
@st.cache_data
def load_all_data():
    return load_data()

df_personas, df_clusters, ticket = load_all_data()

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
