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
# 🔹 TEST CONNEXION BIGQUERY (DEBUG)
# -------------------------------------------------------
st.write("## 🔍 Debug - Connexion BigQuery")

from config import PROJECT_ID, DATASET_ENRICHIE, client

st.write(f"- **Project ID** : `{PROJECT_ID}`")
st.write(f"- **Dataset** : `{DATASET_ENRICHIE}`")

try:
    tables = list(client.list_tables(DATASET_ENRICHIE))
    table_names = [t.table_id for t in tables]
    st.success(f"✅ Dataset trouvé avec {len(table_names)} tables")
    st.write("**Tables disponibles** :")
    for name in table_names:
        st.write(f"  - `{name}`")
except Exception as e:
    st.error(f"❌ Impossible de lister les tables : {e}")
    st.stop()

st.write("---")

# -------------------------------------------------------
# 🔹 Chargement des données centralisé
# -------------------------------------------------------
st.write("## 📊 Chargement des données")

# @st.cache_data  # ⚠️ Désactivé temporairement pour debug
def load_all_data():
    return load_data()

try:
    df_personas, df_clusters, ticket = load_all_data()
    st.success("✅ Toutes les données chargées avec succès")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

st.write("---")

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

# ... reste du code (pages) ...
```

## 🎯 Ce que ça va nous montrer

Une fois déployé, vous verrez **en haut de la page** :

1. ✅ Le Project ID exact
2. ✅ Le nom du dataset
3. ✅ **La liste complète des tables disponibles**
4. ✅ Les messages de debug de `data_loader.py`

## 🔍 Scénarios possibles

### Scénario A : Les tables ont des noms différents
```
Tables disponibles :
  - personas_profile (sans 's')
  - clusters (avec 's')
  - tickets (avec 's')
```
→ Il faudra corriger les noms dans `data_loader.py`

### Scénario B : Le dataset n'existe pas
```
❌ Impossible de lister les tables : 404 Dataset XXX not found
```
→ Le nom du dataset dans `config.py` est incorrect

### Scénario C : Problème de permissions
```
❌ Impossible de lister les tables : 403 Permission denied

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
