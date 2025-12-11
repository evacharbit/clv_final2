import pandas as pd
import streamlit as st
from config import PROJECT_ID, DATASET_ENRICHIE, client

def load_personas_profiles():
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ENRICHIE}.personas_profiles`"
    
    # 🔍 DEBUG : Afficher la requête
    st.write(f"🔍 Debug - Requête : {query}")
    
    try:
        return client.query(query, location="EU").to_dataframe()
    except Exception as e:
        st.error(f"❌ Erreur sur personas_profiles : {e}")
        st.write(f"📋 Requête utilisée : {query}")
        raise

def load_clusters():
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ENRICHIE}.cluster`"
    
    st.write(f"🔍 Debug - Requête : {query}")
    
    try:
        return client.query(query, location="EU").to_dataframe()
    except Exception as e:
        st.error(f"❌ Erreur sur cluster : {e}")
        st.write(f"📋 Requête utilisée : {query}")
        raise

def load_ticket():
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ENRICHIE}.ticket`"
    
    st.write(f"🔍 Debug - Requête : {query}")
    
    try:
        return client.query(query, location="EU").to_dataframe()
    except Exception as e:
        st.error(f"❌ Erreur sur ticket : {e}")
        st.write(f"📋 Requête utilisée : {query}")
        raise

def load_data():
    st.info("⏳ Chargement des données BigQuery...")
    
    personas_profiles = load_personas_profiles()
    st.success("✅ personas_profiles chargé")
    
    clusters = load_clusters()
    st.success("✅ clusters chargé")
    
    ticket = load_ticket()
    st.success("✅ ticket chargé")
    
    return personas_profiles, clusters, ticket
