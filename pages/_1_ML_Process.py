import streamlit as st
import plotly.express as px

def run(df_personas, df_clusters=None):
    st.title("🤖 Méthode de création des personae")
    st.markdown("Cette page présente le processus de clustering utilisé pour générer les personae.")

    # Graphique en barres avec les noms, sans légende et sans titre axe x
    st.image("outputs/elbow_method.png", caption =" Elbow method et K-means" )
    st.image("outputs/heatmap_placeholder.png", caption="Heatmap des caractéristiques par persona")
