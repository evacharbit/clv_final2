import streamlit as st

def run():

    # Titre et description
    st.title("Dashboard Personas")
    st.markdown("Bienvenue dans le dashboard interactif des personas.")

    # Affichage du logo
    st.image("outputs/logo.png")  # Assure-toi que le chemin est correct
    
    # Tu peux ajouter d'autres éléments : texte, images, KPI, etc.
    st.markdown("""
    ### ℹ️ Informations
    Ce dashboard permet de visualiser les différents personas, leurs comportements et recommandations marketing.
    """)
