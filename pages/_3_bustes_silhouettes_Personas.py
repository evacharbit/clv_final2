import streamlit as st
import plotly.express as px
from utils.charts import radar_persona

def run(df_profiles, df_transactions):
    st.title("🧑 Recommandations")

    # --- Construire le mapping id -> nom de persona ---
    # Si df_profiles contient déjà une colonne 'persona_name', on l'utilise.
    if "persona_name" in df_profiles.columns:
        mapping = dict(zip(df_profiles["persona_id"], df_profiles["persona_name"]))
    else:
        # Mapping par défaut — adapte les clés si tes IDs sont différents
        mapping = {
            0: "Champion",
            1: "Click&Collect",
            2: "Actifs standard",
            3: "Transfrontaliers",
            4: "Descendants"
        }

    # Options (IDs uniques). On force le type int si besoin.
    try:
        options = sorted(df_profiles["persona_id"].astype(int).unique())
    except Exception:
        options = list(df_profiles["persona_id"].unique())

    # Afficher la selectbox avec formatage : on montre le nom mais on retourne l'ID
    persona_id = st.selectbox(
        "Sélectionnez un Persona",
        options,
        format_func=lambda x: mapping.get(x, str(x)),
        key="persona_select"
    )

    # Récupérer le label (nom) pour la logique d'affichage
    persona_label = mapping.get(persona_id, str(persona_id))

    st.write("---")

    # Recommandations selon Persona (on compare sur le nom lisible)
    if persona_label == "Champion":
        st.markdown("""
        ## 🏆 Champion — Recommandations

        - ⭐ **Service client VIP** (ligne directe, chat prioritaire)  
        - 🎟️ **Expériences exclusives** (soirées VIP, ateliers cuisine)  
        - 💳 **Carte payante offerte**
        """)

    elif persona_label == "Click&Collect":
        st.markdown("""
        ## 📦 Click & Collect — Recommandations

        - ⚡ **Liste intelligente “les courses en 1 clic”**  
        - 🎯 **Push promos personnalisées**  
        - 🛡️ **Lutte anti-churn** : alertes J+60 / J+90 / J+120 avec offres progressives
        """)

    elif persona_label == "Actifs standard":
        st.markdown("""
        ## 😊 Actifs standard — Recommandations

        - 🎮 **Fidélité gamifiée** (missions, badges, challenges)  
        - 🔁 **Cross-sell intelligent** (IA, paniers types, seuils psychologiques)  
        - 📩 **Rappel d’inactivité** : email si aucun achat depuis 30 jours + idées recettes
        """)

    elif persona_label == "Transfrontaliers":
        st.markdown("""
        ## 🌍 Transfrontaliers — Recommandations

        - 🗣️ **Communication dans la langue du pays**  
        - 🚗 **Click & Collect pour produits volumineux**  
        - 🛍️ **Animation commerciale** : ex. réduction essence si panier > 80€
        """)

    elif persona_label == "Descendants":
        st.markdown("""
        ## 👨‍👩‍👧 Descendants — Recommandations

        - 📞 **Enquête sortie** : appel avec questionnaire incentivé  
        """)

    else:
        st.info("Aucune recommandation disponible pour ce persona.")

    st.write("---")

    # Bloc générique : priorités stratégiques (toujours affiché)
    st.markdown("""
    ## 🎯 Priorités stratégiques globales

    1️⃣ **Protéger les Champions** *(47% du CA)*  
    2️⃣ **Doubler la fréquence des Transfrontaliers** *(+45M€ potentiel)*  
    3️⃣ **Réduire le churn des Click & Collect** *(-23%)*  
    """)
