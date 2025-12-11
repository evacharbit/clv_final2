import streamlit as st

def run(df_personas, df_clusters, ticket):
    st.title("📈 Prévisions CA pour les clients transfrontaliers")
    st.markdown("### Visualisation des prédictions")

    st.write("---")
    # 📈 IMAGE 2 — Prévision globale
    st.subheader("🔮 Prévision du CA global")
    st.image("outputs/prediction.png")

    st.write("---")
    # 📊 IMAGE 1 — Panier Moyen
    st.subheader("🛒 Prévision du panier moyen")
    st.image("outputs/prediction_panier_moyen.png")

    st.write("---")

    st.markdown("""
    ### ℹ️ Informations
    Ces graphiques représentent les projections réalisées sur les clients **transfrontaliers**, 
    avec une estimation basée sur l’historique observé.
    """)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Prédictions Transfrontaliers",
        page_icon="📈",
        layout="wide"
    )
    run()
