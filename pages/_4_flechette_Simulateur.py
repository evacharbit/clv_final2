import streamlit as st

def run(df_personas, df_clusters):
    st.title("🎯 Simulateur de campagnes marketing")

    selected_personas = st.multiselect(
        "Sélectionnez les personas ciblés",
        df_personas["persona_id"].unique(),
        default=df_personas["persona_id"].unique()
    )

    budget = st.slider("Budget marketing (€)", 1000, 50000, 10000, step=500)
    remise = st.slider("Remise proposée (%)", 0, 50, 10)

    st.subheader("Simulation résultats")
    num_clients = df_clusters[df_clusters["persona_id"].isin(selected_personas)].shape[0]
    ca_potentiel = num_clients * remise * 10
    roi = ca_potentiel / budget if budget != 0 else 0

    st.metric("Nombre clients touchés", num_clients)
    st.metric("CA potentiel (€)", int(ca_potentiel))
    st.metric("ROI simulé", f"{roi:.2f}")
