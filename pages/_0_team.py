"""
PAGE TEAM - PRÉSENTATION DU PROJET ET DE L'ÉQUIPE
"""

import streamlit as st
from PIL import Image
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_NAME = "Et si chaque euro investi devenait plus rentable ? "

PROBLEMATIQUE = """
Comment maximiser la rentabilité des investissements marketing en concentrant les efforts sur les clients à plus forte valeur ajoutée (CLV) sur le long terme ?
"""

CONTEXTE = """
Nous dépensons des millions en marketing chaque année, mais traitons tous ses clients 
de la même manière. Résultat : gaspillage budgétaire, perte de clients VIP, 
et ROI impossible à mesurer.

Notre solution : Un système de segmentation intelligente basé sur le Machine Learning 
qui identifie 5 personae distincts et recommande des stratégies marketing personnalisées 
pour chacun.
"""

RESULTATS = {
    'clients_analyses': '440 414',
    'features_utilisees': '15',
    'personas_detectes': '5',
    'silhouette_score': '0.38',
    'roi_attendu': '10.39x'
}

TEAM_MEMBERS = [
    {'nom': 'Virginia', 'photo': 'virginia.png', 'linkedin': 'https://www.linkedin.com/in/virginia-taisne/', 'role':'Data Scientist', 'description':'Expert ML et segmentation'},
    {'nom': 'Bérénice', 'photo': 'berenice.png', 'linkedin': 'https://www.linkedin.com/in/b%C3%A9r%C3%A9nice-lebleu-37b725251/', 'role':'Data Analyst', 'description':'Analyse comportement client'},
    {'nom': 'Blair', 'photo': 'blair.png', 'linkedin': 'https://www.linkedin.com/in/blair-mbibe/', 'role':'ML Engineer', 'description':'Déploiement modèles'},
    {'nom': 'Lens', 'photo': 'lens.png', 'linkedin': 'https://www.linkedin.com/in/echarbit?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app', 'role':'Product Owner', 'description':'Coordination & priorisation'},
    {'nom': 'Eva', 'photo': 'eva.png', 'linkedin': 'https://www.linkedin.com/in/len-s-guello-47258624b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app', 'role':'Business Analyst', 'description':'KPIs et reporting'}
]

# ============================================================================
# FONCTIONS
# ============================================================================

def load_team_photo(filename):
    """Charge une photo d'équipe, renvoie None si manquante"""
    try:
        photo_path = os.path.join('outputs', filename)
        if os.path.exists(photo_path):
            return Image.open(photo_path)
        return None
    except:
        return None

# ============================================================================
# PAGE PRINCIPALE
# ============================================================================

def run():
    # Titre à gauche, logo à droite
    col_title, col_logo = st.columns([5, 1])
    with col_title:
        st.markdown(f"# {PROJECT_NAME}")
    with col_logo:
        st.image("outputs/logo.png", width=120)  # logo plus gros à droite

    st.markdown("## :dart: La Problématique")
    st.info(PROBLEMATIQUE)

    st.markdown("### :bar_chart: Le Contexte")
    st.info(CONTEXTE)

    st.markdown("## :rocket: Résultats Clés du Projet")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Clients Analysés", RESULTATS['clients_analyses'])
    with col2:
        st.metric("Personas Détectés", RESULTATS['personas_detectes'])
    with col3:
        st.metric("ROI Attendu", RESULTATS['roi_attendu'])
    with col4:
         st.metric("Features Utilisées", RESULTATS['features_utilisees'])
    with col5:
        st.metric("Silhouette Score", RESULTATS['silhouette_score'])

    st.markdown("---")
    st.markdown("## :busts_in_silhouette: Notre Équipe - Le Wagon Batch #2025")
    
    cols = st.columns(5)
    for idx, member in enumerate(TEAM_MEMBERS):
        with cols[idx]:
            photo = load_team_photo(member['photo'])
            if photo:
                st.image(photo)
            else:
                initiales = ''.join([w[0] for w in member['nom'].split()])
                st.markdown(f"""
                <div style='width: 100%; padding-top: 100%; position: relative; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border-radius: 50%; margin-bottom: 15px;'>
                    <div style='position: absolute; top: 50%; left: 50%; 
                                transform: translate(-50%, -50%);
                                color: white; font-size: 3em; font-weight: bold;'>
                        {initiales}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"### {member['nom']}")
            st.markdown(f"**{member['role']}**")
            st.markdown(member['description'])
            if member['linkedin']:
                st.markdown(f"[LinkedIn]({member['linkedin']})")

    st.markdown("---")
    st.markdown("## :hammer_and_wrench: Stack Technique")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Data & ML:**\n- Python 3.12\n- Pandas & NumPy\n- Scikit-learn (K-Means, PCA)\n- Prophet\n- Google BigQuery")
    with col2:
        st.markdown("**Visualisation:**\n- Streamlit\n- Plotly\n- Seaborn & Matplotlib")
    with col3:
        st.markdown("**DevOps & Tools:**\n- Git & GitHub\n- Docker\n- GCP\n- VS Code & Jupyter\n- Poetry")

    st.markdown("---")
    st.markdown("## :microscope: Méthodologie")
    st.markdown("""
    1️⃣ Collecte & Nettoyage  
    2️⃣ Feature Engineering  
    3️⃣ Clustering K-Means  
    4️⃣ Profiling & Actions  
    5️⃣ Dashboard & Démo
    """)

    st.markdown("---")
    st.markdown("### Footer")
    st.markdown("Projet réalisé dans le cadre du Bootcamp Data Science du Wagon - Batch 2025")


# ============================================================================
# EXÉCUTION STANDALONE
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="CLV - BOOSTER", page_icon=":busts_in_silhouette:", layout="wide")
    run()
