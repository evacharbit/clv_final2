"""
PAGE VUE D'ENSEMBLE - CARTES PERSONAS AVEC KPI PERSONNALISÉS + KPI GLOBAUX
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from google.cloud import bigquery
from connection import PROJECT_ID

def run(df_personas, df_clusters):
    st.title("👥 Vue d'ensemble des Personae")
    st.markdown("### Cartes d'identité détaillées des 5 segments clients")
    
    # ========================================================================
    # CONFIGURATION DES PERSONAS
    # ========================================================================
    
    persona_config = {
        0: {
            'name': '🚗 Click and Collect',
            'emoji': '🚗',
            'color': '#3498DB',
            'slogan': 'Praticité avant tout',
            'badges': ['⚡️ Rapide', '📱 Digital', '🏃‍♀️‍➡️ Pressé'],
            'kpi_specifiques': [
                {'label': '🚗 Taux Drive', 'col': 'pct_drive', 'format': 'pct', 'icon': '🚗'},
                {'label': '🚨 Taux Churn', 'value': '15', 'format': 'pct', 'icon': '💤'}
            ]
        },
        1: {
            'name': '👑 Les champions',
            'emoji': '👑',
            'color': '#FFD700',
            'slogan': 'Les piliers du CA',
            'badges': ['🥇 Premium', '💰 Haute valeur', '❤️ Fidèle'],
            'kpi_specifiques': [
                {'label': '💎 CLV Moyenne', 'col': 'clv_moyenne', 'format': 'euro', 'icon': '💰'},
                {'label': '👜 Nb Transactions Moy', 'col': 'nb_transactions_moyen', 'format': 'nombre', 'icon': '📊'}
            ]
        },
        2: {
            'name': '🚶‍♂️ Actifs Standards',
            'emoji': '🚶‍♂️',
            'color': '#95A5A6',
            'slogan': 'Le cœur de cible',
            'badges': ['🕐 Régulier', '⚖️ Équilibré', '🔄 Potentiel'],
            'kpi_specifiques': [
                {'label': '📅 Récence Moyenne', 'col': 'recency_moyenne', 'format': 'jours', 'icon': '⏱️'},
                {'label': '💶 Taux Remise Moyen', 'col': 'pct_remise_moyen', 'format': 'pct', 'icon': '🔖'}
            ]
        },
        3: {
            'name': '🇧🇪 Transfrontaliers',
            'emoji': '🇧🇪',
            'color': '#E74C3C',
            'slogan': 'Gros paniers, faible fréquence',
            'badges': ['💶 Gros panier', '🚗 Frontaliers', '🛄 Mensuel'],
            'kpi_specifiques': [
                {'label': '🇧🇪 % Belges', 'value': '75', 'format': 'pct', 'icon': '🌍'},
                {'label': 'Nombre de produits', 'value': '59', 'format': 'nombre', 'icon': '📦'}
            ]
        },
        4: {
            'name': '😴 Descendants',
            'emoji': '😴',
            'color': '#7F8C8D',
            'slogan': 'À réactiver',
            'badges': ['⏰ Inactifs', '🔔 Alerte', '🕐 Win-back'],
            'kpi_specifiques': [
                {'label': '😴 % Churned', 'value': '35', 'format': 'pct', 'icon': '🚨'},
                {'label': '📆 Récence Moyenne', 'col': 'recency_moyenne', 'format': 'jours', 'icon': '⏳'}
            ]
        }
    }
    
    # ========================================================================
    # FONCTIONS UTILITAIRES
    # ========================================================================
    
    def format_value(value, format_type):
        """Formate une valeur selon son type"""
        try:
            if pd.isna(value) or value == 'N/A':
                return 'N/A'
            
            if format_type == 'euro':
                return f"{float(value):,.0f}€".replace(',', ' ')
            elif format_type == 'pct':
                # ✅ CORRECTION : Multiplier par 100 si valeur < 1
                val = float(value)
                if val < 1:
                    val = val * 100
                return f"{val:.1f}%"
            elif format_type == 'nombre':
                return f"{float(value):,.1f}".replace(',', ' ')
            elif format_type == 'jours':
                return f"{int(value)} jours"
            elif format_type == 'freq':
                return f"{float(value):.2f}/mois"
            else:
                return str(value)
        except:
            return 'N/A'
    
    def safe_get(df_row, col, default='N/A'):
        """Récupère une valeur de manière sécurisée"""
        try:
            if col in df_row.index:
                val = df_row[col]
                return val if not pd.isna(val) else default
            return default
        except:
            return default
    
    @st.cache_data(ttl=3600)
    def load_kpi_globaux():
        """Charge les KPI globaux depuis la table ticket"""
        client = bigquery.Client(project=PROJECT_ID)
        
        query = f"""
        WITH monthly_data AS (
            SELECT 
                customer_id,
                DATE_TRUNC(DATE(transaction_date), MONTH) as mois,
                SUM(total_amount) as ca_mensuel,
                AVG(total_amount) as panier_moyen_mensuel,
                COUNT(DISTINCT transaction_id) as nb_transactions
            FROM `{PROJECT_ID}.data_enrichie.ticket`
            WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
            GROUP BY customer_id, mois
        ),
        client_stats AS (
            SELECT
                customer_id,
                AVG(ca_mensuel) as ca_moyen_mensuel,
                AVG(panier_moyen_mensuel) as panier_moyen,
                AVG(nb_transactions) as frequence_mensuelle
            FROM monthly_data
            GROUP BY customer_id
        )
        SELECT
            AVG(ca_moyen_mensuel) as ca_moyen_mensuel_global,
            AVG(panier_moyen) as panier_moyen_global,
            AVG(frequence_mensuelle) as frequence_mensuelle_globale,
            COUNT(DISTINCT customer_id) as nb_clients_total
        FROM client_stats
        """
        
        try:
            df = client.query(query).to_dataframe()
            if len(df) > 0:
                return df.iloc[0].to_dict()
            else:
                return None
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des KPI globaux : {e}")
            return None
    
    # ========================================================================
    # VUE D'ENSEMBLE RAPIDE
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📊 Répartition Globale des Personae")
    
    cols = st.columns(5)
    
    for idx, (persona_id, config) in enumerate(persona_config.items()):
        if persona_id in df_personas["persona_id"].values:
            df_p = df_personas[df_personas["persona_id"] == persona_id].iloc[0]
            
            with cols[idx]:
                taille = safe_get(df_p, 'taille', 0)
                pct = safe_get(df_p, 'pct_base', 0)
                
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; 
                            background: linear-gradient(135deg, {config['color']}20, {config['color']}40); 
                            border-radius: 15px; 
                            border: 2px solid {config['color']};
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
                    <div style='font-size: 3.5em; margin-bottom: 10px;'>{config['emoji']}</div>
                    <div style='font-size: 2em; font-weight: bold; color: {config['color']};'>
                        {pct:.1f}%
                    </div>
                    <div style='font-size: 1em; color: #555; margin-top: 5px;'>
                        {taille:,} clients
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================================================
    # KPI GLOBAUX (NOUVEAUTÉ)
    # ========================================================================
    
    st.subheader("📈 Indicateurs Globaux (Ensemble des Clients)")
    
    with st.spinner('📊 Chargement des KPI globaux...'):
        kpi_globaux = load_kpi_globaux()
    
    if kpi_globaux:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ca_mensuel = kpi_globaux.get('ca_moyen_mensuel_global', 0)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; border-radius: 12px; text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
                <div style='font-size: 3em; margin-bottom: 10px;'>💰</div>
                <div style='font-size: 2.2em; font-weight: bold; color: white;'>
                    {ca_mensuel:,.0f}€
                </div>
                <div style='color: white; margin-top: 10px; font-size: 1em; opacity: 0.9;'>
                    CA Moyen Mensuel/Client
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            panier = kpi_globaux.get('panier_moyen_global', 0)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; border-radius: 12px; text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
                <div style='font-size: 3em; margin-bottom: 10px;'>🛒</div>
                <div style='font-size: 2.2em; font-weight: bold; color: white;'>
                    {panier:,.0f}€
                </div>
                <div style='color: white; margin-top: 10px; font-size: 1em; opacity: 0.9;'>
                    Panier Moyen Mensuel
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            freq = kpi_globaux.get('frequence_mensuelle_globale', 0)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; border-radius: 12px; text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
                <div style='font-size: 3em; margin-bottom: 10px;'>📅</div>
                <div style='font-size: 2.2em; font-weight: bold; color: white;'>
                    {freq:.2f}
                </div>
                <div style='color: white; margin-top: 10px; font-size: 1em; opacity: 0.9;'>
                    Fréquence Mensuelle (visites)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            nb = '440'
            icon = "🕺"
            color = "#764ba2"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color}80, {color}cc); 
                        padding: 25px; border-radius: 12px; text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
                <div style='font-size: 3em; margin-bottom: 10px;'>{icon}</div>
                <div style='font-size: 2.2em; font-weight: bold; color: white;'>
                    {nb}K
                </div>
                <div style='color: white; margin-top: 10px; font-size: 1em; opacity: 0.9;'>
                    Nombre de clients
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================================================
    # CARTES DÉTAILLÉES PAR PERSONA
    # ========================================================================
    
    st.subheader("🎯 Fiches Détaillées par Persona")
    st.markdown("<br>", unsafe_allow_html=True)
    
    for persona_id in sorted(df_personas["persona_id"].unique()):
        df_p = df_personas[df_personas["persona_id"] == persona_id].iloc[0]
        config = persona_config[persona_id]
        
        # ====================================================================
        # CONTAINER PRINCIPAL
        # ====================================================================
        
        with st.container():
            
            # ================================================================
            # EN-TÊTE DU PERSONA
            # ================================================================
            
            col_header1, col_header2 = st.columns([1, 5])
            
            with col_header1:
                # Grande icône avec cercle coloré
                st.markdown(f"""
                <div style='width: 130px; height: 130px; 
                            background: linear-gradient(135deg, {config['color']}60, {config['color']}90);
                            border-radius: 50%; 
                            display: flex; align-items: center; justify-content: center;
                            border: 4px solid {config['color']};
                            box-shadow: 0 6px 15px rgba(0,0,0,0.25);
                            margin: 0 auto;'>
                    <span style='font-size: 5em;'>{config['emoji']}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_header2:
                # Nom du persona
                st.markdown(f"""
                <h2 style='margin: 0 0 10px 0; color: {config['color']}; font-size: 2.5em;'>
                    {config['name']}
                </h2>
                <p style='font-size: 1.3em; color: #666; font-style: italic; margin: 5px 0 15px 0;'>
                    "{config['slogan']}"
                </p>
                """, unsafe_allow_html=True)
                
                # Badges
                badges_html = " ".join([
                    f"<span style='background: {config['color']}; color: white; "
                    f"padding: 8px 16px; border-radius: 20px; margin-right: 10px; "
                    f"font-size: 0.95em; font-weight: bold; display: inline-block; "
                    f"box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>{badge}</span>"
                    for badge in config['badges']
                ])
                st.markdown(f"<div style='margin-top: 15px;'>{badges_html}</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ================================================================
            # KPI PRINCIPAUX (COMMUNS À TOUS)
            # ================================================================
            
            st.markdown(f"""
            <h3 style='color: {config['color']}; margin: 30px 0 20px 0;'>
                📊 Indicateurs Principaux
            </h3>
            """, unsafe_allow_html=True)
            
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            # KPI 1 : Taille du cluster
            with kpi_col1:
                taille = safe_get(df_p, 'taille', 0)
                st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; 
                            text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                            border-top: 4px solid {config['color']};'>
                    <div style='font-size: 2.5em; margin-bottom: 10px;'>👤</div>
                    <div style='font-size: 1.8em; font-weight: bold; color: {config['color']};'>
                        {taille:,}
                    </div>
                    <div style='color: #666; margin-top: 8px; font-size: 0.95em;'>
                        Clients
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # KPI 2 : CA moyen mensuel ✅ CALCULÉ = Panier × Fréquence
            with kpi_col2:
                panier = float(safe_get(df_p, 'panier_moyen', 0))
                frequence = float(safe_get(df_p, 'frequence_mois', 0))
                
                # ✅ CORRECTION : CA mensuel = Panier × Fréquence
                ca_mensuel_persona = panier * frequence
                
                # Comparaison vs global
                if kpi_globaux:
                    delta_pct = ((ca_mensuel_persona / kpi_globaux['ca_moyen_mensuel_global']) - 1) * 100 if kpi_globaux['ca_moyen_mensuel_global'] > 0 else 0
                    delta_color = config['color'] if delta_pct > 0 else '#e74c3c'
                    delta_text = f"<div style='font-size: 0.9em; color: {delta_color}; margin-top: 5px;'>{'▲' if delta_pct > 0 else '▼'} {abs(delta_pct):.0f}% vs global</div>"
                else:
                    delta_text = ""
                
                st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; 
                            text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                            border-top: 4px solid {config['color']};'>
                    <div style='font-size: 2.5em; margin-bottom: 10px;'>💰</div>
                    <div style='font-size: 1.8em; font-weight: bold; color: {config['color']};'>
                        {ca_mensuel_persona:,.0f}€
                    </div>
                    <div style='color: #666; margin-top: 8px; font-size: 0.95em;'>
                        CA Moyen/Mois
                    </div>
                    {delta_text}
                </div>
                """, unsafe_allow_html=True)
            
            # KPI 3 : Panier moyen
            with kpi_col3:
                # Comparaison vs global
                if kpi_globaux:
                    delta_pct = ((panier / kpi_globaux['panier_moyen_global']) - 1) * 100 if kpi_globaux['panier_moyen_global'] > 0 else 0
                    delta_color = config['color'] if delta_pct > 0 else '#e74c3c'
                    delta_text = f"<div style='font-size: 0.9em; color: {delta_color}; margin-top: 5px;'>{'▲' if delta_pct > 0 else '▼'} {abs(delta_pct):.0f}% vs global</div>"
                else:
                    delta_text = ""
                
                st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; 
                            text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                            border-top: 4px solid {config['color']};'>
                    <div style='font-size: 2.5em; margin-bottom: 10px;'>🛒</div>
                    <div style='font-size: 1.8em; font-weight: bold; color: {config['color']};'>
                        {panier:,.0f}€
                    </div>
                    <div style='color: #666; margin-top: 8px; font-size: 0.95em;'>
                        Panier Moyen
                    </div>
                    {delta_text}
                </div>
                """, unsafe_allow_html=True)
            
            # KPI 4 : Fréquence d'achat mensuelle
            with kpi_col4:
                # Comparaison vs global
                if kpi_globaux:
                    delta_pct = ((frequence / kpi_globaux['frequence_mensuelle_globale']) - 1) * 100 if kpi_globaux['frequence_mensuelle_globale'] > 0 else 0
                    delta_color = config['color'] if delta_pct > 0 else '#e74c3c'
                    delta_text = f"<div style='font-size: 0.9em; color: {delta_color}; margin-top: 5px;'>{'▲' if delta_pct > 0 else '▼'} {abs(delta_pct):.0f}% vs global</div>"
                else:
                    delta_text = ""
                
                st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; 
                            text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                            border-top: 4px solid {config['color']};'>
                    <div style='font-size: 2.5em; margin-bottom: 10px;'>📅</div>
                    <div style='font-size: 1.8em; font-weight: bold; color: {config['color']};'>
                        {frequence:.2f}/mois
                    </div>
                    <div style='color: #666; margin-top: 8px; font-size: 0.95em;'>
                        Fréquence d'Achat
                    </div>
                    {delta_text}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ================================================================
            # KPI SPÉCIFIQUES AU PERSONA - CENTRÉS
            # ================================================================
            
            st.markdown(f"""
            <h3 style='color: {config['color']}; margin: 30px 0 20px 0;'>
                🎯 Indicateurs Spécifiques
            </h3>
            """, unsafe_allow_html=True)
            
            # ✅ CENTRER : Utiliser columns avec padding
            nb_kpi_spec = len(config['kpi_specifiques']) + 1  # +1 pour secteur
            
            # Créer colonnes avec espaces pour centrer
            col_layout = [0.5] + [1] * nb_kpi_spec + [0.5]
            kpi_spec_cols = st.columns(col_layout)
            
            for idx, kpi_spec in enumerate(config['kpi_specifiques']):
                with kpi_spec_cols[idx + 1]:  # +1 pour skip la première colonne vide
                    # ✅ Gestion 'col' vs 'value'
                    if 'value' in kpi_spec:
                        valeur = kpi_spec['value']
                    else:
                        valeur = safe_get(df_p, kpi_spec['col'], 'N/A')
                    
                    valeur_formatee = format_value(valeur, kpi_spec['format'])
                    
                    st.markdown(f"""
                    <div style='background: {config['color']}15; padding: 20px; border-radius: 10px; 
                                text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.08);
                                border: 2px solid {config['color']}40;'>
                        <div style='font-size: 2.5em; margin-bottom: 10px;'>{kpi_spec['icon']}</div>
                        <div style='font-size: 1.6em; font-weight: bold; color: {config['color']};'>
                            {valeur_formatee}
                        </div>
                        <div style='color: #555; margin-top: 8px; font-size: 0.9em;'>
                            {kpi_spec['label']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Secteur principal
            with kpi_spec_cols[-2]:  # -2 pour skip la dernière colonne vide
                secteur = safe_get(df_p, 'secteur_principal', 'N/A')
                st.markdown(f"""
                <div style='background: {config['color']}15; padding: 20px; border-radius: 10px; 
                            text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.08);
                            border: 2px solid {config['color']}40;'>
                    <div style='font-size: 2.5em; margin-bottom: 10px;'>🏪</div>
                    <div style='font-size: 1.2em; font-weight: bold; color: {config['color']};'>
                        {secteur if len(str(secteur)) < 15 else str(secteur)[:12] + '...'}
                    </div>
                    <div style='color: #555; margin-top: 8px; font-size: 0.9em;'>
                        Secteur Principal
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
    
    # ========================================================================
    # FOOTER / RÉSUMÉ
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: #f8f9fa; border-radius: 10px;'>
        <h3 style='color: #667eea;'>💡 Insights Clés</h3>
        <p style='font-size: 1.1em; color: #555;'>
            5 personae distincts identifiés par Machine Learning<br>
            Chaque segment a des caractéristiques et besoins uniques<br>
            Stratégies marketing personnalisées pour maximiser le ROI
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# EXÉCUTION STANDALONE
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Vue d'ensemble Personas",
        page_icon="👥",
        layout="wide"
    )
    run(None, None)