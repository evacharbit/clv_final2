"""
PAGE ROI MARKETING - VERSION BUDGET FIXE
Simulation : Si on investit X€ par persona, combien de CA généré ?
Basé sur les vraies données de BigQuery
"""


import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def run(df_personas, df_clusters):
    
    st.title("💰 ROI Marketing")
    st.markdown("### Simulation : Impact d'un investissement fixe par persona")
    
    # ========================================================================
    # CONFIGURATION DES PERSONAS
    # ========================================================================
    
    persona_config = {
        0: {
            'nom': '🚗 Click & Collect',
            'nom_court': 'Click & Collect',
            'color': '#3498DB',
            'priorite': 2,
            'priorite_label': '⭐⭐ Importante',
            'taux_conversion': 0.12,  # 12% de conversion
            'cout_campagne': 0.50     # 0.50€ par campagne
        },
        1: {
            'nom': '👑 Champions VIP',
            'nom_court': 'Champions VIP',
            'color': '#FFD700',
            'priorite': 3,
            'priorite_label': '🔥🔥🔥 Critique',
            'taux_conversion': 0.28,  # 28% de conversion
            'cout_campagne': 1.50     # 1.50€ par campagne (premium)
        },
        2: {
            'nom': '💼 Actifs Standards',
            'nom_court': 'Actifs Standards',
            'color': '#95A5A6',
            'priorite': 2,
            'priorite_label': '⭐⭐ Importante',
            'taux_conversion': 0.10,  # 10% de conversion
            'cout_campagne': 0.40     # 0.40€ par campagne
        },
        3: {
            'nom': '🇧🇪 Transfrontaliers',
            'nom_court': 'Transfrontaliers',
            'color': '#E74C3C',
            'priorite': 3,
            'priorite_label': '🔥🔥 Stratégique',
            'taux_conversion': 0.18,  # 18% de conversion
            'cout_campagne': 0.80     # 0.80€ par campagne
        },
        4: {
            'nom': '😴 Descendants',
            'nom_court': 'Descendants',
            'color': '#7F8C8D',
            'priorite': 1,
            'priorite_label': '⬇️ Basse',
            'taux_conversion': 0.05,  # 5% de conversion (faible)
            'cout_campagne': 0.30     # 0.30€ par campagne
        }
    }
    
    # ========================================================================
    # SIDEBAR : PARAMÈTRES DE SIMULATION
    # ========================================================================
    
    st.sidebar.markdown("## 💰 Budget de Simulation")
    st.sidebar.markdown("---")
    
    # Budget par persona
    budget_par_persona = st.sidebar.slider(
        "💶 Budget par Persona",
        min_value=10000,
        max_value=500000,
        value=100000,
        step=10000,
        help="Budget marketing à investir sur chaque persona"
    )
    
    st.sidebar.markdown("---")
    
    # Filtre personas
    st.sidebar.markdown("### 🎯 Personas à comparer")
    
    personas_selectionnes = []
    for pid, config in persona_config.items():
        if pid in df_personas['persona_id'].values:
            if st.sidebar.checkbox(config['nom'], value=True, key=f"persona_{pid}"):
                personas_selectionnes.append(pid)
    
    st.sidebar.markdown("---")
    
    # Niveau de détail
    niveau_detail = st.sidebar.radio(
        "📊 Affichage",
        ["Synthèse", "Détaillé"],
        help="Synthèse = KPI principaux, Détaillé = calculs complets"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    💡 **Simulation :**
    
    Budget : **{budget_par_persona:,}€** / persona
    
    **Calcul :**
    - Nb campagnes = Budget / Coût campagne
    - Conversions = Campagnes × Taux conversion
    - CA = Conversions × Panier moyen
    - ROI = CA / Budget
    """)
    
    # ========================================================================
    # CALCUL DES SIMULATIONS (BASÉ SUR VRAIES DONNÉES)
    # ========================================================================
    
    simulations = []
    
    for pid in personas_selectionnes:
        config = persona_config[pid]
        
        if pid in df_personas['persona_id'].values:
            df_p = df_personas[df_personas['persona_id'] == pid].iloc[0]
            
            # ✅ DONNÉES RÉELLES DEPUIS BIGQUERY
            taille = int(df_p.get('taille', 0))
            ca_total_actuel = float(df_p.get('ca_total', 0))
            panier_moyen_reel = float(df_p.get('panier_moyen', 0))  # ✅ Panier réel
            
            # ✅ CALCULS SIMULATION
            nb_campagnes_possibles = int(budget_par_persona / config['cout_campagne'])
            nb_clients_touches = min(nb_campagnes_possibles, taille)  # Max = taille du persona
            
            nb_conversions = int(nb_clients_touches * config['taux_conversion'])
            ca_genere = nb_conversions * panier_moyen_reel  # ✅ Utilise panier réel
            
            roi = (ca_genere / budget_par_persona) if budget_par_persona > 0 else 0
            
            # Rentabilité
            benefice = ca_genere - budget_par_persona
            rentable = benefice > 0
            
            simulations.append({
                'persona_id': pid,
                'nom': config['nom'],
                'nom_court': config['nom_court'],
                'color': config['color'],
                'priorite': config['priorite'],
                'priorite_label': config['priorite_label'],
                'taille': taille,
                'ca_actuel': ca_total_actuel,
                'panier_moyen': panier_moyen_reel,  # ✅ Panier réel
                'budget': budget_par_persona,
                'cout_campagne': config['cout_campagne'],
                'taux_conversion': config['taux_conversion'],
                'nb_campagnes': nb_campagnes_possibles,
                'nb_clients_touches': nb_clients_touches,
                'nb_conversions': nb_conversions,
                'ca_genere': ca_genere,
                'roi': roi,
                'benefice': benefice,
                'rentable': rentable
            })
    
    df_sim = pd.DataFrame(simulations).sort_values('roi', ascending=False)
    
    if len(df_sim) == 0:
        st.warning("⚠️ Sélectionnez au moins un persona dans la sidebar")
        st.stop()
    
    # ========================================================================
    # VUE D'ENSEMBLE
    # ========================================================================
    
    st.markdown("---")
    st.subheader(f"📊 Simulation : {budget_par_persona:,}€ investis par Persona")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        budget_total = df_sim['budget'].sum()
        st.metric(
            "💶 Budget Total",
            f"{budget_total/1000:.0f}K€",
            help=f"{len(df_sim)} personas × {budget_par_persona:,}€"
        )
    
    with col2:
        ca_total_genere = df_sim['ca_genere'].sum()
        st.metric(
            "💸 CA Total Généré",
            f"{ca_total_genere/1000000:.2f}M€",
            delta=f"+{((ca_total_genere/budget_total - 1)*100):.0f}%" if budget_total > 0 else "0%"
        )
    
    with col3:
        roi_moyen = (ca_total_genere / budget_total) if budget_total > 0 else 0
        delta_color = "normal" if roi_moyen > 1.5 else "inverse"
        st.metric(
            "📈 ROI Moyen",
            f"{roi_moyen:.2f}x",
            delta="✅ Rentable" if roi_moyen > 1.5 else "⚠️ Faible",
            delta_color=delta_color
        )
    
    with col4:
        benefice_total = df_sim['benefice'].sum()
        color = "normal" if benefice_total > 0 else "inverse"
        st.metric(
            "💰 Bénéfice Total",
            f"{benefice_total/1000:.0f}K€",
            delta="✅" if benefice_total > 0 else "❌",
            delta_color=color
        )
    
    st.markdown("---")
    
    # ========================================================================
    # GRAPHIQUES PRINCIPAUX
    # ========================================================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar Chart : ROI par persona
        fig_roi = go.Figure()
        
        for _, row in df_sim.iterrows():
            fig_roi.add_trace(go.Bar(
                x=[row['nom_court']],
                y=[row['roi']],
                name=row['nom_court'],
                marker_color=row['color'],
                text=[f"{row['roi']:.2f}x"],
                textposition='outside',
                hovertemplate=f"<b>{row['nom']}</b><br>" +
                             f"Budget: {row['budget']:,}€<br>" +
                             f"CA généré: {row['ca_genere']:,.0f}€<br>" +
                             f"ROI: {row['roi']:.2f}x<extra></extra>"
            ))
        
        # Ligne seuil rentabilité
        fig_roi.add_hline(y=1.5, line_dash="dash", line_color="red",
                         annotation_text="Seuil rentabilité (1.5x)")
        
        fig_roi.update_layout(
            title=f"📈 ROI par Persona ({budget_par_persona/1000:.0f}K€ investis)",
            xaxis_title="",
            yaxis_title="ROI (x)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col2:
        # Bar Chart : CA généré par persona
        fig_ca = go.Figure()
        
        for _, row in df_sim.iterrows():
            fig_ca.add_trace(go.Bar(
                x=[row['nom_court']],
                y=[row['ca_genere']/1000],
                name=row['nom_court'],
                marker_color=row['color'],
                text=[f"{row['ca_genere']/1000:.0f}K€"],
                textposition='outside',
                hovertemplate=f"<b>{row['nom']}</b><br>" +
                             f"CA généré: {row['ca_genere']:,.0f}€<br>" +
                             f"Conversions: {row['nb_conversions']}<extra></extra>"
            ))
        
        fig_ca.update_layout(
            title="💸 CA Généré par Persona",
            xaxis_title="",
            yaxis_title="CA Généré (K€)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_ca, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # GRAPHIQUE SCATTER AMÉLIORÉ : BUDGET VS CA
    # ========================================================================
    
    st.subheader("💰 Budget Investi vs CA Généré")
    
    fig_scatter = go.Figure()
    
    # Points pour chaque persona
    for _, row in df_sim.iterrows():
        fig_scatter.add_trace(go.Scatter(
            x=[row['budget']/1000],
            y=[row['ca_genere']/1000],
            mode='markers+text',
            name=row['nom_court'],
            marker=dict(
                size=30,
                color=row['color'],
                line=dict(width=3, color='white'),
                symbol='circle'
            ),
            text=[f"{row['nom_court']}<br>ROI {row['roi']:.1f}x"],
            textposition='top center',
            textfont=dict(size=9, color=row['color'], family='Arial Black'),
            hovertemplate=f"<b>{row['nom']}</b><br>" +
                         f"Budget: {row['budget']:,}€<br>" +
                         f"CA généré: {row['ca_genere']:,.0f}€<br>" +
                         f"Conversions: {row['nb_conversions']:,}<br>" +
                         f"ROI: {row['roi']:.2f}x<br>" +
                         f"Bénéfice: {row['benefice']:,.0f}€<extra></extra>"
        ))
    
    # Ligne break-even (ROI 1x)
    max_budget = df_sim['budget'].max() / 1000 * 1.3  # +30% pour marge
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_budget],
        y=[0, max_budget],
        mode='lines',
        line=dict(dash='dash', color='#e74c3c', width=3),
        name='Break-even (ROI 1x)',
        showlegend=True,
        hovertemplate='ROI 1x (pas de gain)<extra></extra>'
    ))
    
    # Ligne rentabilité (ROI 1.5x)
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_budget],
        y=[0, max_budget * 1.5],
        mode='lines',
        line=dict(dash='dot', color='#f39c12', width=2),
        name='Objectif (ROI 1.5x)',
        showlegend=True,
        hovertemplate='ROI 1.5x (objectif mini)<extra></extra>'
    ))
    
    # Ligne excellence (ROI 2x)
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_budget],
        y=[0, max_budget * 2],
        mode='lines',
        line=dict(dash='dot', color='#27ae60', width=2),
        name='Excellence (ROI 2x)',
        showlegend=True,
        hovertemplate='ROI 2x (excellence)<extra></extra>'
    ))
    
    # Zones colorées pour faciliter la lecture
    fig_scatter.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=max_budget, y1=max_budget,
        fillcolor="rgba(231, 76, 60, 0.08)",
        layer="below",
        line_width=0,
    )
    
    fig_scatter.add_shape(
        type="rect",
        x0=0, y0=max_budget,
        x1=max_budget, y1=max_budget * 1.5,
        fillcolor="rgba(243, 156, 18, 0.08)",
        layer="below",
        line_width=0,
    )
    
    fig_scatter.add_shape(
        type="rect",
        x0=0, y0=max_budget * 1.5,
        x1=max_budget, y1=max_budget * 3,
        fillcolor="rgba(39, 174, 96, 0.08)",
        layer="below",
        line_width=0,
    )
    
    fig_scatter.update_layout(
        title="💰 Budget Investi vs CA Généré",
        xaxis_title="💶 Budget Investi (K€)",
        yaxis_title="💸 CA Généré (K€)",
        height=550,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.info("""
    💡 **Interprétation :** 
    - 🟢 **Zone verte (>1.5x)** : Rentabilité excellente → Investir davantage !
    - 🟠 **Zone orange (1x - 1.5x)** : Rentabilité acceptable → Optimiser les campagnes
    - 🔴 **Zone rouge (<1x)** : Non rentable → Revoir la stratégie
    """)
    
    st.markdown("---")
    
    # ========================================================================
    # TABLEAU DÉTAILLÉ OU SYNTHÈSE
    # ========================================================================
    
    if niveau_detail == "Détaillé":
        st.subheader("📋 Détail des Calculs par Persona")
        
        df_display = df_sim[[
            'nom', 'budget', 'cout_campagne', 'nb_campagnes', 'taux_conversion',
            'nb_conversions', 'panier_moyen', 'ca_genere', 'roi', 'benefice'
        ]].copy()
        
        df_display.columns = [
            'Persona', 'Budget (€)', 'Coût/Camp.', 'Nb Camp.', 'Taux Conv.',
            'Conversions', 'Panier Moy.', 'CA Généré (€)', 'ROI', 'Bénéfice (€)'
        ]
        
        df_display['Budget (€)'] = df_display['Budget (€)'].apply(lambda x: f"{x:,.0f}€")
        df_display['Coût/Camp.'] = df_display['Coût/Camp.'].apply(lambda x: f"{x:.2f}€")
        df_display['Nb Camp.'] = df_display['Nb Camp.'].apply(lambda x: f"{x:,}")
        df_display['Taux Conv.'] = df_display['Taux Conv.'].apply(lambda x: f"{x*100:.0f}%")
        df_display['Conversions'] = df_display['Conversions'].apply(lambda x: f"{x:,}")
        df_display['Panier Moy.'] = df_display['Panier Moy.'].apply(lambda x: f"{x:.0f}€")
        df_display['CA Généré (€)'] = df_display['CA Généré (€)'].apply(lambda x: f"{x:,.0f}€")
        df_display['ROI'] = df_display['ROI'].apply(lambda x: f"{x:.2f}x")
        df_display['Bénéfice (€)'] = df_display['Bénéfice (€)'].apply(
            lambda x: f"{'✅ ' if x > 0 else '❌ '}{x:,.0f}€"
        )
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    else:
        # Vue synthèse
        st.subheader("📋 Tableau Synthèse")
        
        df_display = df_sim[['nom', 'budget', 'ca_genere', 'roi', 'benefice', 'priorite_label']].copy()
        df_display.columns = ['Persona', 'Budget (€)', 'CA Généré (€)', 'ROI', 'Bénéfice (€)', 'Recommandation']
        
        df_display['Budget (€)'] = df_display['Budget (€)'].apply(lambda x: f"{x:,.0f}€")
        df_display['CA Généré (€)'] = df_display['CA Généré (€)'].apply(lambda x: f"{x:,.0f}€")
        df_display['ROI'] = df_display['ROI'].apply(lambda x: f"{x:.2f}x")
        df_display['Bénéfice (€)'] = df_display['Bénéfice (€)'].apply(
            lambda x: f"{'✅' if x > 0 else '❌'} {x:,.0f}€"
        )
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # RECOMMANDATIONS
    # ========================================================================
    
    st.markdown("---")
    st.subheader("💡 Recommandations")
    
    # Meilleur ROI
    best = df_sim.loc[df_sim['roi'].idxmax()]
    worst = df_sim.loc[df_sim['roi'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **🏆 Meilleur ROI : {best['nom']}**
        
        - Budget : {best['budget']:,}€
        - CA généré : {best['ca_genere']:,.0f}€
        - ROI : **{best['roi']:.2f}x**
        - Bénéfice : **{best['benefice']:,.0f}€**
        
        💡 **Action :** Maximiser l'investissement sur ce segment !
        """)
    
    with col2:
        if worst['roi'] < 1.5:
            st.warning(f"""
            **⚠️ ROI Faible : {worst['nom']}**
            
            - Budget : {worst['budget']:,}€
            - CA généré : {worst['ca_genere']:,.0f}€
            - ROI : **{worst['roi']:.2f}x**
            - Bénéfice : **{worst['benefice']:,.0f}€**
            
            💡 **Action :** Réduire le budget ou améliorer le taux de conversion.
            """)
        else:
            st.info(f"""
            **📊 ROI Correct : {worst['nom']}**
            
            - ROI : **{worst['roi']:.2f}x**
            
            💡 Tous les personas sont rentables !
            """)
    
    # ========================================================================
    # OPTIMISATION BUDGÉTAIRE
    # ========================================================================
    
    st.markdown("---")
    st.subheader("🎯 Optimisation Budgétaire Suggérée")
    
    # Calculer allocation optimale basée sur ROI
    df_sim_sorted = df_sim.sort_values('roi', ascending=False)
    budget_total_dispo = budget_par_persona * len(df_sim)
    
    st.markdown(f"""
    Si vous avez **{budget_total_dispo:,}€** à répartir, voici l'allocation optimale basée sur le ROI :
    """)
    
    # Réallocation proportionnelle au ROI
    roi_total = df_sim['roi'].sum()
    
    optimisation = []
    for _, row in df_sim_sorted.iterrows():
        poids_roi = row['roi'] / roi_total if roi_total > 0 else 0
        budget_optimal = budget_total_dispo * poids_roi
        
        # Recalculer avec budget optimal
        nb_camp_opt = int(budget_optimal / row['cout_campagne'])
        nb_conv_opt = int(nb_camp_opt * row['taux_conversion'])
        ca_opt = nb_conv_opt * row['panier_moyen']
        
        optimisation.append({
            'Persona': row['nom'],
            'Budget Actuel': f"{row['budget']:,.0f}€",
            'Budget Optimal': f"{budget_optimal:,.0f}€",
            'CA Actuel': f"{row['ca_genere']:,.0f}€",
            'CA Optimal': f"{ca_opt:,.0f}€",
            'Gain': f"{ca_opt - row['ca_genere']:,.0f}€"
        })
    
    df_optim = pd.DataFrame(optimisation)
    st.dataframe(df_optim, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    st.markdown("---")
    st.subheader("📥 Export")
    
    export_data = df_sim[['nom', 'budget', 'nb_campagnes', 'nb_conversions', 'ca_genere', 'roi', 'benefice']].copy()
    export_data.columns = ['Persona', 'Budget (€)', 'Campagnes', 'Conversions', 'CA Généré (€)', 'ROI', 'Bénéfice (€)']
    
    csv = export_data.to_csv(index=False)
    st.download_button(
        label="📄 Télécharger Simulation (CSV)",
        data=csv,
        file_name=f"simulation_roi_{budget_par_persona}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


# ============================================================================
# STANDALONE
# ============================================================================


if __name__ == "__main__":
    st.set_page_config(
        page_title="ROI Marketing",
        page_icon="💰",
        layout="wide"
    )
    run(None, None)


