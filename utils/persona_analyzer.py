import pandas as pd

def get_persona_stats(df_profiles):
    """Retourne un DataFrame avec les stats clés pour chaque persona"""
    cols_kpi = [
        "persona_id", "taille", "ca_total", "ca_moyen",
        "clv_moyenne", "panier_moyen", "frequence_mois",
        "nb_transactions_moyen", "recency_moyenne"
    ]
    return df_profiles[cols_kpi]

def merge_transactions(df_transactions, df_profiles):
    """Ajoute les infos personas aux transactions individuelles"""
    return df_transactions.merge(df_profiles, on="persona_id", how="left")
