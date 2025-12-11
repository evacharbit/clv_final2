import pandas as pd
from google.cloud import bigquery
from config import PROJECT_ID, DATASET_ENRICHIE

client = bigquery.Client(project=PROJECT_ID)

def load_personas_profiles():
    """
    Charge la table personas_profiles :
    Profils détaillés par persona
    """
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ENRICHIE}.personas_profiles`
    """
    return client.query(query).to_dataframe()

def load_clusters():
    """
    Charge la table cluster :
    Table client enrichie avec la colonne persona
    """
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ENRICHIE}.cluster`
    """
    return client.query(query).to_dataframe()

def load_ticket():
    """
    Charge la table ticket :
    Table transactionnelle jour par jour
    """
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ENRICHIE}.ticket`
    """
    return client.query(query).to_dataframe()

def load_data():
    """
    Charge les trois tables essentielles :
    - personas_profiles
    - cluster
    - ticket
    """
    personas_profiles = load_personas_profiles()
    clusters = load_clusters()
    ticket = load_ticket()
    return personas_profiles, clusters, ticket
