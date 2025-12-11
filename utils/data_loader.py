import pandas as pd
from config import PROJECT_ID, DATASET_ENRICHIE, client

def load_personas_profiles():
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ENRICHIE}.personas_profiles`
    """
    return client.query(query).to_dataframe()

def load_clusters():
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ENRICHIE}.cluster`
    """
    return client.query(query).to_dataframe()

def load_ticket():
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
