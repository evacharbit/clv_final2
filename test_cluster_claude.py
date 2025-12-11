"""
CLUSTERING PERSONAS - 7 CLUSTERS - TOUS LES KPI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib, os, warnings, time

warnings.filterwarnings('ignore')
start_time = time.time()

# --- CONFIGURATION ---
from config import PROJECT_ID, RANDOM_STATE
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
N_PERSONAS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- CHARGEMENT DES DONNEES DE BIGQUERY ---
client = bigquery.Client(project=PROJECT_ID)
query = f"SELECT * FROM `{PROJECT_ID}.data_enrichie.transactionparclient_age_predicted`"
df = client.query(query).to_dataframe()
print(f"✅ {len(df):,} clients chargés depuis BigQuery")

# --- AJOUT DE TOUTES LES COLONNES SI MANQUANTES ---
all_columns = [
    'customer_id','gender','signup_date','pays','type_loyalty_card','age_range',
    'customer_status','CA_total','panier_moyen','nb_total_produits','nb_moyen_produits',
    'total_discount','first_transaction_date','last_transaction_date','nb_transaction',
    'nb_HYP','nb_DRI','customer_lifespan_days','Average_Purchase_Frequency_Per_Month',
    'last_transaction_days','rfm_code','segment_rfm','nb_campagnes_recues','pression_commerciale',
    'secteur_top_1','secteur_top_2','secteur_top_3','Historical_CLV_Revenue','pourcentage_remise',
    'quartile_remise','favorite_store'
]
for col in all_columns:
    if col not in df.columns:
        df[col] = np.nan

# --- IDENTIFICATION DES FEATURES ---
exclude_cols = ['customer_id', 'signup_date', 'first_transaction_date', 'last_transaction_date']
categorical_cols = [c for c in all_columns if c not in exclude_cols and df[c].dtype == 'object']
numeric_cols = [c for c in all_columns if c not in exclude_cols and df[c].dtype in ['int64','float64']]
features = numeric_cols + categorical_cols
print(f"Features utilisées : {len(features)} ({len(numeric_cols)} num + {len(categorical_cols)} cat)")

# --- ENCODAGE ET NORMALISATION ---
df_encoded = df.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].fillna('unknown'))
    label_encoders[col] = le

X = df_encoded[features].fillna(0).replace([np.inf, -np.inf], 0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- CLUSTERING FINAL ---
kmeans = KMeans(n_clusters=N_PERSONAS, random_state=RANDOM_STATE, n_init=50, max_iter=500)
labels_kmeans = kmeans.fit_predict(X_scaled)
df['persona_id'] = labels_kmeans
silhouette_final = silhouette_score(X_scaled, df['persona_id'], sample_size=min(10000, len(df)))
print(f"✅ Silhouette final : {silhouette_final:.4f}")

# --- PROFILING DES PERSONAS ---
personas_profiles = []
for persona_id in range(N_PERSONAS):
    df_persona = df[df['persona_id'] == persona_id]
    profile = {
        'persona_id': persona_id,
        'taille': len(df_persona),
        'pct_base': len(df_persona)/len(df)*100,
        'ca_total': df_persona['CA_total'].sum(),
        'ca_moyen': df_persona['CA_total'].mean(),
        'clv_moyenne': df_persona['Historical_CLV_Revenue'].mean(),
        'panier_moyen': df_persona['panier_moyen'].mean(),
        'frequence_mois': df_persona['Average_Purchase_Frequency_Per_Month'].mean(),
        'nb_transactions_moyen': df_persona['nb_transaction'].mean(),
        'recency_moyenne': df_persona['last_transaction_days'].mean(),
        'pct_drive': (df_persona['nb_DRI']/(df_persona['nb_transaction']+0.001)).mean()*100,
        'pct_hyp': (df_persona['nb_HYP']/(df_persona['nb_transaction']+0.001)).mean()*100,
        'pct_france': (df_persona['pays']=='FR').mean()*100,
        'pct_belgique': (df_persona['pays']=='BE').mean()*100,
        'discount_rate': (df_persona['total_discount']/(df_persona['CA_total']+0.001)).mean()*100,
        'pct_remise_moyen': df_persona['pourcentage_remise'].mean(),
        'pct_churned': (df_persona['last_transaction_days']>180).mean()*100,
        'pct_at_risk': (df_persona['last_transaction_days']>90).mean()*100,
        'secteur_principal': df_persona['secteur_top_1'].mode()[0] if df_persona['secteur_top_1'].notna().any() else 'N/A',
        'secteur_secondaire': df_persona['secteur_top_2'].mode()[0] if df_persona['secteur_top_2'].notna().any() else 'N/A',
        'segment_rfm_mode': df_persona['segment_rfm'].mode()[0] if df_persona['segment_rfm'].notna().any() else 'N/A'
    }
    personas_profiles.append(profile)

df_profiles = pd.DataFrame(personas_profiles)
df_profiles.to_csv(f'{OUTPUT_DIR}/personas_profiles.csv', index=False)
print(f"✅ Profils complets enregistrés : {OUTPUT_DIR}/personas_profiles.csv")

# --- EXPORT CSV AVEC LE CLUSTER DE CHAQUE CLIENT ---
df_clients_clusters = df[['customer_id', 'persona_id']]
df_clients_clusters.to_csv(f'{OUTPUT_DIR}/clients_clusters.csv', index=False)
print(f"✅ CSV clients avec cluster enregistré : {OUTPUT_DIR}/clients_clusters.csv")


# --- HEATMAP DES FEATURES PAR CLUSTER ---
df_scaled = pd.DataFrame(X_scaled, columns=features)
df_scaled['cluster'] = labels_kmeans
cluster_means = df_scaled.groupby('cluster').mean()

plt.figure(figsize=(14, 8))
sns.heatmap(cluster_means.T, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap des features par cluster", fontsize=16)
plt.tight_layout()
heatmap_clusters_path = f"{OUTPUT_DIR}/heatmap_features_clusters.png"
plt.savefig(heatmap_clusters_path, dpi=300)
plt.close()
print(f"✅ Heatmap créée : {heatmap_clusters_path}")

print("\n🎉 Script terminé en {:.1f}s".format(time.time() - start_time))
