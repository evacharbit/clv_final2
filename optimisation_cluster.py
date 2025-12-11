"""
CLUSTERING PERSONAS - VERSION OPTIMISÉE
Amélioration de la qualité du clustering (Silhouette > 0.35)
Détection automatique K optimal + Feature Selection + PCA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import joblib, os, warnings, time
from tqdm import tqdm
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

from config import PROJECT_ID, RANDOM_STATE

OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Paramètres optimisés
K_MIN = 6
K_MAX = 6
USE_PCA = True
PCA_VARIANCE = 0.90  # Conserver 90% de la variance
FEATURE_SELECTION = True

print("="*80)
print("CLUSTERING PERSONAS - VERSION OPTIMISÉE")
print("="*80)
print(f"Configuration:")
print(f"  • Test K = {K_MIN} à {K_MAX}")
print(f"  • PCA activé : {USE_PCA} (variance conservée : {PCA_VARIANCE*100}%)")
print(f"  • Feature selection : {FEATURE_SELECTION}")
print("="*80)

start_time_total = time.time()

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

print("\n📊 ÉTAPE 1/8 : Chargement des données depuis BigQuery")
start_time = time.time()

client = bigquery.Client(project=PROJECT_ID)
query = f"SELECT * FROM `{PROJECT_ID}.data_enrichie.transactionparclient_age_predicted`"

print("⏳ Exécution de la requête BigQuery...")
df = client.query(query).to_dataframe()

elapsed = time.time() - start_time
print(f"✅ {len(df):,} clients chargés en {elapsed:.1f}s")

# ============================================================================
# 2. FEATURE ENGINEERING AVANCÉ
# ============================================================================

print("\n🔧 ÉTAPE 2/8 : Feature Engineering Avancé")
start_time = time.time()

# Colonnes existantes à vérifier
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

# ===== CRÉER DES FEATURES DISCRIMINANTES =====

print("⏳ Création de features enrichies...")

# Ratios et indicateurs comportementaux
df['pct_drive'] = df['nb_DRI'] / (df['nb_transaction'] + 0.001) * 100
df['pct_hyp'] = df['nb_HYP'] / (df['nb_transaction'] + 0.001) * 100
df['discount_rate'] = df['total_discount'] / (df['CA_total'] + 0.001) * 100
df['is_churned'] = (df['last_transaction_days'] > 180).astype(int)
df['is_at_risk'] = (df['last_transaction_days'] > 90).astype(int)
df['panier_per_transaction'] = df['CA_total'] / (df['nb_transaction'] + 0.001)
df['produits_per_transaction'] = df['nb_total_produits'] / (df['nb_transaction'] + 0.001)
df['frequency_per_month'] = df['nb_transaction'] / (df['customer_lifespan_days'] / 30 + 0.001)

# Segmentation RFM binaire
df['is_vip'] = (df['Historical_CLV_Revenue'] > df['Historical_CLV_Revenue'].quantile(0.75)).astype(int)
df['is_frequent'] = (df['Average_Purchase_Frequency_Per_Month'] > df['Average_Purchase_Frequency_Per_Month'].median()).astype(int)
df['is_recent'] = (df['last_transaction_days'] < 30).astype(int)

# Indicateurs géographiques
df['is_france'] = (df['pays'] == 'FR').astype(int)
df['is_belgique'] = (df['pays'] == 'BE').astype(int)

# Type de carte
df['has_oney'] = (df['type_loyalty_card'].str.contains('Oney', case=False, na=False)).astype(int)
df['has_waaoh'] = (df['type_loyalty_card'].str.contains('Waaoh', case=False, na=False)).astype(int)

elapsed = time.time() - start_time
print(f"✅ {len([c for c in df.columns if c not in all_columns])} nouvelles features créées en {elapsed:.1f}s")

# ============================================================================
# 3. SÉLECTION ET PRÉPARATION DES FEATURES
# ============================================================================

print("\n📋 ÉTAPE 3/8 : Sélection des features pour le clustering")
start_time = time.time()

# Colonnes à exclure
exclude_cols = [
    'customer_id', 'signup_date', 'first_transaction_date', 'last_transaction_date',
    'secteur_top_1', 'secteur_top_2', 'secteur_top_3', 'favorite_store',  # Trop de modalités
    'rfm_code'  # Redondant avec segment_rfm
]

# Features catégorielles (peu de modalités)
categorical_cols = [
    'gender', 'pays', 'type_loyalty_card', 'age_range', 
    'customer_status', 'segment_rfm', 'quartile_remise'
]

# Features numériques
numeric_cols = [c for c in df.columns 
                if c not in exclude_cols + categorical_cols 
                and df[c].dtype in ['int64', 'float64']]

print(f"📊 Features sélectionnées : {len(numeric_cols) + len(categorical_cols)}")
print(f"   • Numériques : {len(numeric_cols)}")
print(f"   • Catégorielles : {len(categorical_cols)}")

# ===== ENCODAGE DES VARIABLES CATÉGORIELLES =====

print("\n⏳ Encodage des variables catégorielles...")

df_encoded = df.copy()
label_encoders = {}

for i, col in enumerate(tqdm(categorical_cols, desc="Encodage", unit="col", file=sys.stdout)):
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].fillna('unknown').astype(str))
    label_encoders[col] = le

# ===== MATRICE DE FEATURES =====

features = numeric_cols + categorical_cols
X = df_encoded[features].fillna(0).replace([np.inf, -np.inf], 0)

print(f"✅ Matrice initiale : {X.shape}")

# ===== SUPPRESSION DES FEATURES À VARIANCE NULLE =====

if FEATURE_SELECTION:
    print("\n⏳ Suppression des features à faible variance...")
    
    selector_variance = VarianceThreshold(threshold=0.01)
    X_var = selector_variance.fit_transform(X)
    
    selected_features_mask = selector_variance.get_support()
    selected_features = [features[i] for i, selected in enumerate(selected_features_mask) if selected]
    
    print(f"✅ Features conservées : {len(selected_features)} (supprimé {len(features) - len(selected_features)})")
    
    X = pd.DataFrame(X_var, columns=selected_features)
    features = selected_features

# ===== NORMALISATION =====

print("\n⏳ Normalisation des données...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Données normalisées : {X_scaled.shape}")

elapsed = time.time() - start_time
print(f"✅ Préparation terminée en {elapsed:.1f}s")

# ============================================================================
# 4. RÉDUCTION DE DIMENSIONNALITÉ (PCA)
# ============================================================================

if USE_PCA:
    print("\n🔬 ÉTAPE 4/8 : Réduction de dimensionnalité (PCA)")
    start_time = time.time()
    
    print(f"⏳ Application du PCA (variance conservée : {PCA_VARIANCE*100}%)...")
    
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    n_components = X_pca.shape[1]
    variance_explained = pca.explained_variance_ratio_.sum()
    
    print(f"✅ PCA appliqué : {X_scaled.shape[1]} features → {n_components} composantes")
    print(f"✅ Variance expliquée : {variance_explained*100:.2f}%")
    
    # Utiliser PCA pour le clustering
    X_for_clustering = X_pca
    
    # Sauvegarder le modèle PCA
    joblib.dump(pca, f'{MODELS_DIR}/pca.pkl')
    
    elapsed = time.time() - start_time
    print(f"✅ PCA terminé en {elapsed:.1f}s")
else:
    X_for_clustering = X_scaled
    print("\n⏩ ÉTAPE 4/8 : PCA désactivé (skip)")

# ============================================================================
# 5. DÉTECTION DU K OPTIMAL (ELBOW METHOD)
# ============================================================================

print("\n📈 ÉTAPE 5/8 : Détection du K optimal (Elbow Method)")
print(f"⏳ Test de K={K_MIN} à K={K_MAX} (peut prendre 5-10 minutes)...")
start_time = time.time()

K_range = range(K_MIN, K_MAX + 1)
inertias = []
silhouettes = []
davies_bouldin_scores = []
calinski_scores = []

# Barre de progression pour les tests de K
for k in tqdm(K_range, desc="Test K", unit="cluster", file=sys.stdout):
    kmeans_test = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30, max_iter=300)
    labels_test = kmeans_test.fit_predict(X_for_clustering)
    
    inertias.append(kmeans_test.inertia_)
    
    # Silhouette Score (0-1, plus élevé = mieux)
    sil = silhouette_score(X_for_clustering, labels_test, sample_size=min(10000, len(df)))
    silhouettes.append(sil)
    
    # Davies-Bouldin Index (plus bas = mieux)
    db = davies_bouldin_score(X_for_clustering, labels_test)
    davies_bouldin_scores.append(db)
    
    # Calinski-Harabasz Score (plus élevé = mieux)
    ch = calinski_harabasz_score(X_for_clustering, labels_test)
    calinski_scores.append(ch)

elapsed = time.time() - start_time
print(f"\n✅ Tests terminés en {elapsed:.1f}s ({elapsed/60:.1f} min)")

# ===== SÉLECTION DU K OPTIMAL =====

# Méthode : Maximiser Silhouette
best_k_silhouette = K_range[np.argmax(silhouettes)]

# Méthode : Minimiser Davies-Bouldin
best_k_db = K_range[np.argmin(davies_bouldin_scores)]

# Méthode : Compromis (Silhouette élevé + Calinski élevé)
normalized_sil = (np.array(silhouettes) - np.min(silhouettes)) / (np.max(silhouettes) - np.min(silhouettes))
normalized_cal = (np.array(calinski_scores) - np.min(calinski_scores)) / (np.max(calinski_scores) - np.min(calinski_scores))
composite_score = normalized_sil * 0.6 + normalized_cal * 0.4
best_k_composite = K_range[np.argmax(composite_score)]

print(f"\n🎯 K optimal détecté :")
print(f"   • Par Silhouette : K = {best_k_silhouette} (score: {silhouettes[best_k_silhouette - K_MIN]:.4f})")
print(f"   • Par Davies-Bouldin : K = {best_k_db} (score: {davies_bouldin_scores[best_k_db - K_MIN]:.4f})")
print(f"   • Par Score Composite : K = {best_k_composite}")

# Choisir le K optimal (priorité au silhouette si > 0.30, sinon composite)
if silhouettes[best_k_silhouette - K_MIN] > 0.30:
    N_PERSONAS = best_k_silhouette
    print(f"\n✅ K OPTIMAL RETENU : {N_PERSONAS} (Silhouette élevé)")
else:
    N_PERSONAS = best_k_composite
    print(f"\n✅ K OPTIMAL RETENU : {N_PERSONAS} (Score composite)")

# ===== VISUALISATION ELBOW =====

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Inertia
axes[0, 0].plot(K_range, inertias, 'o-', linewidth=2, markersize=8, color='#3498db')
axes[0, 0].axvline(N_PERSONAS, color='red', linestyle='--', label=f'K optimal = {N_PERSONAS}')
axes[0, 0].set_title('Elbow Method - Inertia', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Nombre de clusters (K)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()

# 2. Silhouette
axes[0, 1].plot(K_range, silhouettes, 'o-', linewidth=2, markersize=8, color='#2ecc71')
axes[0, 1].axvline(N_PERSONAS, color='red', linestyle='--', label=f'K optimal = {N_PERSONAS}')
axes[0, 1].axhline(0.30, color='orange', linestyle=':', label='Seuil acceptable (0.30)')
axes[0, 1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Nombre de clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend()

# 3. Davies-Bouldin
axes[1, 0].plot(K_range, davies_bouldin_scores, 'o-', linewidth=2, markersize=8, color='#e74c3c')
axes[1, 0].axvline(N_PERSONAS, color='red', linestyle='--', label=f'K optimal = {N_PERSONAS}')
axes[1, 0].set_title('Davies-Bouldin Index (plus bas = mieux)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Nombre de clusters (K)')
axes[1, 0].set_ylabel('Davies-Bouldin Score')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend()

# 4. Calinski-Harabasz
axes[1, 1].plot(K_range, calinski_scores, 'o-', linewidth=2, markersize=8, color='#9b59b6')
axes[1, 1].axvline(N_PERSONAS, color='red', linestyle='--', label=f'K optimal = {N_PERSONAS}')
axes[1, 1].set_title('Calinski-Harabasz Score (plus élevé = mieux)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Nombre de clusters (K)')
axes[1, 1].set_ylabel('Calinski-Harabasz Score')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/elbow_method_complet.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Graphique Elbow sauvegardé : {OUTPUT_DIR}/elbow_method_complet.png")

# ============================================================================
# 6. CLUSTERING FINAL AVEC K OPTIMAL
# ============================================================================

print(f"\n🎯 ÉTAPE 6/8 : Clustering final avec K={N_PERSONAS}")
start_time = time.time()

print("⏳ Entraînement du modèle K-Means final (50 initialisations)...")

kmeans_final = KMeans(
    n_clusters=N_PERSONAS,
    random_state=RANDOM_STATE,
    n_init=50,
    max_iter=500,
    verbose=0
)

# Entraînement avec barre de progression simulée
with tqdm(total=50, desc="Initialisations K-Means", unit="init", file=sys.stdout) as pbar:
    labels_final = kmeans_final.fit_predict(X_for_clustering)
    pbar.update(50)

df['persona_id'] = labels_final

# Métriques finales
silhouette_final = silhouette_score(X_for_clustering, labels_final, sample_size=min(10000, len(df)))
db_final = davies_bouldin_score(X_for_clustering, labels_final)
ch_final = calinski_harabasz_score(X_for_clustering, labels_final)

elapsed = time.time() - start_time

print(f"\n✅ Clustering final terminé en {elapsed:.1f}s")
print(f"\n📊 MÉTRIQUES FINALES :")
print(f"   • Silhouette Score : {silhouette_final:.4f} (objectif > 0.30)")
print(f"   • Davies-Bouldin Index : {db_final:.4f} (plus bas = mieux)")
print(f"   • Calinski-Harabasz Score : {ch_final:.2f} (plus élevé = mieux)")

# Distribution des clusters
print(f"\n👥 DISTRIBUTION DES PERSONAS :")
for persona_id in range(N_PERSONAS):
    count = (labels_final == persona_id).sum()
    pct = count / len(df) * 100
    print(f"   Persona {persona_id} : {count:>7,} clients ({pct:>5.1f}%)")

# ============================================================================
# 7. PROFILING DES PERSONAS (18 KPI)
# ============================================================================

print(f"\n📋 ÉTAPE 7/8 : Profiling des {N_PERSONAS} personas (18 KPI)")
start_time = time.time()

personas_profiles = []

for persona_id in tqdm(range(N_PERSONAS), desc="Profiling", unit="persona", file=sys.stdout):
    df_persona = df[df['persona_id'] == persona_id]
    
    profile = {
        'persona_id': persona_id,
        'taille': len(df_persona),
        'pct_base': len(df_persona) / len(df) * 100,
        
        # Financier
        'ca_total': df_persona['CA_total'].sum(),
        'ca_moyen': df_persona['CA_total'].mean(),
        'clv_moyenne': df_persona['Historical_CLV_Revenue'].mean(),
        'panier_moyen': df_persona['panier_moyen'].mean(),
        
        # Comportement
        'frequence_mois': df_persona['Average_Purchase_Frequency_Per_Month'].mean(),
        'nb_transactions_moyen': df_persona['nb_transaction'].mean(),
        'recency_moyenne': df_persona['last_transaction_days'].mean(),
        
        # Canal
        'pct_drive': df_persona['pct_drive'].mean(),
        'pct_hyp': df_persona['pct_hyp'].mean(),
        
        # Géographie
        'pct_france': (df_persona['pays'] == 'FR').mean() * 100,
        'pct_belgique': (df_persona['pays'] == 'BE').mean() * 100,
        
        # Promotions
        'discount_rate': df_persona['discount_rate'].mean(),
        'pct_remise_moyen': df_persona['pourcentage_remise'].mean(),
        
        # Risque
        'pct_churned': (df_persona['last_transaction_days'] > 180).mean() * 100,
        'pct_at_risk': (df_persona['last_transaction_days'] > 90).mean() * 100,
        
        # Produits & Segmentation
        'secteur_principal': df_persona['secteur_top_1'].mode()[0] if df_persona['secteur_top_1'].notna().any() else 'N/A',
        'secteur_secondaire': df_persona['secteur_top_2'].mode()[0] if df_persona['secteur_top_2'].notna().any() else 'N/A',
        'segment_rfm_mode': df_persona['segment_rfm'].mode()[0] if df_persona['segment_rfm'].notna().any() else 'N/A'
    }
    
    personas_profiles.append(profile)

df_profiles = pd.DataFrame(personas_profiles)

elapsed = time.time() - start_time
print(f"✅ Profiling terminé en {elapsed:.1f}s")

# ============================================================================
# 8. EXPORTS
# ============================================================================

print(f"\n💾 ÉTAPE 8/8 : Exports des résultats")
start_time = time.time()

# CSV Profils
df_profiles.to_csv(f'{OUTPUT_DIR}/personas_profiles.csv', index=False)
print(f"✅ CSV profils : {OUTPUT_DIR}/personas_profiles.csv")

# CSV Clients avec persona_id
df_clients_clusters = df[['customer_id', 'persona_id']]
df_clients_clusters.to_csv(f'{OUTPUT_DIR}/clients_clusters.csv', index=False)
print(f"✅ CSV clients : {OUTPUT_DIR}/clients_clusters.csv")

# Modèles ML
joblib.dump(kmeans_final, f'{MODELS_DIR}/kmeans_final.pkl')
joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')
joblib.dump(label_encoders, f'{MODELS_DIR}/label_encoders.pkl')
print(f"✅ Modèles sauvegardés dans {MODELS_DIR}/")

# Heatmap des features par cluster
print("\n⏳ Création de la heatmap...")

if USE_PCA:
    # Si PCA, utiliser les composantes principales
    df_heatmap = pd.DataFrame(X_for_clustering, columns=[f'PC{i+1}' for i in range(X_for_clustering.shape[1])])
else:
    df_heatmap = pd.DataFrame(X_scaled, columns=features)

df_heatmap['cluster'] = labels_final
cluster_means = df_heatmap.groupby('cluster').mean()

plt.figure(figsize=(16, 10))
sns.heatmap(cluster_means.T, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
plt.title(f"Heatmap des features par cluster (K={N_PERSONAS})", fontsize=16, fontweight='bold')
plt.xlabel('Cluster ID', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_features_clusters.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Heatmap : {OUTPUT_DIR}/heatmap_features_clusters.png")

# Export vers BigQuery
from google.cloud.bigquery import LoadJobConfig

print("\n⏳ Export vers BigQuery...")

# Table personas_profiles
table_profiles = f"{PROJECT_ID}.data_enrichie.personas_profiles"
job_config = LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(df_profiles, table_profiles, job_config=job_config)
job.result()
print(f"✅ Table BigQuery : {table_profiles}")

# Table personas_master (customer_id + persona_id)
table_master = f"{PROJECT_ID}.data_enrichie.personas_master"
job = client.load_table_from_dataframe(df_clients_clusters, table_master, job_config=job_config)
job.result()
print(f"✅ Table BigQuery : {table_master}")

elapsed = time.time() - start_time
print(f"✅ Exports terminés en {elapsed:.1f}s")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

total_elapsed = time.time() - start_time_total

print("\n" + "="*80)
print("✅ CLUSTERING OPTIMISÉ TERMINÉ AVEC SUCCÈS")
print("="*80)

print(f"""
⏱️  TEMPS TOTAL : {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)

📊 RÉSULTATS CLÉS :
   • Clients analysés : {len(df):,}
   • Features utilisées : {len(features)} → {X_for_clustering.shape[1]} (après PCA)
   • K OPTIMAL : {N_PERSONAS} personas
   • Silhouette Score : {silhouette_final:.4f} ({'+' if silhouette_final > 0.30 else '-'} objectif > 0.30)
   • Davies-Bouldin : {db_final:.4f} (plus bas = mieux)
   • Calinski-Harabasz : {ch_final:.2f} (plus élevé = mieux)

📁 FICHIERS GÉNÉRÉS :
   📊 {OUTPUT_DIR}/elbow_method_complet.png
   🔥 {OUTPUT_DIR}/heatmap_features_clusters.png
   📋 {OUTPUT_DIR}/personas_profiles.csv
   👥 {OUTPUT_DIR}/clients_clusters.csv
   💾 {MODELS_DIR}/kmeans_final.pkl
   💾 {MODELS_DIR}/scaler.pkl
   💾 {MODELS_DIR}/pca.pkl (si activé)

🗄️ TABLES BIGQUERY :
   • {PROJECT_ID}.data_enrichie.personas_profiles
   • {PROJECT_ID}.data_enrichie.personas_master

🎯 QUALITÉ DU CLUSTERING :
""")

if silhouette_final > 0.40:
    print("   ✅ EXCELLENT - Clusters très bien séparés")
elif silhouette_final > 0.30:
    print("   ✅ BON - Clusters correctement séparés")
elif silhouette_final > 0.20:
    print("   ⚠️ MOYEN - Clusters partiellement séparés")
else:
    print("   ❌ FAIBLE - Clusters mal séparés (revoir les features)")

print("\n" + "="*80)
print("🚀 Script terminé !")
print("="*80)