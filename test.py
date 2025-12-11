from utils.data_loader import load_persona_profiles, load_transactions
from utils.persona_analyzer import get_persona_stats, merge_transactions

df_profiles = load_persona_profiles()
df_transactions = load_transactions()

print("Personas Profiles :")
print(df_profiles.head())

print("\nTransactions :")
print(df_transactions.head())

# Exemple rapide d’utilisation
df_stats = get_persona_stats(df_profiles)
print("\nStats personas :")
print(df_stats.head())
