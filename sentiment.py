import pandas as pd
import os
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & MODE TEST
# ==========================================
DATA_DIR = "data"
TEST_MODE = True  # Mettre à False pour traiter TOUTES les données
N_ROWS_TEST = 30   # Nombre de lignes à tester si TEST_MODE est True

# Modèles sélectionnés pour leur stabilité et performance multilingue
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
THEME_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

CANDIDATE_LABELS = [
    "inflation", "public debt", "interest rates", "economic growth", 
    "monetary policy", "fiscal policy", "sovereign risk", 
    "political uncertainty", "market fragmentation", "labor market"
]
# ==========================================
# 2. CHARGEMENT DES MODÈLES (Safe Load)
# ==========================================
print("--- Initialisation des modèles ---")
try:
    # use_fast=False évite les bugs de tokenisation sur Windows/Python 3.12
    tokenizer_sent = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, use_fast=False)
    sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=tokenizer_sent, device=-1)
    
    theme_pipe = pipeline("zero-shot-classification", model=THEME_MODEL, device=-1)
    print("Modèles chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")
    exit()

# ==========================================
# 3. FONCTION DE TRAITEMENT
# ==========================================
def process_data(file_name, text_column):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"Fichier manquant : {path}")
        return None

    # Chargement (entier ou échantillon)
    if TEST_MODE:
        df = pd.read_csv(path, nrows=N_ROWS_TEST)
        print(f"\n[MODE TEST] Analyse de {N_ROWS_TEST} lignes de {file_name}...")
    else:
        df = pd.read_csv(path)
        print(f"\n[MODE COMPLET] Analyse de {len(df)} lignes de {file_name}...")

    # Règle de majuscule : Seul le premier mot commence par une majuscule
    if 'title' in df.columns:
        df['title'] = df['title'].astype(str).apply(lambda x: x.capitalize())

    sentiments = []
    scores = []
    themes = []

    for text in tqdm(df[text_column].astype(str)):
        clean_text = text[:800] 
        
        # Extraction du sentiment
        s_res = sentiment_pipe(clean_text)[0]
        label = s_res['label']
        prob = s_res['score'] # La confiance du modèle (ex: 0.98)

        # Mapping spécifique au modèle XLM-RoBERTa
        # LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive
        if label == 'LABEL_0' or label.lower() == 'negative':
            final_score = -prob
        elif label == 'LABEL_2' or label.lower() == 'positive':
            final_score = prob
        else:
            final_score = 0  # Neutre reste à 0
            
        sentiments.append(label)
        scores.append(round(final_score, 4)) # On garde 4 décimales pour l'économétrie
        
        # Thématique
        t_res = theme_pipe(clean_text, candidate_labels=CANDIDATE_LABELS)
        themes.append(t_res['labels'][0])

    df['sentiment_label'] = sentiments
    df['sentiment_score'] = scores
    df['main_theme'] = themes
    
    return df

# ==========================================
# 4. EXÉCUTION
# ==========================================
# Liste des fichiers et de leur colonne textuelle cible
tasks = [
    ('ecb_monetary_policy_decisions_2020_2026.csv', 'text'),
    ('french_economy_news.csv', 'title'),
    ('german_economy_news.csv', 'title')
]

results = {}

for file, col in tasks:
    processed_df = process_data(file, col)
    if processed_df is not None:
        # Sauvegarde avec suffixe _analyzed
        output_name = file.replace('.csv', '_analyzed.csv')
        processed_df.to_csv(os.path.join(DATA_DIR, output_name), index=False)
        print(f"Sauvegardé : {output_name}")

print("\n--- Mission accomplie ! ---")
print("Tes données sont prêtes pour ton modèle de Nowcasting.")