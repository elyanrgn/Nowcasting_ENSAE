import pandas as pd
import os
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = "data"
BATCH_SIZE = 8 
TEST_MODE = False # Garde True pour vérifier que les scores ne sont plus à 0

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
THEME_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

CANDIDATE_LABELS = [
    "inflation", "public debt", "interest rates", "economic growth", 
    "monetary policy", "fiscal policy", "sovereign risk"
]

device = 0 if torch.cuda.is_available() else -1

print(f"--- Initialisation sur {'GPU' if device == 0 else 'CPU'} ---")
tokenizer_sent = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, use_fast=False)
sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=tokenizer_sent, device=device)
theme_pipe = pipeline("zero-shot-classification", model=THEME_MODEL, device=device)

# ==========================================
# 2. FONCTION DE MAPPING ROBUSTE
# ==========================================
def get_numeric_score(prediction):
    label = prediction['label'].upper()
    score = prediction['score']
    
    # Mapping universel pour les modèles CardiffNLP (XLM-R)
    # LABEL_0 = Negative | LABEL_1 = Neutral | LABEL_2 = Positive
    if 'LABEL_0' in label or 'NEG' in label:
        return -score
    elif 'LABEL_2' in label or 'POS' in label:
        return score
    else:
        return 0.0

# ==========================================
# 3. EXÉCUTION
# ==========================================
def process_final(file_name, text_column):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path): return None

    df = pd.read_csv(path, nrows=10 if TEST_MODE else None)
    
    # Nettoyage des colonnes comme demandé (on enlève l'url)
    to_drop = ['url', 'link', 'publisher', 'description']
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    # Règle de majuscule
    if 'title' in df.columns:
        df['title'] = df['title'].astype(str).apply(lambda x: x.capitalize())

    print(f"\nTraitement de {file_name}...")
    texts = df[text_column].astype(str).apply(lambda x: x[:500]).tolist()

    # Sentiment (Ligne par ligne pour garantir le mapping)
    final_scores = []
    print("Analyse des sentiments...")
    for res in tqdm(sentiment_pipe(texts), total=len(texts)):
        final_scores.append(round(get_numeric_score(res), 4))

    # Thèmes (Batch autorisé ici car pas de mapping complexe)
    print("Analyse des thèmes...")
    theme_results = []
    for res in tqdm(theme_pipe(texts, candidate_labels=CANDIDATE_LABELS, batch_size=BATCH_SIZE), total=len(texts)):
        theme_results.append(res['labels'][0])

    df['sentiment_score'] = final_scores
    df['main_theme'] = theme_results
    
    return df

# Lancement ('ecb_monetary_policy_decisions_2020_2026.csv', 'text'),('french_economy_news.csv', 'title'),
for file, col in [
                  
                  ('german_economy_news.csv', 'title')]:
    res_df = process_final(file, col)
    if res_df is not None:
        res_df.to_csv(os.path.join(DATA_DIR, f"final_{file}"), index=False)

print("\n--- Terminé ! Vérifie final_french_economy_news.csv ---")