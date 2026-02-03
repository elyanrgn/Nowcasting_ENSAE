import pandas as pd
import os
from transformers import pipeline, AutoTokenizer
import torch
from tqdm import tqdm

# 1. Gestion des chemins (Dossier data)
DATA_DIR = "data"
ecb_path = os.path.join(DATA_DIR, 'ecb_monetary_policy_decisions_2020_2026.csv')
fr_path = os.path.join(DATA_DIR, 'french_economy_news.csv')
de_path = os.path.join(DATA_DIR, 'german_economy_news.csv')

# 2. Chargement des modèles
# Utilisation de modèles "base" extrêmement stables pour éviter les erreurs 401/Repository Not Found
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment" 
THEME_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" # Excellent pour le multilingue FR/DE

print("Initialisation des modèles...")
# On force use_fast=False pour éviter ton bug initial
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, use_fast=False)
sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=tokenizer)
theme_pipe = pipeline("zero-shot-classification", model=THEME_MODEL)

# 3. Fonction de traitement avec ta règle de majuscules
def process_econometrics_data(file_path, text_column, is_news=True):
    if not os.path.exists(file_path):
        print(f"Fichier non trouvé : {file_path}")
        return None
        
    df = pd.read_csv(file_path)
    
    # Règle : Seul le premier mot du titre a une majuscule
    if 'title' in df.columns:
        df['title'] = df['title'].astype(str).apply(lambda x: x.capitalize())

    print(f"Analyse de {os.path.basename(file_path)}...")
    sentiments = []
    themes = []
    candidate_labels = ["inflation", "public debt", "interest rates", "growth"]

    for text in tqdm(df[text_column].astype(str)):
        # Troncature manuelle pour la stabilité
        clean_text = text[:500] 
        
        # Sentiment
        s = sentiment_pipe(clean_text)[0]
        # Conversion en score numérique (-1 à 1)
        score = s['score'] if s['label'] == 'Positive' else (-s['score'] if s['label'] == 'Negative' else 0)
        sentiments.append(score)
        
        # Thème
        t = theme_pipe(clean_text, candidate_labels=candidate_labels)
        themes.append(t['labels'][0])

    df['sentiment_score'] = sentiments
    df['main_theme'] = themes
    return df

# 4. Exécution
ecb_final = process_econometrics_data(ecb_path, 'text', is_news=False)
fr_final = process_econometrics_data(fr_path, 'title')
de_final = process_econometrics_data(de_path, 'title')

# 5. Sauvegarde dans le dossier data
if ecb_final is not None:
    ecb_final.to_csv(os.path.join(DATA_DIR, 'ecb_analyzed.csv'), index=False)
if fr_final is not None:
    fr_final.to_csv(os.path.join(DATA_DIR, 'fr_analyzed.csv'), index=False)
if de_final is not None:
    de_final.to_csv(os.path.join(DATA_DIR, 'de_analyzed.csv'), index=False)

print("\nTraitement terminé. Les fichiers sont dans le dossier /data.")