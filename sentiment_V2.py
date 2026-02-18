######## VERSION DU SENTIMENT AVEC TRADUCTION PUIS FINBERT #########

import pandas as pd
import os
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

# 1. Configuration 
DATA_DIR = "data" 
BATCH_SIZE = 32  

# Détection GPU
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Initialisation sur : {'GPU' if device == 0 else 'CPU'} ({dtype})")

# Modèles
MODEL_SENTIMENT = "ProsusAI/finbert"
MODEL_TRAD_FR_EN = "Helsinki-NLP/opus-mt-fr-en"
MODEL_TRAD_DE_EN = "Helsinki-NLP/opus-mt-de-en"

# 2. Chargement des modèles
print("Chargement des modèles")

# Pipeline Sentiment (FinBERT)
tokenizer_finbert = AutoTokenizer.from_pretrained(MODEL_SENTIMENT)
sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_SENTIMENT, tokenizer=tokenizer_finbert, device=device, torch_dtype=dtype)

# Pipelines Traduction
trad_fr_pipe = pipeline("translation", model=MODEL_TRAD_FR_EN, device=device, torch_dtype=dtype)
trad_de_pipe = pipeline("translation", model=MODEL_TRAD_DE_EN, device=device, torch_dtype=dtype)

# 3. Fonctions de traitement
def map_finbert(result):
    """Transforme les labels de finbert (positive/negative/neutral) en score dans [-1, 1]"""
    label = result['label'].lower()
    score = result['score']
    if label == 'negative': return -score
    if label == 'positive': return score
    return 0.0

def translate_batches(texts, trad_pipe):
    """Traduit par lots sur GPU"""
    translated = []
    for out in tqdm(trad_pipe(texts, batch_size=16, truncation=True), total=len(texts), desc="Traduction"):
        translated.append(out['translation_text'])
    return translated

def process_and_score(file_name, text_col, lang):
    """Traduit et score un fichier complet"""
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"Fichier introuvable : {path}")
        return None

    print(f"\n--- Traitement de {file_name} ({lang.upper()}) ---")
    df = pd.read_csv(path)

    # Préparation des textes source
    texts = df[text_col].fillna("").astype(str).str[:400].tolist()

    # 1. Traduction
    pipe_trad = trad_fr_pipe if lang == 'fr' else trad_de_pipe
    translated_texts = translate_batches(texts, pipe_trad)
    df['title_translated'] = translated_texts

    # 2. Sentiment avec FinBERT
    scores = []
    for out in tqdm(sentiment_pipe(translated_texts, batch_size=BATCH_SIZE, truncation=True), total=len(translated_texts), desc="Analyse finbert"):
        scores.append(round(map_finbert(out), 4))
    
    df['sentiment_score'] = scores
    
    # Sauvegarde du fichier individuel (Output 1 & 2)
    output_path = os.path.join(DATA_DIR, f"finbert_clean_{file_name}")
    df.to_csv(output_path, index=False)
    print(f"Exporté : {output_path}")
    
    return df

# 4. Exécution principale et agrégation
# Exécution sur les deux bases
df_fr = process_and_score('articles_fr_final.csv', 'title', 'fr')
df_de = process_and_score('articles_de_final.csv', 'title', 'de')

print("\n--- Création du fichier agrégé ---")
if df_fr is not None and df_de is not None:
    # Agrégation journalière France
    agg_fr = df_fr.groupby('published date').agg(
        nb_articles_fr=('sentiment_score', 'count'),
        sentiment_moyen_fr=('sentiment_score', 'mean')
    )
    
    # Agrégation journalière Allemagne
    agg_de = df_de.groupby('published date').agg(
        nb_articles_de=('sentiment_score', 'count'),
        sentiment_moyen_de=('sentiment_score', 'mean')
    )
    
    # Fusion des deux sur le calendrier
    df_agg = agg_fr.join(agg_de, how='outer')
    
    # Remplissage des jours sans articles par 0 (pour le comptage)
    df_agg['nb_articles_fr'] = df_agg['nb_articles_fr'].fillna(0).astype(int)
    df_agg['nb_articles_de'] = df_agg['nb_articles_de'].fillna(0).astype(int)

    # Sauvegarde du fichier agrégé
    output_agg = os.path.join(DATA_DIR, "aggregated_daily_finbert.csv")
    df_agg.to_csv(output_agg)
    print(f"Fichier agrégé exporté : {output_agg}")
    print(df_agg.head())

print("\nPipeline terminé avec succès !")