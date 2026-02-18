import pandas as pd
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & FILTRES DE TEST
# ==========================================
DATA_DIR = "data" 
BATCH_SIZE = 32  

# --- PARAMÈTRES DE TEST (DATES) ---
START_DATE = "2020-01-01" 
END_DATE = "2020-01-31"   
# ----------------------------------

device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Initialisation sur : {'GPU' if device == 0 else 'CPU'} ({dtype})")

MODEL_SENTIMENT = "ProsusAI/finbert"
MODEL_TRAD_FR_EN = "Helsinki-NLP/opus-mt-fr-en"
MODEL_TRAD_DE_EN = "Helsinki-NLP/opus-mt-de-en"

# ==========================================
# 2. CHARGEMENT DES MODÈLES
# ==========================================
print("📥 Chargement des modèles en mémoire (Les warnings 'UNEXPECTED' sont normaux)...")

# 1. Pipeline Sentiment 
tokenizer_finbert = AutoTokenizer.from_pretrained(MODEL_SENTIMENT)
sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_SENTIMENT, tokenizer=tokenizer_finbert, device=device, torch_dtype=dtype)

# 2. Traduction 
print("Chargement Traduction FR-EN...")
tok_fr = AutoTokenizer.from_pretrained(MODEL_TRAD_FR_EN)
mod_fr = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TRAD_FR_EN).to(device)
if dtype == torch.float16: mod_fr = mod_fr.half() # Optimisation GPU

print("Chargement Traduction DE-EN...")
tok_de = AutoTokenizer.from_pretrained(MODEL_TRAD_DE_EN)
mod_de = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TRAD_DE_EN).to(device)
if dtype == torch.float16: mod_de = mod_de.half()

# ==========================================
# 3. FONCTIONS DE TRAITEMENT
# ==========================================
def map_finbert(result):
    label = result['label'].lower()
    score = result['score']
    if label == 'negative': return -score
    if label == 'positive': return score
    return 0.0

def translate(texts, tokenizer, model):
    """Traduction"""
    translated = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Traduction"):
        batch = texts[i:i+batch_size]
        
        # Préparation des tenseurs pour le GPU
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=400).to(device)
        
        # Génération de la traduction
        with torch.no_grad():
            outputs = model.generate(**inputs)
            
        # Décodage en texte lisible
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(decoded)
        
    return translated

def process_and_score(file_name, text_col, lang):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"Fichier introuvable : {path}")
        return None

    print(f"\n--- Traitement de {file_name} ({lang.upper()}) ---")
    df = pd.read_csv(path)

    col_date = 'published date' if 'published date' in df.columns else 'date'
    df['date_clean'] = pd.to_datetime(df[col_date], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    
    if START_DATE and END_DATE:
        mask = (df['date_clean'] >= pd.to_datetime(START_DATE)) & (df['date_clean'] <= pd.to_datetime(END_DATE))
        df = df.loc[mask].copy()
        print(f"Filtre : {len(df)} articles entre {START_DATE} et {END_DATE}")
    
    if df.empty:
        print("Aucun article sur cette période.")
        return None

    texts = df[text_col].fillna("").astype(str).str[:400].tolist()

    # 1. Traduction Robuste
    if lang == 'fr':
        translated_texts = translate(texts, tok_fr, mod_fr)
    else:
        translated_texts = translate(texts, tok_de, mod_de)
        
    df['title_translated'] = translated_texts

    # 2. Sentiment FinBERT
    scores = []
    for out in tqdm(sentiment_pipe(translated_texts, batch_size=BATCH_SIZE, truncation=True), total=len(translated_texts), desc="Analyse FinBERT"):
        scores.append(round(map_finbert(out), 4))
    
    df['sentiment_score'] = scores
    
    output_path = os.path.join(DATA_DIR, f"finbert_{file_name.replace('.csv', '')}.csv")
    df.to_csv(output_path, index=False)
    print(f"Exporté : {output_path}")
    
    return df

# ==========================================
# 4. EXÉCUTION & AGRÉGATION
# ==========================================
df_fr = process_and_score('articles_fr_final.csv', 'title', 'fr')
df_de = process_and_score('articles_de_final.csv', 'title', 'de')

print("\n--- Création du fichier agrégé ---")
if df_fr is not None and df_de is not None:
    agg_fr = df_fr.groupby('date_clean').agg(
        nb_articles_fr=('sentiment_score', 'count'),
        sentiment_moyen_fr=('sentiment_score', 'mean')
    )
    
    agg_de = df_de.groupby('date_clean').agg(
        nb_articles_de=('sentiment_score', 'count'),
        sentiment_moyen_de=('sentiment_score', 'mean')
    )
    
    df_agg = agg_fr.join(agg_de, how='outer')
    
    df_agg['nb_articles_fr'] = df_agg['nb_articles_fr'].fillna(0).astype(int)
    df_agg['nb_articles_de'] = df_agg['nb_articles_de'].fillna(0).astype(int)

    output_agg = os.path.join(DATA_DIR, f"aggregated_daily_finbert.csv")
    df_agg.to_csv(output_agg)
    print(f"Fichier agrégé exporté : {output_agg}")

print("\nTerminé !")