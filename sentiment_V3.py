import pandas as pd
import os
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & GPU SETUP
# ==========================================
DATA_DIR = "data" # Assure-toi que ce dossier existe dans ton répertoire courant
BATCH_SIZE = 32   # Profite de la mémoire de ton GPU (T4 ou L4 sur Onyxia)

# Modèle multilingue de référence (XLM-RoBERTa)
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Détection automatique du hardware
device = 0 if torch.cuda.is_available() else -1
# Utilisation de la demi-précision (float16) pour doubler la vitesse sur GPU
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"🚀 Moonshot Engine initialisé sur : {'GPU' if device == 0 else 'CPU'} ({dtype})")

# ==========================================
# 2. CHARGEMENT DU MODÈLE
# ==========================================
# On force use_fast=False pour éviter les erreurs de parsing sentencepiece
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=device,
    torch_dtype=dtype
)

# ==========================================
# 3. LOGIQUE DE MAPPING & FORMATAGE
# ==========================================
def map_sentiment(result):
    """Mappe LABEL_0/1/2 vers un score continu [-1, 1]"""
    label = result['label'].upper()
    score = result['score']
    if 'LABEL_0' in label or 'NEG' in label: return -score
    if 'LABEL_2' in label or 'POS' in label: return score
    return 0.0

def process_file(file_name, text_col):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"❌ Fichier non trouvé : {path}")
        return

    print(f"\n⚡ Traitement de {file_name}...")
    df = pd.read_csv(path)
    
    # Règle de majuscule ENSAE : Seul le premier mot du titre (vectorisé)
    if 'title' in df.columns:
        df['title'] = df['title'].astype(str).str.capitalize()

    # Nettoyage des colonnes pour ton modèle de spread
    # On enlève l'url, le publisher brut, etc.
    cols_to_drop = ['url', 'link', 'description', 'guid']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Préparation des textes (tronqués pour BERT)
    texts = df[text_col].fillna("").astype(str).str[:500].tolist()

    # INFÉRENCE EN BATCH
    scores = []
    for out in tqdm(sentiment_pipe(texts, batch_size=BATCH_SIZE, truncation=True), total=len(texts)):
        scores.append(round(map_sentiment(out), 4))
    
    df['sentiment_score'] = scores
    
    # Sauvegarde dans le dossier data
    output_path = os.path.join(DATA_DIR, f"final_{file_name}")
    df.to_csv(output_path, index=False)
    print(f"✅ Exporté : {output_path}")

# ==========================================
# 4. EXÉCUTION MASSIVE
# ==========================================
# Liste de tes fichiers et de la colonne de texte cible
tasks = [
    ('ecb_monetary_policy_decisions_2020_2026.csv', 'text'),
    ('articles_fr_final.csv', 'title'),
    ('articles_de_final.csv', 'title')
]

for file, col in tasks:
    process_file(file, col)

print("\n--- ✨ Mission terminée ! Tes données sont prêtes pour l'économétrie ---")