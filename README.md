# Projet PESSD

*Spreads souverains, narratif médiatique et fondamentaux macroéconomiques*

## Contexte

Ce projet analyse l’influence du sentiment économique médiatique sur le spread souverain OAT/Bund 10 ans entre la France et l’Allemagne.
L’objectif est de quantifier la part de spread expliquée par les indices de sentiment de presse et par les communications de politique monétaire, en comparaison avec les fondamentaux macroéconomiques.

## Objectif

- Collecter et filtrer des articles économiques français et allemands.
- Extraire un indice de sentiment texto à partir de textes traduits.
- Relier ces indices au spread OAT/Bund.
- Explorer une éventuelle prime de risque narrative indépendante des fondamentaux.

## Structure du dépôt

```text
.
├── compute_residuals.ipynb
├── data/
│   ├── news_processed/
│   ├── outputs/
│   ├── processed/
│   └── raw/
├── README.md
├── sentiment_viz.ipynb
├── src/
│   ├── 1_scraping/
│   └── 2_NLP/
└── uv.lock
```

## Contenu principal

### Scripts

- `src/1_scraping/run_scrapping.py`
  - Extrait les articles GNews.
  - Filtre par domaine et mots-clés.
  - Génère un jeu de données brut d’articles pertinents.

- `src/1_scraping/scrapping_articles_fun.py`
  - Définit l’extraction GNews, le scoring pondéré et le filtrage des résultats.

- `src/1_scraping/exclude_bad_articles.py`
  - Élimine des articles dont le titre ne concerne pas le pays attendu.

- `src/2_NLP/sentiment_finbert.py`
  - Traduit les titres FR/DE en anglais.
  - Applique le modèle `ProsusAI/finbert` pour scorer le sentiment.
  - Exporte les résultats traités dans `data/news_processed/`.

### Notebooks

- `compute_residuals.ipynb`
  - Analyse des résidus et du spread.

- `sentiment_viz.ipynb`
  - Visualisations des indices de sentiment et du spread.

## Données disponibles

### Données brutes (`data/raw/`)

- `articles_fr_final2025.csv`
- `articles_de_final2025.csv`
- `ecb_monetary_policy_decisions_2020_2026.csv`
- `macro_processed.csv`
- `Rendement de l'Obligation Allemagne 10 ans - Données Historiques.csv`
- `Rendement de l'Obligation France 10 ans - Données Historiques (1).csv`

### Données traitées

- `data/news_processed/finbert_articles_fr_final2026.csv`
- `data/news_processed/finbert_articles_de_final2026.csv`
- `data/news_processed/aggregated_daily_finbert.csv`
- `data/outputs/epsilon_spread.csv`

## Pipeline général

1. Collecte d’articles économiques FR/DE via GNews.
2. Filtrage de domaine et scoring par mots-clés.
3. Nettoyage des articles non pertinents.
4. Traduction FR/DE → EN.
5. Scoring de sentiment avec FinBERT.
6. Agrégation des scores journaliers.
7. Analyse du spread OAT/Bund et calcul de la prime narrative.

## Installation

Utilisez Python 3.10+.

```bash
python -m pip install pandas numpy torch transformers tqdm sentencepiece gnews
```

Dependencies utiles selon le code existant :

- `pandas`
- `numpy`
- `torch`
- `transformers`
- `tqdm`
- `sentencepiece`
- `gnews`

Packages optionnels utiles :

- `matplotlib`
- `seaborn`
- `scikit-learn`
- `beautifulsoup4`

## Exécution rapide

1. Préparez un environnement Python 3.10+.
2. Placez les fichiers bruts dans `data/raw/`.
3. Lancez `python src/1_scraping/run_scrapping.py`.
4. Lancez `python src/2_NLP/sentiment_finbert.py`.
5. Ouvrez `compute_residuals.ipynb` et `sentiment_viz.ipynb`.

## Notes

- Le code actuel utilise les jeux de données `articles_fr_final2025.csv` et `articles_de_final2025.csv`.
- Les titres FR/DE sont traduits avant d’être scorés avec FinBERT.
- Le dépôt est structuré pour séparer collecte, NLP et analyse.

## Objectif de la recherche

Le projet cherche à mesurer la contribution du sentiment médiatique au spread OAT/Bund et à identifier une éventuelle prime de risque narrative au-delà des fondamentaux économiques.

