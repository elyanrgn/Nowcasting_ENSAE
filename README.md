# Projet PESSD  
*Spreads souverains, narratif médiatique et fondamentaux macroéconomiques*

## 1. Description du projet

Ce projet étudie dans quelle mesure le sentiment exprimé par la presse économique influence le spread souverain OAT/Bund à 10 ans, indicateur du différentiel de risque entre la France et l’Allemagne. Il vise à comparer le pouvoir explicatif de cette composante « narrative » à celui des fondamentaux macroéconomiques afin d’estimer la part de spread imputable aux narratifs médiatiques plutôt qu’à la réalité économique observée. 

L’approche combine économie, sociologie et science des données : analyse textuelle d’articles de presse, traitement automatique du langage (NLP) et modélisation économétrique des spreads souverains. 

## 2. Question de recherche

> Dans quelle mesure le sentiment exprimé par la presse économique influence-t-il le spread OAT/Bund, et quel est le poids explicatif des informations textuelles par rapport aux fondamentaux macroéconomiques dans les fluctuations de ce spread ? 
L’objectif est de quantifier une **prime** de risque narrative, définie comme le surplus de spread OAT/Bund non justifié par les fondamentaux macroéconomiques.

## 3. Données utilisées

- Presse économique  
  - Articles français (mot-clé « Economie », langue FR, pays France) et allemands (mot-clé « Wirtschaft », langue DE, pays Allemagne) collectés via le module Python GNews.
  - Fréquence : collecte hebdomadaire, jusqu’à 50 articles par pays et par semaine sur la période 2020–2026.
  - Biais assumé vers les grands médias, cohérent avec les narratifs dominants effectivement exposés au public.

- Communiqués de la BCE  
  - Communiqués de presse relatifs aux décisions de politique monétaire (environ toutes les 6 semaines) depuis janvier 2020.
  - 51 documents récupérés via scraping (BeautifulSoup) à partir du site de la BCE. 
- Marchés obligataires  
  - Taux à 10 ans OAT française et Bund allemande (clôture journalière) récupérés gratuitement sur Investing.com depuis le 1er janvier 2020. 
  - Construction du spread OAT/Bund 10 ans, avec analyse de stationnarité (test ADF, différences pré- et post‑Covid).

- Fondamentaux macroéconomiques  
  - Indicateurs standards pour la France et l’Allemagne : inflation (PCE), PIB, taux de chômage, dette, issus de la base FRED.
  - Fréquences hétérogènes (annuelle, trimestrielle, mensuelle) nécessitant une harmonisation temporelle.
## 4. Méthodologie

### 4.1 Harmonisation temporelle

- Passage à une fréquence hebdomadaire commune :  
  - Moyenne hebdomadaire du spread OAT/Bund pour lisser la volatilité intra‑journalière. 
  - Interpolation linéaire des données macroéconomiques de basse fréquence. 
### 4.2 Construction de l’indice de sentiment

Transformation du texte en série temporelle de sentiment $Sent_t$ via deux approches concurrentes : 

1. Approche classifieur (fine‑tuning)  
   - Utilisation de modèles encodeurs CamemBERT (FR) et GBERT (DE). 
   - Fine‑tuning sur un corpus financier annoté pour classifier le sentiment de chaque titre (positif/négatif).
2. Approche LLM (générative)  
   - Interrogation de modèles de langage massifs par prompting pour obtenir un score de sentiment, mieux adapté aux nuances sémantiques et constructions complexes. 
Les scores sont agrégés au niveau hebdomadaire pour produire un indice global de sentiment de presse $Sent^{Presse}_t$, ainsi qu’un indice de sentiment pour les communiqués de la BCE $Sent^{BCE}_t$.

### 4.3 Modèle économétrique

Le spread hebdomadaire $Y_t$ est modélisé comme suit :

```math
Y_t = \alpha + \beta_{\text{Fund}} X_{t-1} + \beta_{\text{Narr}}\,Sent^{\text{Presse}}_{t}
      + \beta_{\text{BCE}}\,Sent^{\text{BCE}}_{t} + \gamma Y_{t-1} + \varepsilon_t


- $Y_t$ : spread OAT/Bund à la semaine $t$.
- $X_{t-1}$ : fondamentaux macroéconomiques retardés (pour limiter les biais de simultanéité).
- $Sent^{Presse}_t$ : sentiment médiatique contemporain.
- $Sent^{BCE}_{t}$ : sentiment des communiqués de politique monétaire (0 en absence de communiqué).
- $Y_{t-1}$ : composante autoregressive capturant la persistance du spread.

```
### 4.4 Prime de risque narrative

- Construction d’un spread contrefactuel $\hat{Y}^{NoNews}_t$ en imposant $Sent^{Presse}_t = 0$ .
- La différence $Y_t - \hat{Y}^{NoNews}_t$ mesure la prime de risque narrative, i.e. le surplus de coût de financement lié aux perceptions médiatiques plutôt qu’aux fondamentaux. 

### 4.5 Analyses de robustesse

- Tests de causalité de Granger pour vérifier l’endogénéité inverse entre spread et sentiment.
- Vérification de l’orthogonalité entre sentiment et fondamentaux macroéconomiques. 
- Discussion des biais liés à l’algorithme de sélection de Google News (concentration sur quelques grands médias).

## 5. Organisation du dépôt

Suggestion de structure de projet :

```text
.
├── data/
│   ├── raw/
│   │   ├── presse_fr/
│   │   ├── presse_de/
│   │   ├── ecb_press/
│   │   ├── rates_oat_bund/
│   │   └── macro_fred/
│   └── processed/
├── notebooks/
│   ├── 01_exploration_spread.ipynb
│   ├── 02_scraping_presse.ipynb
│   ├── 03_nlp_sentiment.ipynb
│   └── 04_modele_econometrique.ipynb
├── src/
│   ├── data_collection/
│   ├── nlp/
│   ├── models/
│   └── utils/
├── figures/
│   ├── spread_oat_bund_2020_2025.png
│   └── spread_oat_bund_2025.png
├── references/
│   └── bibliography.bib
└── README.md
```

## 6. Prérequis techniques

- Python (>= 3.10)  
- Bibliothèques principales :  
  - Scraping et API : `gnews`, `feedparser`, `requests`, `beautifulsoup4`. 
  - NLP : `transformers`, `torch`, `sentencepiece`, `scikit-learn`. 
  - Économétrie et manipulation de données : `pandas`, `numpy`, `statsmodels`, `matplotlib`, `seaborn`.
Installation possible via :

```bash
pip install -r requirements.txt
```

## 7. Exécution rapide

1. Télécharger et mettre en forme les données (scripts de `src/data_collection`).  
2. Construire le corpus textuel nettoyé et indexé dans `data/processed`.  
3. Entraîner/évaluer les modèles de sentiment (CamemBERT, GBERT, LLM) dans `src/nlp` ou via les notebooks dédiés.  
4. Générer les séries temporelles hebdomadaires (spread, fondamentaux, indices de sentiment).  
5. Estimer le modèle économétrique et calculer la prime de risque narrative, puis produire les figures finales.
