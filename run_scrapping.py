import pandas as pd
import json
from scrapping_articles_fun import extract_news_fixed
from exclude_bad_articles import remove_articles_not_about_country

# KEYWORDS USED FOR SCRAPPING ARTICLES FROM GNEWS

DOMAIN_PATTERNS_FR = [
    "lemonde",
    "figaro",
    "echos",
    "tribune",
    "liberation",
    "opinion",
    "capital",
    "challenges",
    "boursorama",
    "parisien",
    "croix",
    "express",
    "obs",
    "nouvelobs",
    "bfm",
    "francetvinfo",
    "franceinfo",
    "20minutes",
    "lepoint",
    "marianne",
    "mediapart",
]

DOMAIN_PATTERNS_DE = [
    "handelsblatt",
    "faz",
    "sueddeutsche",
    "welt",
    "spiegel",
    "wiwo",
    "manager-magazin",
    "capital",
    "boersen-zeitung",
    "zeit",
    "tagesschau",
    "ntv",
    "focus",
    "stern",
    "tagesspiegel",
]

BROAD_QUERIES_FR = [
    "économie",
    "finance",
    "politique",
    "France",
]

BROAD_QUERIES_DE = [
    "wirtschaft",
    "finanzen",
    "politik",
    "Deutschland",
]

# Mots-clés complets (version précédente avec TPE/PME/énergie/etc.)
SCORING_KEYWORDS_FR = {
    # Politique monétaire
    "bce": 5,
    "banque centrale européenne": 5,
    "banque centrale": 3,
    "lagarde": 3,
    "christine lagarde": 3,
    "politique monétaire": 5,
    "taux directeur": 5,
    "taux d'intérêt": 4,
    # Inflation & prix
    "inflation": 5,
    "désinflation": 3,
    "ipc": 3,
    "hausse des prix": 3,
    "prix": 1,
    "prix de l'énergie": 3,
    # Énergie
    "énergie": 2,
    "électricité": 2,
    "prix électricité": 3,
    "gaz": 2,
    "prix du gaz": 3,
    "pétrole": 2,
    "prix du pétrole": 3,
    "carburant": 2,
    "essence": 1,
    # Dette & Budget
    "dette publique": 5,
    "dette": 3,
    "déficit budgétaire": 5,
    "déficit": 4,
    "budget": 3,
    "loi de finances": 4,
    "finances publiques": 4,
    # Obligations
    "oat": 6,
    "obligations": 4,
    "emprunt d'état": 4,
    "trésor": 3,
    "spread": 6,
    "prime de risque": 5,
    "rendement": 3,
    # Notation
    "notation": 4,
    "moody's": 3,
    "fitch": 3,
    "s&p": 3,
    "dégradation": 3,
    # Macro
    "pib": 3,
    "croissance": 2,
    "récession": 3,
    "ralentissement": 2,
    "chômage": 2,
    "emploi": 2,
    "salaires": 2,
    "pouvoir d'achat": 2,
    # Entreprises
    "tpe": 2,
    "pme": 2,
    "petites entreprises": 2,
    "faillites": 2,
    "investissement": 2,
    "production industrielle": 2,
    # Commerce
    "commerce extérieur": 2,
    "exportations": 2,
    "importations": 2,
    "tarifs douaniers": 3,
    "douanes": 2,
    "protectionnisme": 2,
    "tensions commerciales": 3,
    # Politique
    "gouvernement": 2,
    "assemblée nationale": 2,
    "élections": 2,
    "réforme": 2,
    "instabilité politique": 3,
    # Marchés
    "marchés financiers": 2,
    "bourse": 2,
    "cac 40": 2,
    "banques": 2,
    # Crises
    "crise": 2,
    "tension": 2,
    "incertitude": 2,
    "risque": 2,
}

SCORING_KEYWORDS_DE = {
    # Politique monétaire
    "ezb": 5,
    "europäische zentralbank": 5,
    "zentralbank": 3,
    "lagarde": 3,
    "geldpolitik": 5,
    "leitzins": 5,
    "zinsen": 4,
    # Inflation
    "inflation": 5,
    "inflationsrate": 4,
    "verbraucherpreise": 3,
    "preisanstieg": 3,
    "preise": 1,
    # Énergie
    "energie": 2,
    "energiepreise": 3,
    "strom": 2,
    "strompreise": 3,
    "gas": 2,
    "gaspreise": 3,
    "öl": 2,
    "ölpreis": 3,
    "kraftstoff": 2,
    "benzin": 1,
    # Dette & Budget
    "staatsschulden": 5,
    "schulden": 3,
    "haushaltsdefizit": 5,
    "defizit": 4,
    "haushalt": 3,
    "finanzpolitik": 3,
    "schuldenbremse": 4,
    # Obligations
    "bundesanleihe": 6,
    "bund": 4,
    "anleihen": 4,
    "staatsanleihe": 4,
    "spread": 6,
    "risikoprämie": 5,
    "rendite": 3,
    # Notation
    "rating": 4,
    "moody's": 3,
    "fitch": 3,
    "s&p": 3,
    "herabstufung": 3,
    # Macro
    "bip": 3,
    "wachstum": 2,
    "rezession": 3,
    "konjunktur": 2,
    "arbeitslosigkeit": 2,
    "beschäftigung": 2,
    "löhne": 2,
    "kaufkraft": 2,
    # Entreprises
    "kmu": 2,
    "kleine unternehmen": 2,
    "mittelstand": 2,
    "insolvenzen": 2,
    "investitionen": 2,
    "industrieproduktion": 2,
    # Commerce
    "außenhandel": 2,
    "exporte": 2,
    "importe": 2,
    "zölle": 3,
    "protektionismus": 2,
    "handelsspannungen": 3,
    # Politique
    "bundesregierung": 2,
    "bundestag": 2,
    "wahlen": 2,
    "reform": 2,
    "regierungskrise": 3,
    # Marchés
    "finanzmärkte": 2,
    "börse": 2,
    "dax": 2,
    "banken": 2,
    # Crises
    "krise": 2,
    "spannung": 2,
    "unsicherheit": 2,
    "risiko": 2,
}

# May require the use of a VPN or proxy if GNews blocks requests from your IP. Adjust the date range and keywords as needed to get a good number of relevant articles.
articles_fr, _ = extract_news_fixed(
    broad_queries=BROAD_QUERIES_FR,
    scoring_keywords=SCORING_KEYWORDS_FR,
    domain_patterns=DOMAIN_PATTERNS_FR,
    start_date="2025-01-01",
    end_date="2025-12-31",
    language="fr",
    country="FR",
    min_score=2,
    period_days=7,
    verbose=True,
)

if articles_fr:
    print("\nTop 15 articles (meilleurs scores):")
    for i, a in enumerate(articles_fr[:15]):
        print(f"\n{i + 1}. [Score {a['relevance_score']}] [{a['source']}]")
        print(f"{a['title']}")
        print(f"KW: {', '.join(a['matched_keywords'][:5])}")
    else:
        print("\nToujours 0 articles - vérifier les requêtes GNews")

with open("articles_fr_final2026.json", "w", encoding="utf-8") as f:
    json.dump(articles_fr, f, ensure_ascii=False, indent=2)

articles_fr = pd.read_json("articles_fr_final2026.json")
articles_fr["date"] = pd.to_datetime(articles_fr["published"], errors="coerce")
articles_fr.index = articles_fr["date"]
articles_fr.sort_index(inplace=True)
articles_fr = remove_articles_not_about_country(articles_fr, "France", language="fra")
articles_fr.to_csv("data\\raw\\articles_fr_final2026.csv", index=True)


articles_de, _ = extract_news_fixed(
    broad_queries=BROAD_QUERIES_DE,
    scoring_keywords=SCORING_KEYWORDS_DE,
    domain_patterns=DOMAIN_PATTERNS_DE,
    start_date="2025-01-01",
    end_date="2025-12-31",
    language="de",
    country="DE",
    min_score=2,
    period_days=7,
    verbose=True,
)

if articles_de:
    print("\nTop 15 articles (meilleurs scores):")
    for i, a in enumerate(articles_de[:15]):
        print(f"\n{i + 1}. [Score {a['relevance_score']}] [{a['source']}]")
        print(f"{a['title']}")
        print(f"KW: {', '.join(a['matched_keywords'][:5])}")
    else:
        print("\nToujours 0 articles - vérifier les requêtes GNews")

with open("articles_de_final2026.json", "w", encoding="utf-8") as f:
    json.dump(articles_de, f, ensure_ascii=False, indent=2)

articles_de = pd.read_json("articles_de_final2026.json")
articles_de["date"] = pd.to_datetime(articles_de["published"], errors="coerce")
articles_de.index = articles_de["date"]
articles_de.sort_index(inplace=True)
articles_de = remove_articles_not_about_country(
    articles_de, "Deutschland", language="deu"
)
articles_de.to_csv("data\\raw\\articles_de_final2026.csv", index=True)
