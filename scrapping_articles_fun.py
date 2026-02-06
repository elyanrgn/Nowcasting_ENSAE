import time
from datetime import datetime, timedelta
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from gnews import GNews


def extract_real_domain(article):
    """
    Tente d'extraire le vrai domaine depuis publisher['href'].
    """
    publisher = article.get("publisher", {})

    # Si publisher est un dict avec 'href'
    if isinstance(publisher, dict):
        href = publisher.get("href", "")
        if href:
            try:
                netloc = urlparse(href).netloc.lower().replace("www.", "")
                return netloc
            except:
                pass

    # Fallback : essayer l'URL principale (mais ce sera news.google.com)
    url = article.get("url", "")
    try:
        netloc = urlparse(url).netloc.lower().replace("www.", "")
        return netloc
    except:
        return ""


def domain_matches_pattern(domain, patterns):
    """Vérifie si le domaine contient un des patterns"""
    domain_lower = domain.lower()
    for pattern in patterns:
        if pattern.lower() in domain_lower:
            return True, pattern
    return False, None


def calculate_relevance_score(text, keywords_dict):
    """Score pondéré"""
    text_lower = text.lower()
    score = 0
    matched = []

    for keyword, weight in keywords_dict.items():
        if keyword in text_lower:
            count = min(text_lower.count(keyword), 3)
            score += weight * count
            matched.append(keyword)

    return score, matched


def extract_news_fixed(
    broad_queries,
    scoring_keywords,
    domain_patterns,
    start_date="2020-01-01",
    end_date=None,
    language="fr",
    country="FR",
    period_days=7,
    max_results=100,
    min_score=2,
    max_workers=6,
    verbose=True,
):
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    all_articles = []
    seen_urls = set()

    stats = {
        "total_scraped": 0,
        "rejected_domain": 0,
        "rejected_score": 0,
        "accepted": 0,
    }

    def fetch_query(query, d0, d1):
        try:
            gn = GNews(
                language=language,
                country=country,
                max_results=max_results,
                start_date=d0,
                end_date=d1,
            )
            return gn.get_news(query) or []
        except Exception as e:
            if verbose:
                print(f"    ⚠️ Erreur: {e}")
            return []

    periods = []
    current = start_dt
    while current <= end_dt:
        period_end = min(current + timedelta(days=period_days), end_dt)
        periods.append((current, period_end))
        current = period_end + timedelta(days=1)

    if verbose:
        print("Extraction depuis publisher['href'] (vrai domaine)")
        print(f"{len(periods)} semaines × {len(broad_queries)} requêtes")
        print(f"Patterns: {', '.join(domain_patterns[:8])}...")
        print(f"Seuil: score ≥ {min_score}\n")

    for i, (week_start, week_end) in enumerate(periods):
        if verbose and (i % 20 == 0 or i < 5):
            print(f"📅 Semaine {i + 1}/{len(periods)}: {week_start}")

        batch_articles = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(fetch_query, q, week_start, week_end) for q in broad_queries
            ]

            for fut in as_completed(futures):
                batch_articles.extend(fut.result())

        stats["total_scraped"] += len(batch_articles)

        week_added = 0
        for a in batch_articles:
            url = a.get("url")
            if not url or url in seen_urls:
                continue

            # CORRECTION : Extraire depuis publisher['href']
            real_domain = extract_real_domain(a)

            # Filtre 1: Pattern de domaine
            matches, matched_pattern = domain_matches_pattern(
                real_domain, domain_patterns
            )
            if not matches:
                stats["rejected_domain"] += 1
                continue

            # Filtre 2: Score
            title = a.get("title", "")
            desc = a.get("description", "")
            text = title + " " + desc

            score, matched_kw = calculate_relevance_score(text, scoring_keywords)

            if score < min_score:
                stats["rejected_score"] += 1
                continue

            seen_urls.add(url)
            all_articles.append(
                {
                    "url": url,
                    "title": title,
                    "description": desc,
                    "published": a.get("published date", ""),
                    "publisher": str(a.get("publisher", {})),
                    "source": real_domain,  # Le VRAI domaine
                    "matched_pattern": matched_pattern,
                    "week": week_start.strftime("%Y-%m-%d"),
                    "relevance_score": score,
                    "matched_keywords": matched_kw[:10],
                }
            )
            week_added += 1
            stats["accepted"] += 1

        if verbose and (i % 20 == 0 or i < 5):
            print(
                f"{len(batch_articles)} → {week_added} (rejetés: {stats['rejected_domain']} domaine, {stats['rejected_score']} score)"
            )

        time.sleep(0.5)  # Pause courte pour éviter surcharge

    if verbose:
        print("\nTERMINÉ:")
        print(f"{stats['total_scraped']:,} articles scrapés")
        print(f"{stats['rejected_domain']:,} rejetés (domaine non autorisé)")
        print(f"{stats['rejected_score']:,} rejetés (score < {min_score})")
        print(f"{stats['accepted']:,} acceptés")

        if stats["accepted"] > 0:
            print(f"{stats['accepted'] / len(periods):.1f} articles/semaine")
            scores = [a["relevance_score"] for a in all_articles]
            print(
                f"Scores: min={min(scores)}, max={max(scores)}, moy={sum(scores) / len(scores):.1f}"
            )

    all_articles.sort(key=lambda x: x["relevance_score"], reverse=True)

    return all_articles, [a["title"] for a in all_articles]
