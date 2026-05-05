from tqdm import tqdm
from pays import Countries


def remove_articles_not_about_country(data, country_name, language="fra"):
    countries = Countries(language)

    idx_to_remove = []

    for idx in tqdm(data.index):
        title = str(data.loc[idx, "title"]).lower()
        title_words = title.split(" ")

        found_other_country = False
        for country in countries:
            country_names = [str(country).lower(), country.official_name.lower()]

            if any(name in title_words for name in country_names):
                if country_name.lower() not in title_words:
                    found_other_country = True
                    break

        if found_other_country:
            idx_to_remove.append(idx)

    print(f"Nombre d'articles supprimés : {len(idx_to_remove)}")
    return data.drop(index=idx_to_remove).reset_index(drop=True)
