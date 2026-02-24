from tqdm import tqdm
from pays import Countries
def remove_articles_not_about_country(data, country_name, language='fra'):
    countries = Countries(language)
    titles = data['title']
    idx_to_remove = []
    for i in tqdm(range(len(titles))):
        for country in countries:
            if country in titles[i].split(' ') or country.official_name in titles[i].split(' ') or str(country).lower() in titles[i].lower().split(' '):
                if country_name.lower() not in titles[i].lower().split(' '):
                    idx_to_remove.append(i)

    print(f"Nombre d'articles supprimés : {len(idx_to_remove)}")
    return data.drop(index=idx_to_remove).reset_index(drop=True)
    
                