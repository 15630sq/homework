import requests
import json


def get_country_capital_entity_ids():
    sparql_query = '''
    SELECT ?country ?capital WHERE {
      ?country wdt:P31 wd:Q3624078 .  
      ?country wdt:P36 ?capital .  
    }
    '''
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params={"query": sparql_query})
    data = response.json()

    entity_ids = []
    for result in data["results"]["bindings"]:
        country_id = result["country"]["value"].split("/")[-1]
        capital_id = result["capital"]["value"].split("/")[-1]
        entity_ids.extend([country_id, capital_id])

    return entity_ids


def save_to_file(entity_ids, filename="entity_ids.txt"):
    with open(filename, "w") as f:
        for id in entity_ids:
            f.write(f"{id}\n")
    print(f"已保存 {len(entity_ids)} 个实体 ID 到 {filename}")


if __name__ == "__main__":
    entity_ids = get_country_capital_entity_ids()
    save_to_file(entity_ids)
