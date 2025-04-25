import requests
import time


def fetch_wikidata(entity_ids):
    id_chunks = [entity_ids[i:i + 50] for i in range(0, len(entity_ids), 50)]
    all_data = {}
    for chunk in id_chunks:
        id_string = '|'.join(chunk)
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={id_string}&format=json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            all_data.update(data.get('entities', {}))
        except requests.RequestException as e:
            print(f"请求出错: {e}")
        time.sleep(1)  # 避免请求过于频繁
    return all_data
    