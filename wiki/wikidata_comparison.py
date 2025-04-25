import time
from wikidata_data_fetch import fetch_wikidata
from wikidata_mysql_storage import store_to_mysql
from wikidata_mysql_query import query_mysql
from wikidata_neo4j_storage import store_to_neo4j
from wikidata_neo4j_query import query_neo4j


def get_large_entity_ids():
    # 这里可以根据需要添加更多的实体 ID
    entity_ids = []
    with open('entity_ids.txt', 'r') as f:
        for line in f:
            entity_ids.append(line.strip())
    return entity_ids


def compare_storage_efficiency():
    entity_ids = get_large_entity_ids()
    data = fetch_wikidata(entity_ids)

    start_mysql = time.time()
    store_to_mysql(data)
    end_mysql = time.time()
    mysql_storage_time = end_mysql - start_mysql

    start_neo4j = time.time()
    store_to_neo4j(data)
    end_neo4j = time.time()
    neo4j_storage_time = end_neo4j - start_neo4j

    print(f"MySQL 存储时间: {mysql_storage_time} 秒")
    print(f"Neo4j 存储时间: {neo4j_storage_time} 秒")


def compare_query_performance():
    entity_id = "Q148"
    property_id = "P36"

    start_mysql = time.time()
    mysql_result = query_mysql(entity_id, property_id)
    end_mysql = time.time()
    mysql_query_time = end_mysql - start_mysql

    start_neo4j = time.time()
    neo4j_result = query_neo4j(entity_id, property_id)
    end_neo4j = time.time()
    neo4j_query_time = end_neo4j - start_neo4j

    print(f"MySQL 查询时间: {mysql_query_time} 秒，查询结果: {mysql_result}")
    print(f"Neo4j 查询时间: {neo4j_query_time} 秒，查询结果: {neo4j_result}")


if __name__ == "__main__":
    compare_storage_efficiency()
    compare_query_performance()

    