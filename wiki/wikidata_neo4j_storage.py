from neo4j import GraphDatabase


def store_to_neo4j(data):
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "chenhongzhou"))

    def create_nodes_and_relationships(tx, entity_id, entity_name, claims):
        # 创建实体节点
        tx.run("MERGE (e:Entity {id: $entity_id, name: $entity_name})",
               entity_id=entity_id, entity_name=entity_name)
        # 只处理首都属性（P36）
        if 'P36' in claims:
            property_id = 'P36'
            for claim in claims[property_id]:
                mainsnak = claim.get('mainsnak', {})
                if mainsnak.get('snaktype') == 'value':
                    datavalue = mainsnak.get('datavalue', {})
                    value = datavalue.get('value')
                    if isinstance(value, dict) and 'id' in value:
                        target_entity_id = value['id']
                        # 获取首都的名称
                        target_entity_name = data.get(target_entity_id, {}).get('labels', {}).get('en', {}).get('value',
                                                                                                                '')
                        # 创建目标实体节点，同时存储名称
                        tx.run("MERGE (t:Entity {id: $target_entity_id, name: $target_entity_name})",
                               target_entity_id=target_entity_id, target_entity_name=target_entity_name)
                        # 创建关系
                        tx.run("MATCH (e:Entity {id: $entity_id}), (t:Entity {id: $target_entity_id}) "
                               "MERGE (e)-[r:RELATED {property: $property_id}]->(t)",
                               entity_id=entity_id, target_entity_id=target_entity_id, property_id=property_id)

    with driver.session() as session:
        if data:
            for entity_id, entity in data.items():
                entity_name = entity.get('labels', {}).get('en', {}).get('value', '')
                claims = entity.get('claims', {})
                session.write_transaction(create_nodes_and_relationships, entity_id, entity_name, claims)

    driver.close()
