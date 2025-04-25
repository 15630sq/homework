from neo4j import GraphDatabase

from neo4j import GraphDatabase


def query_neo4j(entity_id, property_id):
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "chenhongzhou"))

    def get_related_entity(tx, entity_id, property_id):
        result = tx.run("MATCH (e:Entity {id: $entity_id})-[r:RELATED {property: $property_id}]->(t:Entity) "
                        "RETURN t.name",
                        entity_id=entity_id, property_id=property_id)
        return [record["t.name"] for record in result]

    with driver.session() as session:
        related_entities = session.read_transaction(get_related_entity, entity_id, property_id)
    driver.close()
    return related_entities

