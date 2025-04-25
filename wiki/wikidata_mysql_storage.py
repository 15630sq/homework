import pymysql


def insert_entity(cursor, entity_id, entity_name):
    """插入实体（带唯一约束）"""
    sql = "INSERT IGNORE INTO entities (entity_id, entity_name) VALUES (%s, %s)"
    cursor.execute(sql, (entity_id, entity_name))


def insert_property(cursor, property_id, property_name):
    """插入属性（带唯一约束）"""
    sql = "INSERT IGNORE INTO properties (property_id, property_name) VALUES (%s, %s)"
    cursor.execute(sql, (property_id, property_name))


def insert_relationship(cursor, entity_id, property_id, value):
    target_entity_id = None
    if isinstance(value, dict) and 'id' in value:
        target_entity_id = value['id']  # 提取实体ID（如Q3421）
        value = None  # 实体关系中value字段留空，专注外键关联
    elif isinstance(value, (str, int, float)):
        value = str(value)  # 非实体类型正常存储

    sql = """
        INSERT INTO relationships 
        (entity_id, property_id, target_entity_id, value) 
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(sql, (entity_id, property_id, target_entity_id, value))


def store_to_mysql(data):
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="chenhongzhou",
            database="wikidata_db",
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            if data:
                for entity_id, entity in data.items():
                    entity_name = entity.get('labels', {}).get('en', {}).get('value', '')
                    insert_entity(cursor, entity_id, entity_name)

                    claims = entity.get('claims', {})
                    # 只处理首都属性（P36）
                    if 'P36' in claims:
                        property_id = 'P36'
                        property_name = property_id
                        insert_property(cursor, property_id, property_name)

                        for claim in claims[property_id]:
                            mainsnak = claim.get('mainsnak', {})
                            if mainsnak.get('snaktype') != 'value':
                                continue  # 只处理值类型声明

                            datavalue = mainsnak.get('datavalue', {})
                            value = datavalue.get('value', {})

                            # 处理首都等实体引用（value是含'id'的字典）
                            if isinstance(value, dict) and 'id' in value:
                                # 例如：{"id": "Q3421"} → 提取id作为target_entity_id
                                insert_relationship(cursor, entity_id, property_id, value)
                            else:
                                insert_relationship(cursor, entity_id, property_id, value)
        connection.commit()
    except pymysql.Error as e:
        print(f"MySQL 存储出错: {e}")
    finally:
        if connection:
            connection.close()
