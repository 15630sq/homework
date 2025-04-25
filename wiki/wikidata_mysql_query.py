import pymysql


def query_mysql(entity_id, property_id):
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="chenhongzhou",
            database="wikidata_db",
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            # 修改 SQL 查询语句，查询首都对应的实体名称
            sql = '''
            SELECT e2.entity_name
            FROM entities e1
            JOIN relationships r ON e1.entity_id = r.entity_id
            JOIN properties p ON r.property_id = p.property_id
            JOIN entities e2 ON r.target_entity_id = e2.entity_id
            WHERE e1.entity_id = %s AND p.property_id = %s
            '''
            cursor.execute(sql, (entity_id, property_id))
            result = cursor.fetchone()
            if result:
                return result['entity_name']
    except pymysql.Error as e:
        print(f"MySQL 查询出错: {e}")
    finally:
        if connection:
            connection.close()
    return None
