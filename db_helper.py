import json
import re
import uuid
from neo4j import GraphDatabase

class Neo4jWriter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()

    def close(self):
        self.driver.close()

    def _create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Requirement) REQUIRE r.req_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:RequirementVersion) REQUIRE v.version_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:EntityVersion) REQUIRE v.version_id IS UNIQUE")

    def get_existing_requirements(self):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:Requirement)-[:CURRENT_VERSION]->(v:RequirementVersion)
                RETURN r.req_id AS id, v.text AS text
            """)
            return [{"id": record["id"], "text": record["text"]} for record in result]

    def get_next_req_id_index(self, category: str) -> int:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:Requirement)
                WHERE r.req_id STARTS WITH $prefix
                RETURN r.req_id AS req_id
            """, prefix=f"REQ-{category}-")
            
            max_index = 0
            for record in result:
                try:
                    index_str = record["req_id"].split('-')[-1]
                    index = int(index_str)
                    if index > max_index:
                        max_index = index
                except (ValueError, IndexError):
                    continue
            return max_index + 1

    def write_results(self, data: dict, release_version: str):
        with self.driver.session() as session:
            print("Запись сущностей в Neo4j...")
            session.execute_write(self._create_or_update_entities, data.get("entities", []), release_version)
            
            print("Запись требований в Neo4j...")
            session.execute_write(self._create_or_update_requirements, data.get("requirements", []), release_version)
            
            print("Создание связей между требованиями и сущностями...")
            session.execute_write(self._link_reqs_to_entities, data.get("requirements", []), data.get("entities", []))

            print("Запись зависимостей в Neo4j...")
            session.execute_write(self._create_dependencies, data.get("dependencies", []))
            
            print("Загрузка в Neo4j завершена.")

    @staticmethod
    def _create_or_update_entities(tx, entities, release_version):
        for entity_data in entities:
            props = entity_data.copy()
            name = props.pop('name')
            version_id = str(uuid.uuid4())
            props["version_id"] = version_id

            if 'attributes' in props: props['attributes'] = json.dumps(props['attributes'], ensure_ascii=False)
            if 'states' in props: props['states'] = json.dumps(props['states'], ensure_ascii=False)

            res = tx.run("""
                MERGE (e:Entity {name: $name})
                WITH e
                OPTIONAL MATCH (e)-[r:CURRENT_VERSION]->(previous_version:EntityVersion)
                CREATE (new_version:EntityVersion $props)
                CREATE (e)-[:HAS_VERSION]->(new_version)
                RETURN previous_version, new_version
            """, name=name, props=props)
            
            record = res.single()
            if record and record["previous_version"]:
                tx.run("""
                    MATCH (prev:EntityVersion {version_id: $prev_id})
                    MATCH (new:EntityVersion {version_id: $new_id})
                    CREATE (new)-[:PREVIOUS_VERSION]->(prev)
                """, prev_id=record["previous_version"]["version_id"], new_id=version_id)
            
            tx.run("MATCH (e:Entity {name: $name})-[r:CURRENT_VERSION]->() DELETE r", name=name)
            tx.run("MATCH (e:Entity {name: $name}), (v:EntityVersion {version_id: $vid}) CREATE (e)-[:CURRENT_VERSION]->(v)", name=name, vid=version_id)

    @staticmethod
    def _create_or_update_requirements(tx, requirements, release_version):
        for req_data in requirements:
            props = req_data.copy()
            req_id = props.pop('id')
            version_id = str(uuid.uuid4())
            props["version_id"] = version_id

            if 'conditions' in props: props['conditions'] = json.dumps(props['conditions'], ensure_ascii=False)

            res = tx.run("""
                MERGE (r:Requirement {req_id: $req_id})
                WITH r
                OPTIONAL MATCH (r)-[rel:CURRENT_VERSION]->(previous_version:RequirementVersion)
                CREATE (new_version:RequirementVersion $props)
                SET new_version.timestamp = datetime(), new_version.release_version = $release
                CREATE (r)-[:HAS_VERSION]->(new_version)
                RETURN previous_version, new_version
            """, req_id=req_id, props=props, release=release_version)

            record = res.single()
            if record and record["previous_version"]:
                 tx.run("""
                    MATCH (prev:RequirementVersion {version_id: $prev_id})
                    MATCH (new:RequirementVersion {version_id: $new_id})
                    CREATE (new)-[:PREVIOUS_VERSION]->(prev)
                """, prev_id=record["previous_version"]["version_id"], new_id=version_id)

            tx.run("MATCH (r:Requirement {req_id: $req_id})-[rel:CURRENT_VERSION]->() DELETE rel", req_id=req_id)
            tx.run("MATCH (r:Requirement {req_id: $req_id}), (v:RequirementVersion {version_id: $vid}) CREATE (r)-[:CURRENT_VERSION]->(v)", req_id=req_id, vid=version_id)

    @staticmethod
    def _create_dependencies(tx, dependencies):
        for dep in dependencies:
            # Валидация типа связи, чтобы предотвратить Cypher Injection.
            # Разрешены только заглавные буквы и подчеркивания.
            rel_type = dep.get('type', '').strip().upper()
            if not re.match(r'^[A-Z_]+$', rel_type):
                print(f"--- WARNING: Invalid relationship type skipped: {rel_type} ---")
                continue

            # Используем f-string для безопасной вставки типа связи в запрос.
            # Это необходимо, так как типы связей не могут быть параметризованы.
            query = f"""
                MATCH (s_anchor:Requirement {{req_id: $source_id}})-[:CURRENT_VERSION]->(s_ver:RequirementVersion)
                MATCH (t_anchor:Requirement {{req_id: $target_id}})-[:CURRENT_VERSION]->(t_ver:RequirementVersion)
                MERGE (s_ver)-[:{rel_type}]->(t_ver)
            """
            tx.run(query, source_id=dep.get('source'), target_id=dep.get('target'))

    @staticmethod
    def _link_reqs_to_entities(tx, requirements, entities):
        entity_names = {e['name'].lower() for e in entities}
        for req in requirements:
            req_text_lower = req.get('text', '').lower()
            req_id = req.get('id')
            if not req_id:
                continue
            for entity_name in entity_names:
                if re.search(r'\b' + re.escape(entity_name) + r'\b', req_text_lower):
                    tx.run("""
                        MATCH (r:Requirement {req_id: $req_id})-[:CURRENT_VERSION]->(v:RequirementVersion)
                        MATCH (e:Entity {name: $entity_name})
                        MERGE (v)-[:CONTAINS_ENTITY]->(e)
                    """, req_id=req_id, entity_name=entity_name.capitalize())
