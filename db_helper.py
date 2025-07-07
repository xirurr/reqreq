import uuid
from typing import Dict, List

from neo4j import GraphDatabase
from traits.trait_types import self


class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def upload_to_neo4j(self, data):
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_constraints)
                session.execute_write(self._create_indices)
                session.execute_write(self._load_requirements, data['requirements'])
                session.execute_write(self._load_entities, data['entities'])
                session.execute_write(self._load_system_states, data['system_states'])
                session.execute_write(self._create_relations, data['requirements'])
                print("Загрузка в neo4j завершена")
            except Exception as e:
                print(f"Ошибка загрузки в neo4j: {str(e)}")
                raise e

    def upload_requirement(self, req):
        with self.driver.session() as session:
            try:
                session.execute_write(self._load_requirement, req)
            except Exception as e:
                print(f"Ошибка загрузки в neo4j: {str(e)}")
                raise e

    @staticmethod
    def _create_indices(tx):
        tx.run("CREATE INDEX FOR (r:Requirement) ON (r.id) IF NOT EXISTS")

    @staticmethod
    def _create_constraints(tx):
        constraints = [
            {
                'label': 'Requirement',
                'property': 'id',
                'query': "CREATE CONSTRAINT requirement_id_unique FOR (r:Requirement) REQUIRE r.id IS UNIQUE"
            },
            {
                'label': 'Entity',
                'property': 'id',
                'query': "CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            },
            {
                'label': 'SystemState',
                'property': 'id',
                'query': "CREATE CONSTRAINT system_state_id_unique FOR (s:SystemState) REQUIRE s.id IS UNIQUE"
            },
            {
                'label': 'RequirementVersion',
                'property': 'version_id',
                'query': "CREATE CONSTRAINT requirement_version_id_unique FOR (v:RequirementVersion) REQUIRE v.version_id IS UNIQUE"
            }
        ]

        for constraint in constraints:
            # Проверяем существование ограничения по метке и свойству
            check_query = """
                SHOW CONSTRAINTS 
                YIELD labelsOrTypes, properties, type
                WHERE $label IN labelsOrTypes 
                  AND $property IN properties 
                  AND type = 'UNIQUENESS'
                RETURN count(*) as count
                """
            result = tx.run(check_query, label=constraint['label'], property=constraint['property'])
            count = result.single()['count']

            if count == 0:
                try:
                    tx.run(constraint['query'])
                    print(f"Создано ограничение для {constraint['label']}.{constraint['property']}")
                except Exception as e:
                    print(f"Ошибка при создании ограничения {constraint['label']}.{constraint['property']}: {e}")
            else:
                print(f"Ограничение для {constraint['label']}.{constraint['property']} уже существует")

    def _load_requirements(self, tx, requirements):
        for req in requirements:
            self._load_requirement(tx, req)

    def _load_requirement(self, tx, req):
        # Основное требование
        query = """
            MERGE (r:Requirement {id: $id})
            ON CREATE SET 
                r.domain = $domain,
                r.type = $type,
                r.created_at = datetime()
            ON MATCH SET 
                r.domain = $domain,
                r.type = $type
            RETURN r"""
        tx.run(query, id=req['id'], domain=req['domain'], type=req['type'])

        # Проверка наличия текущей версии
        result = tx.run("""
                         MATCH (r:Requirement {id: $id})
                         RETURN COALESCE(r.current_version_id, null) as current_version""",
                        id=req['id'])
        record = result.single()
        current_version = record["current_version"] if record else None

        if not current_version:
            # Создание первой версии
            version_id = str(uuid.uuid4())
            tx.run("""
                    MATCH (r:Requirement {id: $id})
                    CREATE (v:RequirementVersion {
                        version_id: $version_id,
                        text: $text,
                        timestamp: datetime()
                    })
                    SET r.current_version_id = $version_id
                    CREATE (r)-[:HAS_CURRENT_VERSION]->(v)
                    RETURN v""",
                   id=req['id'], version_id=version_id, text=req['text'])

            if 'dependencies' in req:
                self._update_requirement_dependencies(tx, req['id'], req['dependencies'])
            if 'conditions' in req:
                self._update_requirement_conditions(tx, req['id'], req['conditions'])
        else:
            # Проверка отличий текста
            result = tx.run("""
                    MATCH (v:RequirementVersion {version_id: $version_id})
                    RETURN v.text as text""",
                            version_id=current_version)

            record = result.single()
            stored_text = record["text"] if record else ""

            if stored_text != req['text']:
                # Создание новой версии
                new_version_id = str(uuid.uuid4())
                tx.run("""
                        MATCH 
                            (r:Requirement {id: $id}),
                            (current:RequirementVersion {version_id: $current_version})
                        CREATE (new:RequirementVersion {
                            version_id: $new_version_id,
                            text: $text,
                            timestamp: datetime()
                        })
                        CREATE (current)-[:REPLACED_BY]->(new)
                        CREATE (r)-[:HAS_CURRENT_VERSION]->(new)
                        SET r.current_version_id = $new_version_id
                        WITH r, current
                        MATCH (r)-[old:HAS_CURRENT_VERSION]->(current)
                        DELETE old""",
                       id=req['id'],
                       current_version=current_version,
                       new_version_id=new_version_id,
                       text=req['text'])

        # Обновляем зависимости и условия с проверкой валидности
        self._update_requirement_dependencies(tx, req['id'], req.get('dependencies', []))
        self._update_requirement_conditions(tx, req['id'], req.get('conditions', []))

    @staticmethod
    def _load_entities(tx, entities):
        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {id: entity.id})
        ON CREATE SET 
            e.name = entity.name,
            e.description = entity.description,
            e.created_at = datetime()
        ON MATCH SET 
            e.name = entity.name,
            e.description = entity.description
        RETURN count(*)"""

        result = tx.run(query, entities=entities)

    @staticmethod
    def _load_system_states(tx, states):
        query = """
        UNWIND $states AS state
        MERGE (s:SystemState {id: state.id})
        ON CREATE SET 
            s.name = state.name,
            s.description = state.description,
            s.entity = state.entity,
            s.created_at = datetime()
        ON MATCH SET 
            s.name = state.name,
            s.description = state.description,
            s.entity = state.entity
        RETURN count(*)"""

        result = tx.run(query, states=states)

    @staticmethod
    def _create_relations(tx, requirements):
        for req in requirements:
            # Обработка зависимостей
            if 'dependencies' in req:
                tx.run("""
                    MATCH (r:Requirement {id: $id})
                    OPTIONAL MATCH (r)-[d:DEPENDS_ON]->()
                    DELETE d
                    WITH r
                    UNWIND $deps AS dep_id
                    MATCH (d:Requirement {id: dep_id})
                    MERGE (r)-[:DEPENDS_ON]->(d)
                """, id=req['id'], deps=req['dependencies'])

            # Обработка условий
            if 'conditions' in req:
                for condition in req['conditions']:
                    if condition.get('source_req') is not None and condition.get('source_req') != '':
                        tx.run("""
                            MATCH (r:Requirement {id: $id})
                            MATCH (s:SystemState {id: $source_req})
                            MERGE (r)-[rel:CONDITION_ON]->(s)
                            SET rel.condition = $condition
                        """,
                               id=req['id'],
                               source_req=condition['source_req'],
                               condition=condition['condition'])

    def create_semantic_link(self, source_id: str, target_id: str, relation_type: str):
        """
        Создание семантической связи с типом

        Args:
            source_id: ID исходного узла
            target_id: ID целевого узла
            relation_type: Тип семантической связи

        Returns:
            bool: True если связь создана успешно, False в противном случае
        """

        def _create_link(tx, tx_source_id: str, tx_target_id: str, tx_relation_type: str):
            query = """
            MATCH (a {id: $source_id})
            MATCH (b {id: $target_id})
            MERGE (a)-[r:SEMANTIC_LINK {type: $relation_type}]->(b)
            RETURN a.id as source, b.id as target, r.type as relation
            """
            result = tx.run(query,
                            source_id=tx_source_id,
                            target_id=tx_target_id,
                            relation_type=tx_relation_type)
            return result.single()

        try:
            with self.driver.session() as session:
                record = session.execute_write(_create_link, source_id, target_id, relation_type)

                if record:
                    print(
                        f"Создана семантическая связь: {record['source']} -[{record['relation']}]-> {record['target']}")
                    return True
                else:
                    print(f"Не удалось найти узлы с ID: {source_id} или {target_id}")
                    return False

        except Exception as e:
            print(f"Ошибка при создании семантической связи между {source_id} и {target_id}: {str(e)}")
            return False

    def update_requirement_metadata(self, req: Dict):
        with self.driver.session() as session:
            """Обновляет метаданные и зависимости/условия существующего требования"""
            # 1. Обновление базовых полей (domain, type)
            session.run("""
                MATCH (r:Requirement {id: $id})
                SET r.domain = $domain, r.type = $type_req
            """, id=req["id"], domain=req.get("domain"), type_req=req.get("type"))

            # 2. Обновление зависимостей
            self._update_requirement_dependencies(session, req["id"], req.get("dependencies", []))

            # 3. Обновление условий
            self._update_requirement_conditions(session, req["id"], req.get("conditions", []))

    def _update_requirement_dependencies(self, tx, req_id: str, dependencies: List[str]):
            """Удаляет старые зависимости и добавляет новые"""
            # Удаление старых зависимостей
            tx.run("""
                MATCH (r:Requirement {id: $id})-[d:DEPENDS_ON]->()
                DELETE d
            """, id=req_id)

            # Добавление новых зависимостей
            if dependencies:
                tx.run("""
                    MATCH (r:Requirement {id: $id})
                    UNWIND $deps AS dep_id
                    MERGE (d:Requirement {id: dep_id})
                    MERGE (r)-[:DEPENDS_ON]->(d)
                """, id=req_id, deps=dependencies)

    def _update_requirement_conditions(self, tx, req_id: str, conditions: List[Dict]):
            """Удаляет старые условия и добавляет новые"""
            tx.run("""
                MATCH (r:Requirement {id: $id})-[c:CONDITION_ON]->()
                DELETE c
            """, id=req_id)

            # Добавление новых условий
            if conditions:
                valid_conditions = [
                    cond for cond in conditions
                    if cond.get('source_req') is not None and cond.get('source_req') != ''
                ]

                tx.run("""
                    MATCH (r:Requirement {id: $id})
                    UNWIND $conditions AS cond
                    MERGE (s:SystemState {id: cond.source_req})
                    MERGE (r)-[:CONDITION_ON {condition: cond.condition}]->(s)
                """, id=req_id, conditions=valid_conditions)

    def get_next_id(self, domain: str, req_type: str) -> str:
        with self.driver.session() as session:
            result = session.run("""
            MATCH (r:Requirement)
            WHERE r.id STARTS WITH $prefix
            RETURN MAX(
                CASE 
                    WHEN apoc.text.regexGroups(r.id, '\\\\d+$') IS NOT NULL 
                    THEN TOINTEGER(HEAD(HEAD(apoc.text.regexGroups(r.id, '\\\\d+$')))) 
                    ELSE 0 
                END
            ) AS max_number
            """, prefix=f"{domain}-{req_type}-")
            record = result.single()

            next_number = (record["max_number"] or 0) + 1
            return f"{domain}-{req_type}-{next_number}"