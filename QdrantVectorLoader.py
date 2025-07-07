import uuid
from typing import Dict, List, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, VectorParams, Distance, FieldCondition, MatchValue

from LLMClientManager import LLMClientManager
from services.CacheService import CacheService


class QdrantVectorLoader:
    def __init__(
            self,
            config: dict,  # Передаем ваш LLM-менеджер
            cache: CacheService,
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            collection_name: str = "system_artifacts"
    ):
        self.llm_manager = LLMClientManager(config)
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self._initialize_collection()
        self.cache = cache

    def _initialize_collection(self):
        collections = self.client.get_collections().collections
        example_vector = self.llm_manager.call_embedding_llm("Тест")
        expected_dim = len(example_vector)

        if not any(col.name == self.collection_name for col in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=expected_dim,  # Размер вектора зависит от модели (например, text-embedding-ada-002)
                    distance=Distance.COSINE
                )
            )

    def build_embeddings(self, data: Dict) -> list[Any]:
        """Загружает векторы в Qdrant через LLM API"""

        filtered_data = {
            "requirements": [req for req in data.get("requirements", [])
                             if req.get("text", "").strip()],
            "entities": [entity for entity in data.get("entities", [])
                         if entity.get("description", "").strip()],
            "system_states": [state for state in data.get("system_states", [])
                              if state.get("description", "").strip()]
        }

        points: list[Any] = []
        # Обработка требований
        for req in filtered_data.get("requirements", []):
            points.append(self.build_requirement(req))

        # Обработка сущностей
        for entity in filtered_data.get("entities", []):
            points.append(self.build_entity(entity))

        # Обработка системных состояний
        for state in filtered_data.get("system_states", []):
            points.append(self.build_system_state(state))

        return points

    def upload_embeddings(self, points):
        if not isinstance(points, list):
            points = [points]

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            print(f"Ошибка загрузки в qdrant: {str(e)}")
            raise

    def build_system_state(self, state):
        model_name = self.llm_manager.get_embedding_model_name()
        hash_key = "req.state:"+self.cache.generate_hash(state["description"], model_name)
        if self.cache.exists(hash_key):
            vector = self.cache.get(hash_key)
        else:
            vector = self.llm_manager.call_embedding_llm(state["description"])
            self.cache.set(hash_key, vector)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_X500, state["id"]))  # конвертируем в UUID
        return {
            "id": point_id,
            "vector": vector,
            "payload": {
                "type": "system_state",
                "id": state["id"],
                "name": state["name"]
            }
        }

    def build_entity(self, entity):
        model_name = self.llm_manager.get_embedding_model_name()
        hash_key = "req.entity:"+self.cache.generate_hash(entity["description"], model_name)
        if self.cache.exists(hash_key):
            vector = self.cache.get(hash_key)
        else:
            vector = self.llm_manager.call_embedding_llm(entity["description"])
            self.cache.set(hash_key, vector)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_X500, entity["id"]))  # конвертируем в UUID
        return {
            "id": point_id,
            "vector": vector,
            "payload": {
                "type": "entity",
                "id": entity["id"],
                "name": entity["name"]
            }
        }

    def build_requirement(self, req):
        # Проверяем обязательные поля
        if not req.get("text"):
            raise ValueError("Требование должно содержать текст")

        if not req.get("id"):
            raise ValueError("Требование должно содержать ID")

        # Проверяем, что ID является строкой
        req_id = str(req["id"])
        if not req_id.strip():
            raise ValueError("ID требования не может быть пустым")

        model_name = self.llm_manager.get_embedding_model_name()
        hash_key = "req:"+self.cache.generate_hash(req["text"], model_name)
        if self.cache.exists(hash_key):
            vector = self.cache.get(hash_key)
        else:
            vector = self.llm_manager.call_embedding_llm(req["text"])
            self.cache.set(hash_key, vector)
        try:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_X500, req_id))
        except Exception as e:
            print("Не удалось получить UUID")
            raise Exception(e)

        return {
            "id": point_id,
            "vector": vector,
            "payload": {
                "type": "requirement",
                "id": req["id"],
                "domain": req.get("domain"),
                "type_req": req.get("type"),
                "text": req["text"]
            }
        }

    def _build_filter_from_payload(self, filter_payload: Dict) -> Filter:
        """
        Создает корректный Filter объект из переданного словаря
        """
        if "must" in filter_payload:
            conditions = []
            for condition in filter_payload["must"]:
                if "key" in condition and "value" in condition:
                    field_condition = FieldCondition(
                        key=condition["key"],
                        match=MatchValue(value=condition["value"])
                    )
                    conditions.append(field_condition)
            return Filter(must=conditions)
        return Filter()

    def search(
            self,
            query_vector: List[float],
            limit: int = 5,
            score_threshold: float = 0.7,
            filter_payload: Optional[Dict] = None
    ):
        """
        Выполняет семантический поиск в Qdrant

        Args:
            query_vector: Вектор для поиска
            limit: Количество результатов
            score_threshold: Порог схожести
            filter_payload: Фильтр по типу объекта (например, {"type": "requirement"})

        Returns:
            Список найденных точек с payload и score
        """
        try:
            query_filter = None
            if filter_payload:
                query_filter = self._build_filter_from_payload(filter_payload)

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )

            # Преобразуем результаты в удобный формат
            return results

        except Exception as e:
            print(f"Ошибка поиска в Qdrant: {str(e)}")
            raise
