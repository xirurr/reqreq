import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance

from LLMClientManager import LLMClientManager
from services import CacheService


class QdrantVectorLoader:
    def __init__(
            self,
            config: dict,  # Передаем ваш LLM-менеджер
            cache: CacheService,
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            collection_name: str = "system_artifacts",
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

    def load_requirements(self, requirements: list):
        """Векторизует и загружает требования в Qdrant."""
        model_name = self.llm_manager.get_embedding_model_name()
        points = []
        for req in requirements:
            if "id" in req and "text" in req:
                hash_key = "req.full:" + self.cache.generate_hash(req["text"], model_name)
                if self.cache.exists(hash_key):
                    vector = self.cache.get(hash_key)
                else:
                    vector = self.llm_manager.call_embedding_llm(req["text"])
                    self.cache.set(hash_key, vector)

                point_id = str(uuid.uuid5(uuid.NAMESPACE_X500, req["id"]))  # конвертируем в UUID
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": req["text"], "id": req["id"]}
                ))

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

    def search_requirements(self, text: str, limit: int = 5) -> list:
        """Ищет семантически близкие требования."""
        query_vector = self.llm_manager.call_embedding_llm(text)
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return [{"qdrant_id": hit.id, "id": hit.payload["id"], "text": hit.payload["text"], "score": hit.score} for hit
                in hits]
