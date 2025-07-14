import uuid

from qdrant_client import QdrantClient, models
from LLMClientManager import LLMClientManager
from services.CacheService import CacheService


class QdrantVectorLoader:
    def __init__(self, llm_manager: LLMClientManager, collection_name: str, cache: CacheService):
        self.client = QdrantClient(host="localhost", port=6333)
        self.llm_manager = llm_manager
        self.collection_name = collection_name
        self.cache = cache
        embedding_size = self.llm_manager.get_embedding_dim()
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE)
        )

    def load_requirements(self, requirements: list, release_version: str):
        points = []
        model_name = self.llm_manager.get_embedding_model_name()

        for req in requirements:
            if "id" in req and "text" in req:
                req_text = req["text"]
                hash_key = f"embedding:{self.cache.generate_hash(req_text, model_name)}"

                if self.cache.exists(hash_key):
                    vector = self.cache.get(hash_key)
                else:
                    vector = self.llm_manager.call_embedding_llm(req_text)
                    self.cache.set(hash_key, vector)

                payload = req.copy()
                payload["release_version"] = release_version
                payload["id"] = req["id"]
                point_id = str(uuid.uuid5(uuid.NAMESPACE_X500, req["id"]))  # конвертируем в UUID
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

    def search_requirements(self, text: str, release_version: str, limit: int = 5) -> list:
        """Ищет семантически близкие требования в рамках релиза."""
        model_name = self.llm_manager.get_embedding_model_name()
        hash_key = f"embedding:{self.cache.generate_hash(text, model_name)}"

        if self.cache.exists(hash_key):
            query_vector = self.cache.get(hash_key)
        else:
            query_vector = self.llm_manager.call_embedding_llm(text)
            self.cache.set(hash_key, query_vector)

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="release_version",
                        match=models.MatchValue(value=release_version)
                    )
                ]
            ),
            limit=limit
        )
        return [{"qdrant_id": hit.id, "id": hit.payload["id"], "text": hit.payload["text"], "score": hit.score} for hit
                in hits]

    def cleanup(self):
        self.client.delete_collection(collection_name=self.collection_name)
