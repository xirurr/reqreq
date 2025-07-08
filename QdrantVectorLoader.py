import uuid

from qdrant_client import QdrantClient, models
from LLMClientManager import LLMClientManager


class QdrantVectorLoader:
    def __init__(self, llm_manager: LLMClientManager, collection_name: str):
        self.client = QdrantClient(host="localhost", port=6333)
        self.llm_manager = llm_manager
        self.collection_name = collection_name
        embedding_size = self.llm_manager.get_embedding_dim()
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE)
        )

    def load_requirements(self, requirements: list, release_version: str):
        points = []
        for req in requirements:
            if "id" in req and "text" in req:
                vector = self.llm_manager.call_embedding_llm(req["text"])
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
        query_vector = self.llm_manager.call_embedding_llm(text)
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
