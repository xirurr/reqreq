import uuid
import hashlib
import os
from qdrant_client import QdrantClient, models
from LLMClientManager import LLMClientManager
from models.model_types import ModelType
from services.CacheService import CacheService

class QdrantVectorLoader:
    def __init__(self, llm_manager: LLMClientManager, cache: CacheService):
        self.client = QdrantClient(host="localhost", port=6333)
        self.llm_manager = llm_manager
        self.cache = cache
        self.embedding_size = self.llm_manager.get_embedding_dim()

    def get_collection_names(self, release_version: str) -> (str, str):
        """Генерирует постоянные имена коллекций для узлов и чанков на основе версии релиза."""
        safe_version_name = release_version.replace('.', '_').replace('-', '_')
        nodes_collection = f"nodes_{safe_version_name}"
        chunks_collection = f"chunks_{safe_version_name}"
        return nodes_collection, chunks_collection

    def create_collection_if_not_exist(self, collection_name: str):
        """Создает коллекцию, если она еще не существует."""
        try:
            self.client.get_collection(collection_name=collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=self.embedding_size, distance=models.Distance.COSINE)
            )

    def load_graph_nodes(self, nodes: list, collection_name: str):
        """Загружает узлы графа (требования, сущности) в указанную коллекцию."""
        self.create_collection_if_not_exist(collection_name)
        
        points = []
        model_name = self.llm_manager.get_model_name(model_type=ModelType.EMBEDDING)

        for node in nodes:
            node_text_content = node.get('text') or node.get('name', '')
            node_type = 'Requirement' if 'text' in node else 'Entity'
            node_text_for_embedding = f"{node_type}: {node_text_content}"

            hash_key = f"embedding:{self.cache.generate_hash(node_text_for_embedding, model_name)}"

            if self.cache.exists(hash_key):
                vector = self.cache.get(hash_key)
            else:
                vector = self.llm_manager.call_embedding_llm(node_text_for_embedding)
                self.cache.set(hash_key, vector)

            payload = node.copy()
            node_id = node.get("id") or node.get("name")
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection_name}-{node_id}"))

            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        if points:
            self.client.upsert(collection_name=collection_name, points=points, wait=True)

    def _generate_chunk_id(self, source_file: str, chunk_text: str) -> str:
        """Генерирует детерминированный ID для чанка на основе его содержимого."""
        chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
        return f"{os.path.basename(source_file)}_{chunk_hash[:16]}"

    def load_source_chunks(self, chunks: list, collection_name: str, source_file: str):
        """Загружает исходные текстовые чанки в указанную коллекцию."""
        self.create_collection_if_not_exist(collection_name)

        points = []
        model_name = self.llm_manager.get_model_name(model_type=ModelType.EMBEDDING)

        for chunk_text in chunks:
            hash_key = f"embedding:{self.cache.generate_hash(chunk_text, model_name)}"
            
            if self.cache.exists(hash_key):
                vector = self.cache.get(hash_key)
            else:
                vector = self.llm_manager.call_embedding_llm(chunk_text)
                self.cache.set(hash_key, vector)
            
            chunk_id = self._generate_chunk_id(source_file, chunk_text)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection_name}-{chunk_id}"))
            
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload={"text": chunk_text, "chunk_id": chunk_id, "source_file": source_file}
            ))

        if points:
            self.client.upsert(collection_name=collection_name, points=points, wait=True)

    def search_graph_nodes(self, text: str, collection_name: str, limit: int = 5) -> list:
        """Ищет семантически близкие узлы графа в указанной коллекции."""
        try:
            self.client.get_collection(collection_name=collection_name)
        except Exception:
            return [] 

        model_name = self.llm_manager.get_model_name(ModelType.EMBEDDING)
        hash_key = f"embedding:{self.cache.generate_hash(text, model_name)}"

        if self.cache.exists(hash_key):
            query_vector = self.cache.get(hash_key)
        else:
            query_vector = self.llm_manager.call_embedding_llm(text)
            self.cache.set(hash_key, query_vector)

        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return [{**hit.payload, 'score': hit.score} for hit in hits]

    def delete_collection(self, collection_name: str):
        """Удаляет любую коллекцию по имени."""
        try:
            self.client.delete_collection(collection_name=collection_name)
        except Exception:
            pass

    def delete_collections_for_version(self, release_version: str):
        """Удаляет постоянные коллекции, связанные с конкретной версией релиза (для администрирования)."""
        nodes_collection, chunks_collection = self.get_collection_names(release_version)
        self.delete_collection(nodes_collection)
        self.delete_collection(chunks_collection)

    def list_collections(self) -> list[str]:
        """Возвращает список имен всех коллекций."""
        response = self.client.get_collections()
        return [collection.name for collection in response.collections]

    def get_chunks_by_ids(self, chunk_ids: list, collection_name: str) -> list[str]:
        """Получает тексты чанков по их ID из указанной коллекции."""
        if not chunk_ids:
            return []
        
        # Qdrant требует, чтобы ID были в формате UUID, если они так хранились
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection_name}-{chunk_id}")) for chunk_id in chunk_ids]

        try:
            records = self.client.retrieve(
                collection_name=collection_name,
                ids=point_ids,
                with_payload=True,
                with_vectors=False
            )
            return [record.payload['text'] for record in records]
        except Exception as e:
            print(f"Ошибка при получении чанков из Qdrant: {e}")
            return []