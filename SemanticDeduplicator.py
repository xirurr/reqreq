from typing import List, Dict, Optional

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from LLMClientManager import LLMClientManager
from QdrantVectorLoader import QdrantVectorLoader
from db_helper import Neo4jLoader
from services.CacheService import CacheService


class SemanticDeduplicator:
    def __init__(self, config: dict, qdrant_client: QdrantVectorLoader, neo4j_loader: Neo4jLoader, cache: CacheService):
        self.qdrant_client = qdrant_client
        self.neo4j_loader = neo4j_loader
        self.llm_manager = LLMClientManager(config)
        self.similarity_threshold = 0.95  # Порог для семантического совпадения
        self.cache = cache

    def _generate_vector(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Текст не может быть пустым")
        pure_vector = self.llm_manager.call_embedding_llm(text)
        normalized_vector = normalize([pure_vector])[0].tolist()
        return normalized_vector

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        vector1 = self._generate_vector(text1)
        vector2 = self._generate_vector(text2)
        return cosine_similarity([vector1], [vector2])[0][0]

    def _is_semantically_similar(self, text1: str, text2: str) -> bool:
        """Проверяет, являются ли тексты семантически эквивалентными"""
        similarity = self._calculate_similarity(text1, text2)
        return similarity >= self.similarity_threshold

    def _find_similar_requirements(self, text: str) -> List[Dict]:
        try:
            if not text or not text.strip():
                return []

            model_name = self.llm_manager.get_embedding_model_name()
            hash_key = "req:"+self.cache.generate_hash(text, model_name)
            if self.cache.exists(hash_key):
                vector = self.cache.get(hash_key)
            else:
                vector = self._generate_vector(text)
                self.cache.set(hash_key, vector)

            results = self.qdrant_client.search(
                query_vector=vector,
                score_threshold=self.similarity_threshold,
                limit=1,  # Ищем только точное совпадение
                filter_payload={"must": [{"key": "type", "value": "requirement"}]}
            )

            return [point.payload for point in results.points if point.payload["type"] == "requirement"]

        except Exception as e:
            print(f"Ошибка поиска сходств в qdrant: {str(e)}")
            raise

    def _merge_metadata(self, new_req: Dict, existing_req: Dict) -> Dict:
        """
        Объединяет только метаданные (зависимости, условия), 
        сохраняя оригинальный текст требования
        """
        merged = existing_req.copy()

        # Объединяем зависимости
        merged["dependencies"] = list(set(
            merged.get("dependencies", []) + new_req.get("dependencies", [])
        ))

        # Объединяем условия
        merged_conditions = {cond["condition"]: cond for cond in merged.get("conditions", [])}
        new_conditions = {cond["condition"]: cond for cond in new_req.get("conditions", [])}
        merged["conditions"] = list({**merged_conditions, **new_conditions}.values())
        merged["id"] = existing_req["id"]
        return merged

    def process_requirements(self, requirements: List[Dict]) -> List[Dict]:
        """Обрабатывает требования, не обновляя текст при семантическом совпадении"""
        try:
            updated_requirements = []

            for req in requirements:
                similar = self._find_similar_requirements(req["text"])

                if similar:
                    existing = similar[0]
                    # Если семантически совпадает — обновляем только метаданные
                    merged = self._merge_metadata(req, existing)
                    updated_requirements.append(merged)
                else:
                    # Новое требование
                    updated_requirements.append(req)

            return updated_requirements
        except Exception as e:
            print(f"Ошибка обработки требований {str(e)}")
            raise

    def process_single_requirement(self, req: Dict) -> Optional[Dict]:
        """Обрабатывает требования, не обновляя текст при семантическом совпадении"""
        try:
            if not req.get("text"):
                print("Пропуск требования без текста")
                return None

            if not req.get("domain"):
                req["domain"] = "general"

            if not req.get("type"):
                req["type"] = "functional"

            similar = self._find_similar_requirements(req["text"])

            if similar:
                print("Найдено похожее требование, обновляем метаданные...")
                existing = similar[0]
                merged = self._merge_metadata(req, existing)
                self.neo4j_loader.update_requirement_metadata(merged)
                return merged
            else:
                print("Новое требование, создаем...")
                if not req.get("id"):
                    req["id"] = self.neo4j_loader.get_next_id(req["domain"], req["type"])

                req["id"] = self.neo4j_loader.get_next_id(req["domain"], req["type"])
                self.neo4j_loader.upload_requirement(req)
                requirement = self.qdrant_client.build_requirement(req)
                self.qdrant_client.upload_embeddings(requirement)
                return req
        except Exception as e:
            print(f"Ошибка обработки требования: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

