import os
import re
import json
from packaging.version import parse as parse_version, InvalidVersion
from LLMClientManager import LLMClientManager
from QdrantVectorLoader import QdrantVectorLoader
from db_helper import Neo4jWriter
from services.CacheService import CacheService
from prompts.promts import QNA_PROMPT
from models.model_types import ModelType
from processors.result_processor import ResultProcessor

class QuestionAnsweringService:
    def __init__(self, config: dict):
        self.config = config
        self.llm_manager = LLMClientManager(self.config)
        self.qdrant_loader = QdrantVectorLoader(self.llm_manager, cache=CacheService())
        self.neo4j_driver = Neo4jWriter(uri=config.get("neo4j_uri"), user=config.get("neo4j_user"),
                                       password=config.get("neo4j_password"))
        self.result_processor = ResultProcessor(self.llm_manager)
        self.CONFIDENCE_THRESHOLD = 0.6 # Порог уверенности для переключения на резервный поиск

    def answer_question(self, question: str, release_version: str = "latest") -> dict:
        """Основной метод, оркестрирующий весь RAG-конвейер."""
        
        target_version = self._get_target_version(release_version)
        if not target_version:
            return {"answer": "Не найдено ни одной проанализированной версии релиза.", "sources": []}

        # --- Шаг 1: Основной, "умный" поиск ---
        entry_nodes = self._find_entry_points(question, target_version)
        if not entry_nodes:
            print("Точки входа не найдены. Переключение на резервный поиск...")
            return self._fallback_search(question, target_version)

        context_graph = self._get_context_graph(entry_nodes, target_version)
        if not context_graph.get("nodes"):
            print("Контекстный граф пуст. Переключение на резервный поиск...")
            return self._fallback_search(question, target_version)

        enriched_text = self._enrich_context_with_source_text(context_graph, target_version)
        structured_answer = self._synthesize_final_answer(question, context_graph, enriched_text)

        # --- Шаг 2: Проверка на неуверенность ---
        confidence = structured_answer.get("confidence_score", 0.0)
        if confidence < self.CONFIDENCE_THRESHOLD:
            print(f"Низкая уверенность ({confidence}). Переключение на резервный поиск...")
            return self._fallback_search(question, target_version)
        
        print("Ответ с высокой уверенностью получен основным методом.")
        return {
            "answer": structured_answer.get("answer", "Не удалось извлечь ответ."),
            "reasoning": structured_answer.get("reasoning", ""),
            "sources": entry_nodes
        }

    def _fallback_search(self, question: str, release_version: str) -> dict:
        """Резервный поиск по всем исходным чанкам."""
        # TODO: Реализовать "классический" RAG по коллекции chunks_*
        print("Вызван резервный поиск (пока не реализован).")
        return {
            "answer": "Основной метод не дал уверенного ответа, а резервный поиск еще не реализован.",
            "reasoning": "fallback_not_implemented",
            "sources": []
        }

    def _get_target_version(self, release_version: str) -> str | None:
        if release_version != "latest":
            return release_version
        # ... (остальные методы без изменений) ...
        print("Определение последней версии релиза...")
        all_collections = self.qdrant_loader.list_collections()
        versions = []
        for name in all_collections:
            if name.startswith("nodes_"):
                version_str = name.replace("nodes_", "").replace("_", ".")
                try:
                    versions.append(parse_version(version_str))
                except InvalidVersion:
                    print(f"Предупреждение: Некорректное имя коллекции проигнорировано: {name}")
                    continue
        if not versions:
            return None
        latest_version = max(versions)
        print(f"Найдена последняя версия: {latest_version}")
        return str(latest_version)

    def _find_entry_points(self, question: str, release_version: str, limit: int = 5) -> list:
        print(f"Ищем точки входа для вопроса в релизе {release_version}...")
        nodes_collection, _ = self.qdrant_loader.get_collection_names(release_version)
        found_nodes = self.qdrant_loader.search_graph_nodes(
            text=question, 
            collection_name=nodes_collection, 
            limit=limit
        )
        print(f"Найдено {len(found_nodes)} возможных точек входа.")
        return found_nodes

    def _get_context_graph(self, entry_nodes: list, release_version: str, depth: int = 2) -> dict:
        print(f"Строим контекстный граф на основе {len(entry_nodes)} точек входа...")
        node_ids = [node.get('id') or node.get('name') for node in entry_nodes]
        node_ids = [id for id in node_ids if id]
        if not node_ids:
            return {"nodes": [], "relationships": []}
        subgraph = self.neo4j_driver.get_subgraph_for_nodes(node_ids, release_version, depth)
        print(f"Контекстный граф получен: {len(subgraph.get('nodes', []))} узлов, {len(subgraph.get('relationships', []))} связей.")
        return subgraph

    def _enrich_context_with_source_text(self, context_graph: dict, release_version: str) -> str:
        print(f"Обогащаем контекст исходным текстом...")
        chunk_ids = set()
        for node in context_graph.get("nodes", []):
            source_id = node.get("source_chunk_id")
            if isinstance(source_id, list):
                chunk_ids.update(source_id)
            elif source_id:
                chunk_ids.add(source_id)
        if not chunk_ids:
            print("Не найдено ссылок на исходные чанки.")
            return ""
        _, chunks_collection = self.qdrant_loader.get_collection_names(release_version)
        print(f"Запрашиваем {len(chunk_ids)} исходных чанков из коллекции '{chunks_collection}'...")
        source_texts = self.qdrant_loader.get_chunks_by_ids(list(chunk_ids), chunks_collection)
        full_context = "\n\n---\n\n".join(source_texts)
        print(f"Контекст обогащен текстом длиной {len(full_context)} символов.")
        return full_context

    def _synthesize_final_answer(self, question: str, context_graph: dict, enriched_text: str) -> dict:
        print(f"Синтезируем финальный ответ...")
        def json_serializer(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        context_graph_str = json.dumps(context_graph, indent=2, ensure_ascii=False, default=json_serializer)
        prompt = QNA_PROMPT.format(
            question=question,
            context_graph=context_graph_str,
            enriched_text=enriched_text
        )
        response_str = self.llm_manager.call_text_llm(messages=[{"role": "user", "content": prompt}], model_type=ModelType.SMART_TEXT)
        print("Финальный ответ получен. Используем ResultProcessor для извлечения JSON...")
        structured_answer = self.result_processor.extract_json(response_str)
        if not structured_answer:
            print(f"Ошибка: Не удалось распарсить JSON из ответа модели. Ответ был: {response_str}")
            return {"answer": response_str, "reasoning": "Failed to parse JSON", "source_ids": [], "confidence_score": 0.0}
        return structured_answer