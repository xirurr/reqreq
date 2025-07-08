import os
import tempfile
import json
import re
from itertools import combinations
from collections import defaultdict

from langchain.text_splitter import RecursiveCharacterTextSplitter

from LLMClientManager import LLMClientManager
from QdrantVectorLoader import QdrantVectorLoader
from processors.document_processor import DocumentProcessor
from processors.image_processor import ImageProcessor
from processors.result_processor import ResultProcessor
from prompts.promts import ENTITIES_PROMPT, REQUIREMENTS_PROMPT, BATCH_DEPENDENCY_PROMPT
from services.CacheService import CacheService


class UniversalMultimodalAnalyzer:
    def __init__(self, config: dict, qdrant_client: QdrantVectorLoader):
        self.config = config
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheService()
        self.llm_manager = LLMClientManager(self.config)
        self.qdrant_loader = qdrant_client

    def analyze_file(self, file_path: str) -> dict:
        ext = os.path.splitext(file_path)[1].lower()
        document_processor = DocumentProcessor(temp_dir=self.temp_dir, cache=self.cache)
        if ext == '.docx':
            text, _ = document_processor.process_docx(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")

        image_processor = ImageProcessor(temp_dir=self.temp_dir, config=self.config, cache=self.cache)
        full_text = image_processor.process_images_in_text(text)

        return self._analyze_document_pipeline(full_text)

    def _analyze_document_pipeline(self, document_text: str) -> dict:
        chunks = self.split_to_chunks(document_text)
        model_name = self.llm_manager.get_text_model_name()

        # Этап 1: Извлечение сущностей
        print("Этап 1: Извлечение сущностей...")
        entities = self._extract_entities(chunks, model_name)
        print(f"Найдено {len(entities)} уникальных сущностей.")

        # Этап 2: Извлечение требований
        print("\nЭтап 2: Извлечение требований...")
        requirements = self._extract_requirements(chunks, entities, model_name)
        print(f"Найдено {len(requirements)} уникальных требований.")

        # Этап 3: Индексация требований
        print("\nЭтап 3: Индексация требований в векторную базу...")
        self.qdrant_loader.load_requirements(requirements)
        print("Индексация завершена.")

        # Этап 4: Поиск и классификация зависимостей
        print("\nЭтап 4: Поиск и классификация зависимостей...")
        dependencies = self._find_and_classify_dependencies_batched(requirements, entities)
        print(f"Найдено {len(dependencies)} зависимостей.")

        return {
            "entities": entities,
            "requirements": requirements,
            "dependencies": dependencies
        }

    def _extract_entities(self, chunks: list, model_name: str) -> list:
        # ... (код остался без изменений)
        all_entities = []
        for i, chunk in enumerate(chunks):
            print(f"  Обработка чанка {i + 1}/{len(chunks)}")
            prompt = ENTITIES_PROMPT.format(text=chunk)
            hash_key = f"entities:{self.cache.generate_hash(prompt, model_name)}"
            if self.cache.exists(hash_key):
                parsed = self.cache.get(hash_key)
            else:
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_manager.call_text_llm(messages)
                parsed = ResultProcessor.extract_json(response)
                self.cache.set(hash_key, parsed)
            all_entities.extend(parsed.get("entities", []))
        
        unique_entities = {item['name'].lower(): item for item in all_entities}
        return list(unique_entities.values())

    def _extract_requirements(self, chunks: list, entities: list, model_name: str) -> list:
        # ... (код остался без изменений, но теперь он извлекает и trigger_words)
        all_requirements = []
        entities_context = json.dumps({"entities": entities}, indent=2, ensure_ascii=False)
        for i, chunk in enumerate(chunks):
            print(f"  Обработка чанка {i + 1}/{len(chunks)}")
            prompt = REQUIREMENTS_PROMPT.format(text=chunk, entities_context=entities_context)
            hash_key = f"requirements:{self.cache.generate_hash(prompt, model_name)}"
            if self.cache.exists(hash_key):
                parsed = self.cache.get(hash_key)
            else:
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_manager.call_text_llm(messages)
                parsed = ResultProcessor.extract_json(response)
                self.cache.set(hash_key, parsed)
            all_requirements.extend(parsed.get("requirements", []))

        unique_reqs = {}
        for i, req in enumerate(all_requirements):
            req_text = req.get("text", "").strip()
            if req_text and req_text not in unique_reqs:
                req["id"] = f"REQ-{i+1:03d}"
                unique_reqs[req_text] = req
        return list(unique_reqs.values())

    def _find_and_classify_dependencies_batched(self, document_text: str, requirements: list, entities: list) -> list:
        if not requirements or not entities:
            return []

        candidate_pairs = self._get_candidate_pairs(requirements, entities)
        
        candidates_by_source = defaultdict(list)
        req_map = {req["id"]: req for req in requirements}
        for id1, id2 in candidate_pairs:
            candidates_by_source[id1].append(req_map[id2])

        all_dependencies = []
        print(f"\nНайдено {len(candidate_pairs)} пар-кандидатов. Группировка в {len(candidates_by_source)} батчей.")

        for i, (source_id, candidates) in enumerate(candidates_by_source.items()):
            print(f"  Анализ батча {i+1}/{len(candidates_by_source)} для требования {source_id} ({len(candidates)} кандидатов)")
            source_req = req_map[source_id]
            
            candidates_list_str = "\n".join([f"- ID: {c['id']}, Текст: \"{c['text']}\"" for c in candidates])
            involved_ids = {source_id} | {c['id'] for c in candidates}
            relevant_entities = self._get_relevant_entities(requirements, entities, involved_ids)
            entities_context_str = json.dumps(relevant_entities, indent=2, ensure_ascii=False)

            prompt = BATCH_DEPENDENCY_PROMPT.format(
                req_a_id=source_id,
                req_a_text=source_req['text'],
                candidates_list=candidates_list_str,
                relevant_entities=entities_context_str
            )

            if len(prompt) > 7000: # Условный лимит
                print(f"    ВНИМАНИЕ: Батч для {source_id} слишком большой. Переключаюсь на попарную проверку.")
                for candidate in candidates:
                    pair_prompt = PAIR_DEPENDENCY_PROMPT.format(
                        context=document_text, # Для попарной проверки даем полный контекст
                        text1=source_req["text"],
                        text2=candidate["text"]
                    )
                    messages = [{"role": "user", "content": pair_prompt}]
                    relation = self.llm_manager.call_text_llm(messages).strip()
                    if relation != "NO_RELATION":
                        all_dependencies.append({
                            "source": source_id,
                            "target": candidate["id"],
                            "type": relation
                        })
                continue

            messages = [{"role": "user", "content": prompt}]
            response = self.llm_manager.call_text_llm(messages)
            parsed = ResultProcessor.extract_json(response)
            
            if parsed and "dependencies" in parsed:
                all_dependencies.extend(parsed["dependencies"])

        return all_dependencies

    def _get_candidate_pairs(self, requirements: list, entities: list) -> set:
        candidate_pairs = set()
        similarity_threshold = 0.75 # Порог для семантического сходства
        common_entity_stopwords = {"система", "пользователь"} # Игнорируем общие сущности

        # 1. Семантический поиск
        for req in requirements:
            similar = self.qdrant_loader.search_requirements(req["text"], limit=5)
            for other in similar:
                if req["id"] != other["id"] and other["score"] > similarity_threshold:
                    candidate_pairs.add(tuple(sorted((req["id"], other["id"]))))

        # 2. Поиск по общим сущностям
        entity_names = {e['name'].lower() for e in entities} - common_entity_stopwords
        req_to_entities = {req['id']: {word for word in re.findall(r'\w+', req['text'].lower()) if word in entity_names} for req in requirements}
        for r1_id, r2_id in combinations(req_to_entities.keys(), 2):
            if req_to_entities[r1_id] & req_to_entities[r2_id]:
                candidate_pairs.add(tuple(sorted((r1_id, r2_id))))

        # 3. Поиск по словам-маркерам (упрощенная логика)
        for req in requirements:
            if req.get("trigger_words"):
                for other_req in requirements:
                    if req["id"] != other_req["id"]:
                        candidate_pairs.add(tuple(sorted((req["id"], other_req["id"]))))
        
        return candidate_pairs

    def _get_relevant_entities(self, requirements: list, all_entities: list, requirement_ids: set) -> list:
        """Возвращает только те сущности, которые упоминаются в данном наборе требований."""
        req_texts = " ".join([req['text'] for req in requirements if req['id'] in requirement_ids])
        relevant_entity_names = {e['name'] for e in all_entities if e['name'].lower() in req_texts.lower()}
        return [e for e in all_entities if e['name'] in relevant_entity_names]

    def split_to_chunks(self, document: str) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 3000),
            chunk_overlap=200,
            separators=["\n\n", "\n", ". "]
        )
        return splitter.split_text(document)
