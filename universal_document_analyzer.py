import io
import logging
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
from prompts.promts import ENTITIES_PROMPT, REQUIREMENTS_PROMPT, BATCH_DEPENDENCY_PROMPT, PAIR_DEPENDENCY_PROMPT
from services.CacheService import CacheService
from db_helper import Neo4jWriter

class UniversalMultimodalAnalyzer:
    def __init__(self, config: dict, release_version: str, files: list):
        self.config = config
        self.release_version = release_version
        self.files = files
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheService()
        self.llm_manager = LLMClientManager(self.config)
        self.result_processor = ResultProcessor(self.llm_manager)
        self.qdrant_loader = QdrantVectorLoader(self.llm_manager, collection_name=f"reqs_{self.release_version}")
        self.neo4j_writer = Neo4jWriter(uri=config.get("neo4j_uri"), user=config.get("neo4j_user"), password=config.get("neo4j_password"))

    def analyze(self) -> dict:
        all_entities = []
        all_requirements = []
        canonical_entity_names = set()

        for file_data in self.files:
            filename = file_data['filename']
            content = file_data['content']
            print(f"\n--- Обработка файла: {filename} ---")
            document_text = self._process_file(content, filename)
            chunks = self.split_to_chunks(document_text)
            model_name = self.llm_manager.get_text_model_name()

            entities = self._extract_entities(chunks, model_name, filename, canonical_entity_names)
            all_entities.extend(entities)
            
            for entity in entities:
                canonical_entity_names.add(entity['name'])

            requirements = self._extract_requirements(chunks, entities, model_name, filename)
            all_requirements.extend(requirements)

        final_entities = self._merge_entities(all_entities)
        unique_reqs = {}
        for i, req in enumerate(all_requirements):
            req_text = req.get("text", "").strip()
            if req_text and req_text not in unique_reqs:
                req["id"] = f"REQ-{i+1:03d}"
                unique_reqs[req_text] = req
        final_requirements = list(unique_reqs.values())

        print(f"\n--- Всего найдено уникальных сущностей: {len(final_entities)} ---")
        print(f"--- Всего найдено уникальных требований: {len(final_requirements)} ---")

        try:
            print("\nЭтап 3: Индексация требований в векторную базу...")
            self.qdrant_loader.load_requirements(final_requirements, self.release_version)
            print("Индексация завершена.")

            print("\nЭтап 4: Поиск и классификация зависимостей...")
            dependencies = self._find_and_classify_dependencies_batched(final_requirements, final_entities)
            print(f"Найдено {len(dependencies)} зависимостей.")

            final_result = {
                "entities": final_entities,
                "requirements": final_requirements,
                "dependencies": dependencies
            }

            print("\nЭтап 5: Сохранение результатов в Neo4j...")
            self.neo4j_writer.write_results(final_result, self.release_version)

            return final_result
        finally:
            print("\nЭтап 6: Удаление временной коллекции из Qdrant...")
            self.qdrant_loader.cleanup()
            self.neo4j_writer.close()

    def _process_file(self, file_content: bytes, filename: str) -> str:
        try:
            logging.info(f"Starting processing of file: {filename}")
            ext = os.path.splitext(filename)[1].lower()
            document_processor = DocumentProcessor(temp_dir=self.temp_dir, cache=self.cache)
            
            if ext == '.docx':
                text, _ = document_processor.process_docx(io.BytesIO(file_content), filename)
            elif ext == '.pdf':
                text, _ = document_processor.process_pdf(file_content, filename)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            if not text:
                logging.warning(f"No text extracted from {filename}")

            image_processor = ImageProcessor(temp_dir=self.temp_dir, config=self.config, cache=self.cache)
            processed_text = image_processor.process_images_in_text(text)
            logging.info(f"Successfully processed file: {filename}")
            return processed_text
        except Exception as e:
            logging.error(f"Failed to process file {filename}: {e}", exc_info=True)
            raise

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 2) -> dict:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(max_retries):
            try:
                response = self.llm_manager.call_text_llm(messages)
                parsed = self.result_processor.extract_json(response)
                if parsed:  # Успешный парсинг
                    return parsed
                logging.warning(f"Attempt {attempt + 1}/{max_retries}: Failed to extract JSON, retrying...")
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{max_retries}: Error calling LLM or parsing: {e}", exc_info=True)
        
        logging.error(f"Failed to get valid JSON after {max_retries} attempts.")
        raise RuntimeError(f"Could not get a valid response from LLM after {max_retries} attempts.")

    def _extract_entities(self, chunks: list, model_name: str, source_file: str, existing_entity_names: set) -> list:
        extracted_entities = []
        file_name = os.path.basename(source_file)
        current_known_names = existing_entity_names.copy()
        for i, chunk in enumerate(chunks):
            print(f"  Обработка чанка {i + 1}/{len(chunks)} для сущностей...")
            entities_list_str = json.dumps(list(current_known_names), ensure_ascii=False, indent=2) if current_known_names else "[]"
            prompt = ENTITIES_PROMPT.format(text=chunk, existing_entities=entities_list_str)
            hash_key = f"entities:ch-{i+1}:{self.cache.generate_hash(prompt, model_name)}"
            if self.cache.exists(hash_key):
                parsed = self.cache.get(hash_key)
            else:
                parsed = self._call_llm_with_retry(prompt)
                if parsed:
                    self.cache.set(hash_key, parsed)
            
            chunk_entities = parsed.get("entities", [])
            for entity in chunk_entities:
                entity["source_file"] = file_name
                entity["release_version"] = self.release_version
                extracted_entities.append(entity)
                current_known_names.add(entity['name'])
        return extracted_entities

    def _extract_requirements(self, chunks: list, entities: list, model_name: str, source_file: str) -> list:
        extracted_requirements = []
        file_name = os.path.basename(source_file)
        entities_context = json.dumps({"entities": entities}, indent=2, ensure_ascii=False)
        for i, chunk in enumerate(chunks):
            print(f"  Обработка чанка {i + 1}/{len(chunks)} для требований...")
            prompt = REQUIREMENTS_PROMPT.format(text=chunk, entities_context=entities_context)
            hash_key = f"requirements:{self.cache.generate_hash(prompt, model_name)}"
            if self.cache.exists(hash_key):
                parsed = self.cache.get(hash_key)
            else:
                parsed = self._call_llm_with_retry(prompt)
                if parsed:
                    self.cache.set(hash_key, parsed)

            for req in parsed.get("requirements", []):
                req["source_file"] = file_name
                req["release_version"] = self.release_version
                extracted_requirements.append(req)
        return extracted_requirements

    def _find_and_classify_dependencies_batched(self, requirements: list, entities: list) -> list:
        if not requirements:
            return []

        candidate_pairs = self._get_candidate_pairs(requirements, entities)
        candidates_by_source = defaultdict(list)
        req_map = {req["id"]: req for req in requirements}
        for id1, id2 in candidate_pairs:
            if id1 in req_map and id2 in req_map:
                candidates_by_source[id1].append(req_map[id2])

        all_dependencies = []
        model_name = self.llm_manager.get_text_model_name()
        print(f"\nНайдено {len(candidate_pairs)} пар-кандидатов. Группировка в {len(candidates_by_source)} батчей для анализа зависимостей.")

        for i, (source_id, initial_candidates) in enumerate(candidates_by_source.items()):
            print(f"  Анализ батча {i+1}/{len(candidates_by_source)} для требования {source_id}...")
            source_req = req_map[source_id]
            
            stack = [(initial_candidates, self._get_relevant_entities(requirements, entities, {c['id'] for c in initial_candidates} | {source_id}))]

            while stack:
                current_candidates, current_entities = stack.pop()
                
                candidates_list_str = "\n".join([f"- ID: {c['id']}, Текст: \"{c['text']}\"" for c in current_candidates])
                entities_context_str = json.dumps(current_entities, indent=2, ensure_ascii=False)

                prompt = BATCH_DEPENDENCY_PROMPT.format(
                    req_a_id=source_id,
                    req_a_text=source_req['text'],
                    candidates_list=candidates_list_str,
                    relevant_entities=entities_context_str
                )

                if len(prompt) > 8500 and len(current_candidates) > 1:
                    print(f"    -> Батч из {len(current_candidates)} кандидатов слишком большой. Разделяем.")
                    mid = len(current_candidates) // 2
                    part1 = current_candidates[:mid]
                    part2 = current_candidates[mid:]
                    
                    entities1 = self._get_relevant_entities(requirements, entities, {c['id'] for c in part1} | {source_id})
                    entities2 = self._get_relevant_entities(requirements, entities, {c['id'] for c in part2} | {source_id})
                    
                    stack.append((part2, entities2))
                    stack.append((part1, entities1))
                    continue

                hash_key = f"batch_deps:{self.cache.generate_hash(prompt, model_name)}"
                if self.cache.exists(hash_key):
                    parsed = self.cache.get(hash_key)
                else:
                    parsed = self._call_llm_with_retry(prompt)
                    if parsed:
                        self.cache.set(hash_key, parsed)

                if parsed and "dependencies" in parsed:
                    all_dependencies.extend(parsed.get("dependencies", []))
        
        return all_dependencies

    def _get_candidate_pairs(self, requirements: list, entities: list) -> set:
        candidate_pairs = set()
        similarity_threshold = 0.75
        common_entity_stopwords = {"система", "пользователь"}

        for req in requirements:
            similar = self.qdrant_loader.search_requirements(req["text"], self.release_version, limit=5)
            for other in similar:
                if req["id"] != other["id"] and other["score"] > similarity_threshold:
                    candidate_pairs.add(tuple(sorted((req["id"], other["id"]))))

        entity_names = {e['name'].lower() for e in entities} - common_entity_stopwords
        req_to_entities = {req['id']: {word for word in re.findall(r'\w+', req['text'].lower()) if word in entity_names} for req in requirements}
        for r1_id, r2_id in combinations(req_to_entities.keys(), 2):
            if req_to_entities[r1_id] & req_to_entities[r2_id]:
                candidate_pairs.add(tuple(sorted((r1_id, r2_id))))

        for req in requirements:
            if req.get("trigger_words"):
                for other_req in requirements:
                    if req["id"] != other_req["id"]:
                        candidate_pairs.add(tuple(sorted((req["id"], other_req["id"]))))
        
        return candidate_pairs

    def _get_relevant_entities(self, requirements: list, all_entities: list, requirement_ids: set) -> list:
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

    def _merge_entities(self, entities: list) -> list:
        merged_entities = defaultdict(lambda: {"attributes": [], "states": [], "descriptions": []})

        for entity in entities:
            key = entity['name'].lower()
            merged_entities[key]['name'] = entity['name'] # Keep original capitalization
            if 'description' in entity and entity['description']:
                merged_entities[key]['descriptions'].append(entity['description'])
            if 'attributes' in entity:
                merged_entities[key]['attributes'].extend(entity['attributes'])
            if 'states' in entity:
                merged_entities[key]['states'].extend(entity['states'])

        final_list = []
        for key, value in merged_entities.items():
            # Combine descriptions, preferring the longest one
            if value['descriptions']:
                value['description'] = max(value['descriptions'], key=len)
            else:
                value['description'] = ""
            del value['descriptions']

            # Deduplicate attributes and states
            value['attributes'] = list({attr['name']: attr for attr in value['attributes']}.values())
            value['states'] = list({state['name']: state for state in value['states']}.values())
            final_list.append(value)

        return final_list