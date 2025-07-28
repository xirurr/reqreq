import io
import logging
import os
import tempfile
import json
import re
import uuid
import hashlib
from itertools import combinations
from collections import defaultdict

from langchain.text_splitter import RecursiveCharacterTextSplitter

from LLMClientManager import LLMClientManager
from QdrantVectorLoader import QdrantVectorLoader
from processors.document_processor import DocumentProcessor, SafeDocumentProcessor
from processors.image_processor import ImageProcessor
from processors.result_processor import ResultProcessor
from models.model_types import ModelType
from prompts.prompt_factory import PromptFactory
from prompts.prompt_factory import PromptFactory

class UniversalMultimodalAnalyzer:
    def __init__(self, config: dict, release_version: str, files: list):
        self.config = config
        self.release_version = release_version
        self.files = files
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheService()
        self.llm_manager = LLMClientManager(self.config)
        self.result_processor = ResultProcessor(self.llm_manager)
        self.qdrant_loader = QdrantVectorLoader(self.llm_manager, cache=self.cache)
        self.neo4j_writer = Neo4jWriter(uri=config.get("neo4j_uri"), user=config.get("neo4j_user"),
                                        password=config.get("neo4j_password"))
        self.prompt_factory = PromptFactory()
        self.prompt_factory = PromptFactory()

    def analyze(self) -> dict:
        # --- Этап 1: Извлечение данных из документов ---
        all_entities = []
        all_requirements = []
        all_chunks = []
        canonical_entity_names = set()

        for file_data in self.files:
            filename = file_data['filename']
            content = file_data['content']
            print(f"\n--- Обработка файла: {filename} ---")
            document_text = self._process_file(content, filename)
            chunks = self.split_to_chunks(document_text)
            all_chunks.extend([(chunk, filename) for chunk in chunks])

            entities_from_file = self._extract_entities(chunks, filename, canonical_entity_names)
            merged_entities_for_file = self._merge_entities(entities_from_file)
            all_entities.extend(merged_entities_for_file)

            for entity in merged_entities_for_file:
                canonical_entity_names.add(entity['name'])

            requirements = self._extract_requirements(chunks, merged_entities_for_file, filename)
            all_requirements.extend(requirements)

        # --- Этап 2: Подготовка временных данных для анализа зависимостей ---
        temp_reqs = self._get_unique_temp_reqs(all_requirements)
        temp_collection_name = f"temp_analysis_{uuid.uuid4().hex}"

        try:
            print(f"\n--- Создание временной коллекции \"{temp_collection_name}\" для анализа зависимостей ---")
            self.qdrant_loader.load_graph_nodes(temp_reqs, temp_collection_name)

            # --- Этап 3: Поиск и классификация зависимостей ---
            print("\n--- Поиск и классификация зависимостей ---")
            dependencies = self._find_and_classify_dependencies_batched(temp_reqs, all_entities, temp_collection_name)
            print(f"Найдено {len(dependencies)} зависимостей.")

            # --- Этап 4: Присвоение стабильных ID и финализация данных ---
            print("\n--- Присвоение стабильных ID требованиям ---")
            final_requirements = self._assign_stable_requirement_ids(temp_reqs)
            final_entities = self._merge_entities(all_entities)

            final_dependencies = self._update_dependency_ids(dependencies, final_requirements)

            final_result = {
                "entities": final_entities,
                "requirements": final_requirements,
                "dependencies": final_dependencies
            }

            # --- Этап 5: Сохранение в постоянные хранилища ---
            print("\n--- Сохранение результатов в постоянные хранилища ---")
            nodes_collection, chunks_collection = self.qdrant_loader.get_collection_names(self.release_version)
            
            print(f"  Сохранение {len(final_entities) + len(final_requirements)} узлов графа в коллекцию '{nodes_collection}'...")
            self.qdrant_loader.load_graph_nodes(final_entities + final_requirements, nodes_collection)
            
            print(f"  Сохранение {len(all_chunks)} исходных чанков в коллекцию '{chunks_collection}'...")
            chunks_by_file = defaultdict(list)
            for chunk_text, source_file in all_chunks:
                chunks_by_file[source_file].append(chunk_text)
            for source_file, file_chunks in chunks_by_file.items():
                self.qdrant_loader.load_source_chunks(file_chunks, chunks_collection, source_file)

            self.neo4j_writer.write_results(final_result, self.release_version)

            return final_result

        finally:
            print(f"\n--- Удаление временной коллекции \"{temp_collection_name}\" ---")
            self.qdrant_loader.delete_collection(temp_collection_name)
            self.neo4j_writer.close()
            print("\nАнализ завершен.")

    def _get_unique_temp_reqs(self, all_requirements: list) -> list:
        unique_reqs = {}
        for i, req in enumerate(all_requirements):
            req_text = req.get("text", "").strip()
            if req_text and req_text not in unique_reqs:
                req["id"] = f"TEMP_REQ-{i + 1:03d}"
                unique_reqs[req_text] = req
        return list(unique_reqs.values())

    def _update_dependency_ids(self, dependencies: list, final_requirements: list) -> list:
        id_map = {req.get('temp_id'): req.get('id') for req in final_requirements if req.get('temp_id')}
        updated_deps = []
        for dep in dependencies:
            source_id = id_map.get(dep['source'])
            target_id = id_map.get(dep['target'])
            if source_id and target_id:
                updated_deps.append({"source": source_id, "target": target_id, "type": dep["type"]})
        return updated_deps

    def _generate_chunk_id(self, source_file: str, chunk_text: str) -> str:
        """Генерирует детерминированный ID для чанка на основе его содержимого."""
        chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
        return f"{os.path.basename(source_file)}_{chunk_hash[:16]}"

    def _extract_entities(self, chunks: list, source_file: str, existing_entity_names: set) -> list:
        extracted_entities = []
        file_name = os.path.basename(source_file)
        current_known_names = existing_entity_names.copy()
        for chunk in chunks:
            chunk_id = self._generate_chunk_id(file_name, chunk)
            print(f"  Обработка чанка {chunk_id} для сущностей...")
            sorted_known_names = sorted(list(current_known_names))
            prompt = self.prompt_factory.get_entities_prompt(text=chunk, existing_entities=sorted_known_names)
            
            parsed = self._get_cached_text_llm_response(prompt, ModelType.DEFAULT_TEXT, "entities")

            chunk_entities = parsed.get("entities", [])
            for entity in chunk_entities:
                entity["source_file"] = file_name
                entity["source_chunk_id"] = chunk_id
                entity["release_version"] = self.release_version
                extracted_entities.append(entity)
                current_known_names.add(entity['name'])
        return extracted_entities

    def _extract_requirements(self, chunks: list, entities: list, source_file: str) -> list:
        extracted_requirements = []
        file_name = os.path.basename(source_file)
        # Создаем "чистую" версию сущностей специально для промпта
        prompt_entities = []
        for entity in entities:
            clean_entity = {
                "name": entity.get("name"),
                "description": entity.get("description"),
                "attributes": entity.get("attributes"),
                "states": entity.get("states")
            }
            # Убираем ключи с пустыми значениями, чтобы сделать промпт еще чище
            prompt_entities.append({k: v for k, v in clean_entity.items() if v})

        # Сортируем "чистые" данные непосредственно перед созданием JSON для кеширования
        prompt_entities.sort(key=lambda x: x.get('name', ''))

        entities_context = {"entities": prompt_entities}

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(file_name, chunk)
            print(f"  Обработка чанка {chunk_id}({i + 1}/{len(chunks)}) для требований...")
            prompt = self.prompt_factory.get_requirements_prompt(text=chunk, entities_context=entities_context)

            parsed = self._get_cached_text_llm_response(prompt, ModelType.DEFAULT_TEXT, "requirements")

            for req in parsed.get("requirements", []):
                req["source_file"] = file_name
                req["source_chunk_id"] = chunk_id
                req["release_version"] = self.release_version
                extracted_requirements.append(req)
        return extracted_requirements

    def _find_and_classify_dependencies_batched(self, requirements: list, entities: list, collection_name: str) -> list:
        if not requirements:
            return []

        candidate_pairs = sorted(list(self._get_candidate_pairs(requirements, entities, collection_name)))
        candidates_by_source = defaultdict(list)
        req_map = {req["id"]: req for req in requirements}

        for id1, id2 in candidate_pairs:
            if id1 in req_map and id2 in req_map:
                source_id, target_id = min(id1, id2), max(id1, id2)
                if source_id in req_map and target_id in req_map:
                    candidates_by_source[source_id].append(req_map[target_id])
                    candidates_by_source[target_id].append(req_map[source_id])

        for source_id in candidates_by_source:
            unique_candidates = sorted(list({v['id']: v for v in candidates_by_source[source_id]}.values()), key=lambda x: x['id'])
            candidates_by_source[source_id] = unique_candidates

        all_dependencies = []
        print(f"\nНайдено {len(candidate_pairs)} пар-кандидатов. Группировка в {len(candidates_by_source)} батчей для анализа зависимостей.")

        sorted_source_ids = sorted(candidates_by_source.keys())

        for i, source_id in enumerate(sorted_source_ids):
            initial_candidates = candidates_by_source[source_id]
            print(f"  Анализ батча {i+1}/{len(sorted_source_ids)} для требования {source_id}...")
            source_req = req_map[source_id]

            stack = [(initial_candidates, self._get_relevant_entities(requirements, entities, {c['id'] for c in initial_candidates} | {source_id}))]

            while stack:
                current_candidates, current_entities = stack.pop()
                sorted_candidates = sorted(current_candidates, key=lambda x: x.get('id', ''))
                sorted_entities = sorted(current_entities, key=lambda x: x.get('name', ''))

                candidates_list_str = "\n".join([f"- ID: {c['id']}, Текст: \"{c['text']}\"" for c in sorted_candidates])
                
                prompt = self.prompt_factory.get_batch_dependency_prompt(
                    req_a_id=source_id,
                    req_a_text=source_req['text'],
                    candidates_list=candidates_list_str,
                    relevant_entities=sorted_entities
                )

                if len(prompt) > 7500 and len(current_candidates) > 1:
                    print(f"    -> Батч из {len(current_candidates)} кандидатов слишком большой. Разделяем.")
                    mid = len(current_candidates) // 2
                    part1 = current_candidates[:mid]
                    part2 = current_candidates[mid:]
                    entities1 = self._get_relevant_entities(requirements, entities, {c['id'] for c in part1} | {source_id})
                    entities2 = self._get_relevant_entities(requirements, entities, {c['id'] for c in part2} | {source_id})
                    stack.append((part2, entities2))
                    stack.append((part1, entities1))
                    continue

                parsed = self._get_cached_text_llm_response(prompt, ModelType.SMART_TEXT, "batch_deps")

                if parsed and "dependencies" in parsed:
                    all_dependencies.extend(parsed.get("dependencies", []))

        return all_dependencies

    def _get_candidate_pairs(self, requirements: list, entities: list, collection_name: str) -> set:
        candidate_pairs = set()
        similarity_threshold = 0.75
        common_entity_stopwords = {"система", "пользователь"}

        for req in requirements:
            similar = self.qdrant_loader.search_graph_nodes(req["text"], collection_name, limit=5)
            for other in similar:
                if other.get('text') and req["id"] != other.get("id") and other.get("score", 0) > similarity_threshold:
                    candidate_pairs.add(tuple(sorted((req["id"], other.get("id")))))

        entity_names = {e['name'].lower() for e in entities} - common_entity_stopwords
        req_to_entities = {req['id']: {word for word in re.findall(r'\w+', req['text'].lower()) if word in sorted(entity_names)} for req in requirements}
        for r1_id, r2_id in combinations(req_to_entities.keys(), 2):
            if r1_id in req_to_entities and r2_id in req_to_entities and req_to_entities[r1_id] & req_to_entities[r2_id]:
                candidate_pairs.add(tuple(sorted((r1_id, r2_id))))

        for req in requirements:
            if req.get("trigger_words"):
                for other_req in requirements:
                    if req["id"] != other_req["id"]:
                        candidate_pairs.add(tuple(sorted((req["id"], other_req["id"]))))

        return candidate_pairs

    def _assign_stable_requirement_ids(self, requirements: list) -> list:
        if not requirements:
            return []

        existing_reqs = self.neo4j_writer.get_existing_requirements()
        # Получаем все существующие категории один раз в начале
        existing_categories = self.neo4j_writer.get_all_categories()

        for req in requirements:
            req['temp_id'] = req['id']

        prompt = self._build_matching_prompt(existing_reqs, requirements)
        match_result = self._get_cached_text_llm_response(prompt, ModelType.SMART_TEXT, "req_matching")

        if match_result:
            for req in requirements:
                temp_id = req['temp_id']
                matched_id = match_result.get(str(temp_id))
                if matched_id:
                    req['id'] = matched_id
                else:
                    category = self._get_requirement_category(req['text'], ModelType.DEFAULT_TEXT, existing_categories)
                    if category not in existing_categories:
                        existing_categories.append(category)
                    
                    next_index = self.neo4j_writer.get_next_req_id_index(category)
                    req['id'] = f"REQ-{category}-{next_index:03d}"
        return requirements
    
    def _process_file(self, file_content: bytes, filename: str) -> str:
        try:
            logging.info(f"Starting processing of file: {filename}")
            ext = os.path.splitext(filename)[1].lower()
            document_processor = DocumentProcessor(temp_dir=self.temp_dir, cache=self.cache)
            safe_document_processor = SafeDocumentProcessor(temp_dir=self.temp_dir, cache=self.cache)

            if ext == '.docx':
                text, _ = safe_document_processor.process_docx(io.BytesIO(file_content), filename)
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

    def _call_text_llm_with_retry(self, prompt: str, model_type: ModelType, max_retries: int = 2) -> dict | None:
        """Вызывает LLM и в режиме чата просит исправить JSON, если он некорректен."""
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries):
            response_str = self.llm_manager.call_text_llm(messages, model_type=model_type)
            parsed_json = self.result_processor.extract_json(response_str)

            if parsed_json:
                return parsed_json  # Успех!

            logging.warning(f"Attempt {attempt + 1}/{max_retries}: Не удалось извлечь JSON. Просим модель уточнить ответ.")

            # Если не удалось, добавляем сообщения в историю для следующей попытки
            messages.append({"role": "assistant", "content": response_str})
            messages.append({
                "role": "user",
                "content": "Твой предыдущий ответ не является валидным JSON. Пожалуйста, исправь его и верни **только** JSON-объект, обернутый в теги `<json>` и `</json>`."
            })

        logging.error(f"Не удалось получить валидный JSON после {max_retries} попыток чата.")
        return None

    def _get_cached_text_llm_response(self, prompt: str, model_type: ModelType, cache_key_prefix: str) -> dict:
        model_name = self.llm_manager.get_model_name(model_type)
        hash_key = f"{cache_key_prefix}:{self.cache.generate_hash(prompt, model_name)}"

        if self.cache.exists(hash_key):
            return self.cache.get(hash_key)
        else:
            # Теперь этот метод возвращает готовый словарь или None
            parsed_response = self._call_text_llm_with_retry(prompt, model_type)
            if parsed_response:
                self.cache.set(hash_key, parsed_response)
            # Возвращаем словарь или пустой словарь, если был None
            return parsed_response or {}

    def _get_relevant_entities(self, requirements: list, all_entities: list, requirement_ids: set) -> list:
        req_texts = " ".join([req['text'] for req in requirements if req['id'] in requirement_ids])
        relevant_entity_names = {e['name'] for e in all_entities if e['name'].lower() in req_texts.lower()}
        return [e for e in all_entities if e['name'] in relevant_entity_names]

    def _get_requirement_category(self, req_text: str, model_type: ModelType, existing_categories: list[str]) -> str:
        prompt = self.prompt_factory.get_requirement_category_prompt(req_text=req_text, existing_categories=existing_categories)
        
        parsed_response = self._get_cached_text_llm_response(prompt, model_type, "req_category")

        try:
            category = parsed_response.get("category", "UNCATEGORIZED").strip().upper().replace(" ", "_")
            return category or "UNCATEGORIZED"
        except (AttributeError, KeyError) as e:
            logging.error(f"Failed to get or parse category for requirement: {req_text}. Error: {e}")
            return "UNCATEGORIZED"

    def _build_matching_prompt(self, old_reqs: list, new_reqs: list) -> str:
        old_reqs_str = json.dumps([{'id': r['id'], 'text': r['text']} for r in old_reqs], ensure_ascii=False, indent=2)
        new_reqs_str = json.dumps([{'temp_id': r['temp_id'], 'text': r['text']} for r in new_reqs], ensure_ascii=False, indent=2)

        return f"""Ниже даны два списка требований: 'старые' (с существующими ID) и 'новые' (с временными ID).\n\nТвоя задача: Для каждого 'нового' требования найди семантически эквивалентное 'старое' требование.\n\nФормат ответа: Верни JSON объект, где ключ - это 'temp_id' нового требования, а значение - это 'id' старого требования, которому оно соответствует. Если для нового требования нет соответствия среди старых, используй `null`.\n\nСтарые требования:\n{old_reqs_str}\n\nНовые требования:\n{new_reqs_str}\n\nОтвет:\n"""

    def split_to_chunks(self, document: str) -> list:
        """Интеллектуальное деление на чанки, сохраняющее блоки описания изображений."""
        
        # Паттерн ищет блок [START...END] и опционально следующую за ним строку с подписью "Рисунок..."
        pattern = re.compile(r"(\[START OF IMAGE DESCRIPTION:.*?\].*?\[END OF IMAGE DESCRIPTION\](?:\n*Рисунок[^\n]+)?)", re.DOTALL)
        
        # 1. Находим все семантические блоки и заменяем их плейсхолдерами
        placeholders = {}
        temp_document = document
        
        # Используем finditer, чтобы получить объекты совпадений и их позиции
        matches = list(pattern.finditer(document))
        
        # Итерируемся в обратном порядке, чтобы не сбивать позиции при замене
        for i, match in enumerate(reversed(matches)):
            placeholder = f"__SEMANTIC_BLOCK_{len(matches) - 1 - i}__"
            full_block_text = match.group(1)
            
            # Проверяем, не превышает ли блок размер чанка
            if len(full_block_text) > self.config.get("chunk_size", 3000):
                print(f"Warning: Блок с изображением превышает размер чанка и может быть обрезан. Path: {match.group(1)[:100]}...")
                # В этом случае мы не заменяем его, оставляя стандартному сплиттеру
                continue

            placeholders[placeholder] = full_block_text
            # Заменяем найденный блок на плейсхолдер
            start, end = match.span(1)
            temp_document = temp_document[:start] + placeholder + temp_document[end:]

        # 2. Используем стандартный сплиттер на "облегченном" документе
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 3000),
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
        chunks_with_placeholders = splitter.split_text(temp_document)

        # 3. "Раскрываем" плейсхолдеры обратно
        final_chunks = []
        for chunk in chunks_with_placeholders:
            for placeholder, original_text in placeholders.items():
                chunk = chunk.replace(placeholder, original_text)
            final_chunks.append(chunk)
            
        return final_chunks

    def _merge_entities(self, entities: list) -> list:
        merged_entities = defaultdict(lambda: {"attributes": [], "states": [], "descriptions": [], "source_chunk_ids": set()})

        for entity in entities:
            key = entity['name'].lower()
            merged_entities[key]['name'] = entity['name']
            if 'description' in entity and entity['description']:
                merged_entities[key]['descriptions'].append(entity['description'])
            if 'attributes' in entity:
                merged_entities[key]['attributes'].extend(entity['attributes'])
            if 'states' in entity:
                merged_entities[key]['states'].extend(entity['states'])
            if 'source_chunk_id' in entity:
                # Проверяем, является ли значение списком или отдельной строкой
                if isinstance(entity['source_chunk_id'], list):
                    merged_entities[key]['source_chunk_ids'].update(entity['source_chunk_id'])
                else:
                    merged_entities[key]['source_chunk_ids'].add(entity['source_chunk_id'])

        final_list = []
        for key, value in merged_entities.items():
            if value['descriptions']:
                value['description'] = max(value['descriptions'], key=len)
            else:
                value['description'] = ""
            del value['descriptions']

            value['attributes'] = list({attr['name']: attr for attr in value['attributes']}.values())
            value['states'] = list({state['name']: state for state in value['states']}.values())
            value['source_chunk_id'] = list(value['source_chunk_ids'])
            del value['source_chunk_ids']
            final_list.append(value)

        final_list = sorted(final_list, key=lambda x: x.get('name', ''))
        return final_list