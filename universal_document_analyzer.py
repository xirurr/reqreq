import os
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter

from LLMClientManager import LLMClientManager
from processors.document_processor import DocumentProcessor
from processors.image_processor import ImageProcessor
from processors.result_processor import ResultProcessor
from prompts.promts import extractor_prompt_template
from services.CacheService import CacheService


class UniversalMultimodalAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheService()

    def analyze_file(self, file_path: str) -> dict:
        """
        Анализирует файл (DOCX), извлекая текст и изображения

        :param file_path: Путь к файлу документа
        :return: Структурированные результаты
        """
        # Определяем тип файла
        ext = os.path.splitext(file_path)[1].lower()

        document_processor = DocumentProcessor(temp_dir=self.temp_dir, cache=self.cache)
        if ext == '.docx':
            text, image_files = document_processor.process_docx(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")

        # Устанавливаем директорию с изображениями
        self.config["image_directory"] = os.path.dirname("parsed/images")

        """
               Анализирует документ с встроенными ссылками на изображения
               Формат ссылок: [IMAGE:path/to/image.png] Подпись изображения
               """
        # Обработка изображений в тексте
        image_processor = ImageProcessor(temp_dir=self.temp_dir, config=self.config, cache=self.cache)
        text_with_described_images = image_processor.process_images_in_text(text)

        # Анализируем документ
        document = self._analyze_parsed_document(text_with_described_images)
        return document

    def _analyze_parsed_document(self, document: str) -> dict:
        chunks = self.split_to_chunks(document)

        results = {"requirements": [], "system_states": [], "entities": []}
        llm_client_manager = LLMClientManager(self.config)
        model_name = llm_client_manager.get_text_model_name()
        # Обработка каждого чанка
        for i, chunk in enumerate(chunks):
            print(f"Обработка чанка {i + 1}/{len(chunks)}")
            try:
                formated_prompt = extractor_prompt_template.format(text=chunk)

                hash_key = "chunk:"+self.cache.generate_hash(formated_prompt, model_name)
                if self.cache.exists(hash_key):
                    parsed = self.cache.get(hash_key)
                else:
                    response = llm_client_manager.call_text_llm(formated_prompt)
                    parsed = ResultProcessor.extract_json(response)
                    self.cache.set(hash_key, parsed)
                ResultProcessor.merge_results(results, parsed)
            except Exception as e:
                print(f"Ошибка обработки чанка: {str(e)}")

        #return ResultProcessor.postprocess_results(results)
        return results

    def split_to_chunks(self, document):
        # Разбиение документа на чанки
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 3000),
            chunk_overlap=200,
            separators=["\n\n", "\n", ". "]
        )
        chunks = splitter.split_text(document)
        return chunks
