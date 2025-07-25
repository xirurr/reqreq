import base64
import hashlib
import os
import re
import json
from io import BytesIO

from PIL import Image

from LLMClientManager import LLMClientManager
from models.model_types import ModelType
from services.CacheService import CacheService
from prompts.promts import UI_IMAGE_PROMPT
from processors.result_processor import ResultProcessor

class ImageProcessor:
    def __init__(self, temp_dir: str, config: dict, cache: CacheService):
        self.config = config
        self.temp_dir = temp_dir
        self.llm_client = LLMClientManager(config)
        self.cache = cache
        # Используем ResultProcessor для надежного извлечения JSON
        self.result_processor = ResultProcessor(self.llm_client)

    def process_images_in_text(self, text: str) -> str:
        """Находит в тексте маркеры изображений и заменяет их на структурированное описание."""
        # Паттерн для поиска маркеров вида [IMAGE:путь/к/файлу.png]
        pattern = r'\[IMAGE:([^\]]+)\]'
        
        # Используем re.sub с функцией-заменителем для обработки каждого найденного изображения
        processed_text = re.sub(pattern, self._replace_image_marker, text)
        
        return processed_text

    def _replace_image_marker(self, match: re.Match) -> str:
        """Функция, вызываемая для каждого найденного маркера изображения."""
        img_path = match.group(1).strip()
        
        # Получаем структурированное описание изображения
        description_json = self.get_image_description_json(img_path)

        # Если описание получить не удалось, возвращаем сообщение об ошибке
        if not description_json:
            return f"\n[IMAGE ANALYSIS FAILED: {img_path}]\n"

        # Сериализуем JSON обратно в красивую строку для вставки в текст
        description_str = json.dumps(description_json, indent=2, ensure_ascii=False)

        # Обрамляем результат в четкие маркеры
        return f"\n[START OF IMAGE DESCRIPTION: {img_path}]\n{description_str}\n[END OF IMAGE DESCRIPTION]\n"

    def get_image_description_json(self, img_path: str) -> dict | None:
        """Анализирует изображение и возвращает его описание в виде словаря (JSON)."""
        if not os.path.exists(img_path):
            print(f"Файл изображения не найден: {img_path}")
            return None

        # Ключ для кэша зависит от хэша файла и имени модели
        model_name = self.llm_client.get_model_name(model_type=ModelType.GRAPHIC)
        file_hash = self._get_file_hash(img_path)
        cache_key = f"img_json_desc:{self.cache.generate_hash(file_hash, model_name)}"

        if self.cache.exists(cache_key):
            return self.cache.get(cache_key)

        base64_image = self._encode_image(img_path)
        
        # Вызываем LLM с новым, правильным промптом
        raw_response = self.llm_client.call_graphic_llm(base64_image, UI_IMAGE_PROMPT)
        
        # Извлекаем JSON из ответа с помощью ResultProcessor
        description_json = self.result_processor.extract_json(raw_response)

        if description_json:
            self.cache.set(cache_key, description_json)
        
        return description_json

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Кодирует изображение в base64 для передачи в API."""
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            with BytesIO() as buffer:
                img.save(buffer, format="JPEG", quality=85)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _get_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """Вычисляет хэш файла для использования в качестве ключа кэша."""
        hash_algo = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)
        return hash_algo.hexdigest()