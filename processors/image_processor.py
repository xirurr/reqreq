import base64
import hashlib
import os
import re
from io import BytesIO

from PIL import Image

from LLMClientManager import LLMClientManager
from models.model_types import ModelType
from services.CacheService import CacheService


class ImageProcessor:
    def __init__(self, temp_dir: str, config: dict, cache: CacheService):
        self.config = config
        self.temp_dir = temp_dir
        self.llm_client = LLMClientManager(config)
        self.cache = cache

    def process_images_in_text(self, text: str) -> str:
        """Обрабатывает все изображения в тексте, заменяя ссылки на описания"""
        pattern = r'\[IMAGE:([^\]]+)\]([\s\S]*?)(?=\n\n|\[IMAGE:|$)'

        for match in re.finditer(pattern, text):
            img_path = match.group(1).strip()
            caption = match.group(2).strip()

            description = self.get_image_description(img_path, caption)

            original_block = match.group(0)
            text = text.replace(original_block, f"\n\nОписание изображения {caption}:\n{description}\n\n", 1)

        return text

    def get_image_description(self, img_path: str, caption: str) -> str:
        """Анализирует изображение с использованием контекстной подписи"""
        # Проверяем существование файла
        if not os.path.exists(img_path):
            print(f"Файл изображения не найден: {img_path}")
            return f"[Изображение недоступно: {img_path}]"

        # Проверка кэша
        cache_key = "img:"+self.cache.generate_hash(self._get_file_hash(img_path), caption,
                                             self.llm_client.get_model_name(model_type=ModelType.GRAPHIC))

        if self.cache.exists(cache_key):
            return self.cache.get(cache_key)

        # Подготовка изображения
        base64_image = self._encode_image(img_path)

        # Формируем промпт с использованием подписи
        prompt = f"""
        Ты анализируешь скриншот интерфейса для технической документации.
            Контекстная подпись: "{caption}"

            === Инструкции ===
            1. Обязательно выдели ВСЕ функциональные элементы:
               - Кнопки, поля ввода, переключатели
               - Меню, формы, интерактивные области
               - Элементы навигации и управления

            2. Для не-функциональных элементов (лого, декорации):
               - Фиксируем только факт наличия
               - Не анализируем содержание

            3. Формат описания каждого элемента:
               [тип] [расположение] [краткое описание] [соответствие]
               • Тип: button/field/logo/menu/icon/text
               • Расположение: top/bottom/left/right/center
               • Описание: 2-3 слова для функциональных элементов
               • Соответствие: match/extras/not_mentioned

            4. Критерии противоречий (если есть → вернуть CONFLICT):
               - Отсутствует ключевой функциональный элемент из подписи
               - Найдено несовместимое функциональное отличие
               - Более 1 критичного несоответствия

            5. Требования к выводу:
                - Строго в указанном формате без пояснений
                - Только фактические данные об элементах
                - Никаких вступлений/заключений
            === Пример вывода ===
            [кнопка] [низ] [Вход] [соответствует]
            [поле] [центр] [Введите логин] [соответствует] 
            [лого] [верх] [не упомянут]
            [иконка] [право] [глаз для показа пароля] [дополняет]
        """

        description = self.llm_client.call_graphic_llm(base64_image, prompt)
        self.cache.set(cache_key, description)

        return description

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Кодирует изображение в base64"""
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            with BytesIO() as buffer:
                img.save(buffer, format="JPEG", quality=85)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _get_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """Получить хеш файла с выбранным алгоритмом"""
        hash_algo = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)
        return hash_algo.hexdigest()
