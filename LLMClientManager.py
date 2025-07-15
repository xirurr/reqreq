from typing import List

from openai import OpenAI

from models.model_types import ModelType


class LLMClientManager:
    def __init__(self, config: dict):
        self.config = config
        self.text_client = self._initialize_client(config["api_base"])
        image_api_base = config.get("image_api_base", config["api_base"])
        embedding_api_base = config.get("embedding_api_base", config["api_base"])
        self.image_client = self._initialize_client(image_api_base)
        self.embedding_client = self._initialize_client(embedding_api_base)

        self.model_mapping = {
            ModelType.DEFAULT_TEXT: self.config.get("model"),
            ModelType.SMART_TEXT: self.config.get("smart_model", self.config.get("model")), # Fallback to default
            ModelType.GRAPHIC: self.config.get("image_model"),
            ModelType.EMBEDDING: self.config.get("embedding_model")
        }

    def _initialize_client(self, api_base: str):
        return OpenAI(
            base_url=api_base,
            api_key=self.config.get("api_key", "not-needed"),
            timeout=self.config.get("timeout", 500.0)
        )

    def get_model_name(self, model_type: ModelType) -> str:
        model_name = self.model_mapping.get(model_type)
        if not model_name:
            raise ValueError(f"Model configuration not found for type {model_type}")
        return model_name

    def call_text_llm(self, messages: List[dict], model_type: ModelType = ModelType.DEFAULT_TEXT) -> str:
        """
        Вызов LLM для текстового анализа с использованием предоставленных сообщений.
        :param messages: Список сообщений для отправки в LLM
        :param model_type: Тип модели для использования (ModelType.DEFAULT_TEXT или ModelType.SMART_TEXT)
        """
        model_name = self.get_model_name(model_type)
        return self.text_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=self.config.get("temperature", 0.1),
            stop=["<|end_of_text|>", "<|im_end|>"],
            stream=False
        ).choices[0].message.content

    def call_graphic_llm(self, base64_image: str, prompt: str) -> str:
        """Вызов мультимодальной LLM для анализа изображений"""
        model_name = self.get_model_name(ModelType.GRAPHIC)
        return self.image_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.config.get("image_max_tokens"),
            temperature=0.1
        ).choices[0].message.content

    def call_embedding_llm(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для текста через LLM API.
        """
        model_name = self.get_model_name(ModelType.EMBEDDING)
        response = self.text_client.embeddings.create(
            input=text,
            model=model_name
        )
        return response.data[0].embedding

    def get_embedding_dim(self) -> int:
        return 1024
