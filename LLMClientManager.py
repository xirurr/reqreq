from typing import List

from openai import OpenAI


class LLMClientManager:
    def __init__(self, config: dict):
        self.config = config
        self.text_client = self._initialize_client(config["api_base"])
        image_api_base = config.get("image_api_base", config["api_base"])
        embedding_api_base = config.get("embedding_api_base", config["api_base"])
        self.image_client = self._initialize_client(image_api_base)
        self.embedding_client = self._initialize_client(embedding_api_base)

    def _initialize_client(self, api_base: str):
        return OpenAI(
            base_url=api_base,
            api_key=self.config.get("api_key", "not-needed"),
            timeout=self.config.get("timeout", 500.0)
        )

    def call_text_llm(self, messages: List[dict]) -> str:
        """
        Вызов LLM для текстового анализа с использованием предоставленных сообщений.
        :param messages: Список сообщений для отправки в LLM
        """
        return self.text_client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            max_tokens=self.config.get("max_tokens"),
            temperature=self.config.get("temperature", 0.1),
            stop=["<|end_of_text|>", "<|im_end|>"],
            stream=False
        ).choices[0].message.content

    def call_graphic_llm(self, base64_image: str, prompt: str) -> str:
        """Вызов мультимодальной LLM для анализа изображений"""
        return self.image_client.chat.completions.create(
            model=self.config.get("image_model", self.config["model"]),
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
        Поддерживается только для моделей, которые умеют создавать векторы.
        """
        response = self.text_client.embeddings.create(
            input=text,
            model=self.config["embedding_model"]  # Убедитесь, что модель поддерживает эмбеддинги
        )
        return response.data[0].embedding

    def get_text_model_name(self) -> str:
        return self.config["model"]

    def get_image_model_name(self) -> str:
        return self.config["image_model"]

    def get_embedding_dim(self) -> int:
        return 1024
