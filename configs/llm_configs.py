lm_studio_config = {
    "api_base": "http://localhost:1234/v1",
    "smart_api_base": "http://localhost:1234/v1",
    "image_api_base": "http://localhost:11434/v1",
    "embedding_api_base": "http://localhost:1234/v1",
    "api_key": "any-string",  # LM Studio не требует реального ключа
    "model": "qwen/qwen3-4b",  # текстовая модель
    "smart_model": "qwen/qwen3-8b",  # текстовая модель
    "image_model": "gemma3:4b", # модель анализа изображений
    "embedding_model": "text-embedding-mxbai-embed-large-v1", # модель векторизации текста
    "image_directory": "/parsed/images",  # Директория с изображениями
    "max_tokens": 8000,
    "image_max_tokens": 1000,
    "temperature": 0.1,
    "chunk_size": 3000,
    "timeout": 120.0  # Увеличим таймаут для больших моделей
}