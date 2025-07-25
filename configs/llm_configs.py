lm_studio_config = {
    "api_base": "http://localhost:11434/v1",
    "smart_api_base": "http://localhost:1234/v1",
    "image_api_base": "http://localhost:11434/v1",
    "embedding_api_base": "http://localhost:1234/v1",
    "api_key": "null",  # LM Studio не требует реального ключа
    "model": "qwen3:4b",
    "smart_model": "qwen/qwen3-8b",
    "image_model": "gemma3:4b",
    "embedding_model": "text-embedding-mxbai-embed-large-v1",
    "image_directory": "/parsed/images",
    "max_tokens": 8000,
    "image_max_tokens": 1000,
    "temperature": 0.1,
    "chunk_size": 3000,
    "timeout": 900.0
}