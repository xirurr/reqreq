import hashlib
import json
import logging
import pickle
import struct
from typing import Any, Optional, Union, List, Dict

import redis


class CacheService:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 password: Optional[str] = None, decode_responses: bool = True):
        """
        Инициализация клиента кеша Redis

        :param host: Хост Redis
        :param port: Порт Redis
        :param db: Номер базы данных Redis
        :param password: Пароль для Redis
        :param decode_responses: Декодировать ответы в строки
        """
        self.logger = logging.getLogger(__name__)

        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Проверяем соединение
            self.redis_client.ping()
            self.logger.info(f"Подключение к Redis успешно: {host}:{port}")
        except Exception as e:
            self.logger.error(f"Ошибка подключения к Redis: {str(e)}")
            raise ConnectionError(f"Не удалось подключиться к Redis: {str(e)}")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Записать данные в кеш по ключу

        :param key: Ключ для сохранения
        :param value: Значение для сохранения (будет сериализовано в JSON)
        :param ttl: Время жизни в секундах (опционально)
        :return: True если успешно, False если ошибка
        """
        try:
            # Сериализуем значение в JSON
            serialized_value = json.dumps(value, ensure_ascii=False)

            if ttl:
                result = self.redis_client.setex(key, ttl, serialized_value)
            else:
                result = self.redis_client.set(key, serialized_value)

            self.logger.debug(f"Данные записаны в кеш с ключом: {key}")
            return bool(result)

        except Exception as e:
            self.logger.error(f"Ошибка записи в кеш для ключа {key}: {str(e)}")
            return False

    import pickle

    def generate_hash(self, *args) -> str:
        """
        Генерирует детерминированный хеш SHA-256 от любых переданных объектов с помощью pickle.

        :param args: Любое количество объектов для хеширования.
        :return: Строка с хешем SHA-256.
        """
        try:
            # pickle.dumps сериализует практически любой объект Python в байтовую строку.
            # Используем самый высокий протокол для лучшей эффективности и стабильности.
            serialized_data = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)

            # Вычисляем хеш от сериализованной байтовой строки
            hash_object = hashlib.sha256(serialized_data)
            return hash_object.hexdigest()

        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при генерации хеша с помощью pickle: {e}")
            raise ValueError(f"Не удалось сгенерировать хеш: {e}")

    def exists(self, key: str) -> bool:
        """
        Проверить, есть ли данные в кеше по ключу

        :param key: Ключ для проверки
        :return: True если ключ существует, False если нет
        """
        try:
            result = self.redis_client.exists(key)
            self.logger.debug(f"Проверка существования ключа {key}: {bool(result)}")
            return bool(result)

        except Exception as e:
            self.logger.error(f"Ошибка проверки существования ключа {key}: {str(e)}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Получить данные по ключу из кеша

        :param key: Ключ для получения данных
        :return: Десериализованные данные или None если не найдено
        """
        try:
            cached_data = self.redis_client.get(key)

            if cached_data is None:
                self.logger.debug(f"Данные не найдены для ключа: {key}")
                return None

            # Десериализуем из JSON
            result = json.loads(cached_data)
            self.logger.debug(f"Данные получены из кеша для ключа: {key}")
            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка десериализации данных для ключа {key}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Ошибка получения данных для ключа {key}: {str(e)}")
            return None

    def delete(self, key: str) -> bool:
        """
        Удалить данные по ключу

        :param key: Ключ для удаления
        :return: True если удалено, False если ошибка или ключ не существовал
        """
        try:
            result = self.redis_client.delete(key)
            self.logger.debug(f"Удаление ключа {key}: {bool(result)}")
            return bool(result)

        except Exception as e:
            self.logger.error(f"Ошибка удаления ключа {key}: {str(e)}")
            return False

    def ping(self) -> bool:
        """
        Проверить соединение с Redis

        :return: True если соединение активно, False если нет
        """
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка проверки соединения: {str(e)}")
            return False

    def flush_db(self) -> bool:
        """
        Очистить всю текущую базу данных Redis

        :return: True если успешно, False если ошибка
        """
        try:
            self.redis_client.flushdb()
            self.logger.info("База данных Redis очищена")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка очистки базы данных: {str(e)}")
            return False