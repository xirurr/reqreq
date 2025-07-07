import hashlib
import json
import logging
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

    def generate_hash(self, *objects) -> str:
        """
        Сгенерировать хешсумму от переданных объектов

        :param objects: Объекты для хеширования
        :return: SHA-256 хеш в виде строки
        """
        try:
            # Преобразуем все объекты в строки и объединяем
            hash_input = ""
            for obj in objects:
                if isinstance(obj, (dict, list)):
                    # Для словарей и списков используем JSON с сортировкой ключей
                    hash_input += json.dumps(obj, sort_keys=True, ensure_ascii=False)
                else:
                    # Для остальных типов просто преобразуем в строку
                    hash_input += str(obj)
                hash_input += "|"  # Разделитель между объектами

            # Генерируем SHA-256 хеш
            hash_object = hashlib.sha256(hash_input.encode('utf-8'))
            return hash_object.hexdigest()

        except Exception as e:
            self.logger.error(f"Ошибка генерации хеша: {str(e)}")
            raise ValueError(f"Не удалось сгенерировать хеш: {str(e)}")

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