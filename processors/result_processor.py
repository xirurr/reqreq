import json
import re

class ResultProcessor:
    @staticmethod
    def extract_json(raw_response: str) -> dict:
        """
        Извлекает JSON из ответа LLM, очищая от возможных артефактов,
        включая markdown-блоки и <think> блоки.
        """
        # 1. Сначала ищем явный markdown-блок
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_response, re.DOTALL)
        try:
            if json_match:
                json_str = json_match.group(1)
            else:
                # 2. Если не найдено, удаляем <think> блоки и ищем JSON после них
                cleaned_response = re.sub(r"<think>[\s\S]*?<\/think>", "", raw_response).strip()

                # Ищем JSON объект (начиная с '{' и заканчивая '}')
                json_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
                if not json_match:
                    print(f"--- ОШИБКА: JSON не найден в ответе после очистки. Исходный ответ: ---\n{raw_response}\n-----------------")
                    return {}
                json_str = json_match.group(0)


            # 3. Очистка от комментариев и парсинг
            json_str_cleaned = re.sub(r"\/\/.*", "", json_str)
            # Дополнительная очистка от возможных запятых в конце
            json_str_cleaned = re.sub(r",\s*([\}\]])", r"\1", json_str_cleaned)
            return json.loads(json_str_cleaned)
        except json.JSONDecodeError as e:
            print(f"--- ОШИБКА: Не удалось декодировать JSON. Ошибка: {e} ---")
            print(f"--- Проблемная JSON-строка: ---\n{json_str}\n-----------------")
            return {}