import json
import re


class ResultProcessor:
    @staticmethod
    def extract_json(raw_response: str) -> dict:
        """Извлекает JSON из ответа LLM"""
        json_match = re.search(r'\{[\s\S]*\}', raw_response)

        if not json_match:
            print("JSON не найден в ответе. Попытка исправления...")
            return ResultProcessor._retry_with_repair(raw_response)

        json_str = json_match.group(0)

        try:
            json_str = re.sub(r'//.*?\n', '', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Невалидный JSON: {str(e)}. Попытка исправления...")
            return ResultProcessor._retry_with_repair(json_str)

    @staticmethod
    def merge_results(results: dict, new_data: dict):
        """Объединяет результаты анализа"""
        for key in ["requirements", "system_states", "entities"]:
            if key in new_data:
                results[key].extend(new_data[key])

    @staticmethod
    def postprocess_results(data: dict) -> dict:
        """Очистка и дедупликация результатов"""
        unique_reqs = {}
        for req in data.get("requirements", []):
            if "id" in req and "text" in req:
                unique_reqs[req["id"]] = req

        unique_states = {}
        for state in data.get("system_states", []):
            if "id" in state and "name" in state:
                unique_states[state["id"]] = state

        unique_entities = {}
        for entity in data.get("entities", []):
            if "id" in entity and "name" in entity:
                unique_entities[entity["id"]] = entity

        return {
            "requirements": list(unique_reqs.values()),
            "system_states": list(unique_states.values()),
            "entities": list(unique_entities.values())
        }

    @staticmethod
    def _retry_with_repair(raw_response: str) -> dict:
        # Логика восстановления JSON
        return {"requirements": [], "system_states": [], "entities": []}