import json

class BasePrompt:
    """
    Класс-конструктор для системных промптов.
    """
    def __init__(self):
        self.role_description: str = "Ты — полезный AI ассистент."
        self.instructions: list[str] = []
        self.examples: str = ""  # Для полных текстовых примеров (few-shot)
        self.json_structure: dict | None = None # Для описания структуры JSON
        self.output_constraints: list[str] = [
            "Твой ответ должен быть ТОЛЬКО валидным JSON объектом.",
            "Оберни финальный JSON в теги `<json>` и `</json>`."
        ]
        self.context_data: dict = {}
        self.task_description: str = ""

    def _format_instructions(self) -> str:
        if not self.instructions:
            return ""
        lines = ["### ИНСТРУКЦИИ:"]
        for i, instruction in enumerate(self.instructions, 1):
            lines.append(f"{i}.  {instruction}")
        return "\n".join(lines)

    def _format_output_constraints(self) -> str:
        lines = ["### ФОРМАТ ВЫВОДА:"]
        lines.extend(self.output_constraints)
        if self.json_structure:
            structure_str = json.dumps(self.json_structure, ensure_ascii=False, indent=2)
            lines.append("\n### СТРУКТУРА JSON:")
            lines.append(f"```json\n{structure_str}\n```")
        return "\n".join(lines)

    def _format_context(self) -> str:
        if not self.context_data:
            return ""
        
        lines = []
        for key, value in self.context_data.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                value_str = str(value)
            lines.append(f"### {key.replace('_', ' ').upper()}:\n{value_str}")
        return "\n".join(lines)

    def _format_task(self) -> str:
        if not self.task_description:
            return ""
        return f"### {self.task_description.upper()}:\n"

    def generate(self) -> str:
        """Собирает финальный текст промпта из всех компонентов."""
        parts = [
            self.role_description,
            self._format_instructions(),
            self.examples,
            self._format_output_constraints(),
            self._format_context(),
            self._format_task()
        ]
        return "\n\n".join(filter(None, parts))