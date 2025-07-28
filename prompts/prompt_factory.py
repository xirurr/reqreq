import json
from .base_prompt import BasePrompt


class PromptFactory:
    """
    Фабрика для создания системных промптов через типизированные методы.
    """

    def get_entities_prompt(self, text: str, existing_entities: list) -> BasePrompt:
        return self._build_entities_prompt(text, existing_entities)

    def get_requirements_prompt(self, text: str, entities_context: dict) -> BasePrompt:
        return self._build_requirements_prompt(text, entities_context)

    def get_batch_dependency_prompt(self, req_a_id: str, req_a_text: str, candidates_list: str,
                                    relevant_entities: dict) -> BasePrompt:
        return self._build_batch_dependency_prompt(req_a_id, req_a_text, candidates_list, relevant_entities)

    def get_requirement_category_prompt(self, req_text: str, existing_categories: list) -> BasePrompt:
        return self._build_requirement_category_prompt(req_text, existing_categories)

    def get_ui_image_prompt(self) -> BasePrompt:
        return self._build_ui_image_prompt()

    def get_qna_prompt(self, question: str, context_graph: dict, enriched_text: str) -> BasePrompt:
        return self._build_qna_prompt(question, context_graph, enriched_text)

    def get_matching_prompt(self, old_reqs: list, new_reqs: list) -> BasePrompt:
        return self._build_matching_prompt(old_reqs, new_reqs)

    # ===================================================================
    #                       ПРИВАТНЫЕ МЕТОДЫ-СБОРЩИКИ
    # ===================================================================

    def _build_matching_prompt(self, old_reqs: list, new_reqs: list) -> BasePrompt:
        old_reqs_str = json.dumps([{'id': r['id'], 'text': r['text']} for r in old_reqs], ensure_ascii=False, indent=2)
        new_reqs_str = json.dumps([{'temp_id': r['temp_id'], 'text': r['text']} for r in new_reqs], ensure_ascii=False, indent=2)

        p = BasePrompt()
        p.role_description = "Твоя задача: Для каждого 'нового' требования найди семантически эквивалентное 'старое' требование."
        p.instructions = [
            "Верни JSON объект, где ключ - это 'temp_id' нового требования, а значение - это 'id' старого требования, которому оно соответствует.",
            "Если для нового требования нет соответствия среди старых, используй `null`."
        ]
        p.json_structure = {
            "TEMP_REQ-001": "REQ-AUTH-005",
            "TEMP_REQ-002": None
        }
        p.context_data = {
            "Старые требования": old_reqs_str,
            "Новые требования": new_reqs_str
        }
        p.task_description = "ТВОЙ ОТВЕТ (ТОЛЬКО JSON)"
        p.output_constraints = ["Верни ТОЛЬКО JSON объект без тегов."]
        return p

    # ===================================================================
    #                       ПРИВАТНЫЕ МЕТОДЫ-СБОРЩИКИ
    # ===================================================================

    def _build_entities_prompt(self, text: str, existing_entities: list) -> BasePrompt:
        p = BasePrompt()
        p.role_description = "Ты — системный аналитик. Твоя задача — извлечь из текста ключевые **сущности** системы, их **атрибуты** и **состояния**, согласуя их с уже существующим списком."
        p.instructions = [
            "Проверь существующие сущности: Прежде чем создавать новую сущность, проверь, нет ли в списке выше подходящего синонима. Если синоним найден, используй существующее имя из списка. Если сущность новая, создай новую, если она действительно уникальна.",
            "Фокус на сущностях: Ищи существительные, которые обозначают ключевые объекты системы (например, \"Пользователь\", \"Заказ\", \"Отчет\", \"Система\").",
            "Атрибуты и состояния: Для каждой сущности определи ее характеристики (атрибуты) и возможные состояния.",
            "Игнорируй UI и детали реализации: Пропускай описания интерфейса (кнопки, формы) и технические детали.",
            "Игнорируй нерелевантный текст: Пропускай оглавление, титулы, номера страниц, служебные пометки.",
        ]
        p.examples = """### ПРИМЕР:
        -   **Существующие сущности:** `["Пользователь", "Заказ"]`
        -   **Текст:** "Зарегистрированный пользователь может создать заявку. После создания заявка находится в статусе 'Новая'. У заявки должен быть уникальный номер и дата создания."
        -   **Результат:**
            <json>
            {
              "entities": [
                {
                  "name": "Пользователь",
                  "description": "Субъект, взаимодействующий с системой.",
                  "attributes": [
                    {"name": "статус регистрации", "type": "boolean", "description": "Зарегистрирован ли пользователь"}
                  ]
                },
                {
                  "name": "Заказ",
                  "description": "Заявка на покупку товара.",
                  "attributes": [
                    {"name": "номер", "type": "string"},
                    {"name": "дата создания", "type": "datetime"}
                  ],
                  "states": [
                    {"name": "Новый"},
                    {"name": "Оплачен"},
                    {"name": "Отменен"}
                  ]
                }
              ]
            }
            </json>
        """
        p.json_structure = {
            "entities": [
                {
                    "name": "ИмяСущности",
                    "description": "Описание.",
                    "attributes": [
                        {"name": "имя_атрибута", "type": "тип_данных"}
                    ],
                    "states": [
                        {"name": "имя_состояния"}
                    ]
                }
            ]
        }
        p.output_constraints.append("Если сущности не найдены, верни пустой JSON `{\"entities\": []}`.")
        p.context_data = {"СУЩЕСТВУЮЩИЕ СУЩНОСТИ": existing_entities, "ТЕКСТ ДЛЯ АНАЛИЗА": text}
        p.task_description = "ТВОЙ ОТВЕТ (ТОЛЬКО JSON В ТЕГАХ <json>)"
        return p

    def _build_requirements_prompt(self, text: str, entities_context: dict) -> BasePrompt:
        p = BasePrompt()
        p.role_description = "Ты — системный аналитик. Твоя задача — извлечь из текста **технические требования** к системе."
        p.instructions = [
            "Фокус на требованиях: Ищи фразы, которые описывают, что система **должна делать** (функциональные требования) или какими **свойствами обладать** (нефункциональные).",
            "Игнорируй UI: Пропускай описания интерфейса (\"кнопка\", \"форма\", \"уведомление\"). Вместо этого фиксируй суть операции.",
            "Игнорируй нерелевантный текст: Пропускай оглавление, титулы, номера страниц, служебные пометки.",
            "Условия и триггеры: Внимательно ищи слова-маркеры, указывающие на условия или последовательность действий. Если находишь такое слово или фразу, ОБЯЗАТЕЛЬНО добавь его в поле `trigger_words`.",
            "Контекст: Используй предоставленные **сущности и их состояния** для более точного описания условий."
        ]
        p.json_structure = {
            "requirements": [{
                "text": "Текст требования",
                "conditions": [{"condition": "условие"}],
                "trigger_words": ["триггерные слова"]
            }]
        }
        p.examples = """### ПРИМЕРЫ:
*   **Текст:** "После успешной авторизации, система должна отобразить личный кабинет."
*   **Результат:**
    <json>
    {
      "requirements": [
        {
          "text": "Система отображает личный кабинет.",
          "conditions": [{"condition": "user_status == 'authorized'"}],
          "trigger_words": ["После успешной авторизации"]
        }
      ]
    }
    </json>

*   **Текст:** "Отправка отчета возможна только для верифицированных пользователей."
*   **Результат:**
    <json>
    {
      "requirements": [
        {
          "text": "Система позволяет отправлять отчет.",
          "conditions": [{"condition": "user_verification_status == 'verified'"}],
          "trigger_words": ["только для"]
        }
      ]
    }
    </json>
"""
        p.output_constraints.append("Если требования не найдены, верни пустой JSON `{\"requirements\": []}`.")
        p.context_data = {"СУЩНОСТИ И СОСТОЯНИЯ (КОНТЕКСТ)": entities_context, "ТЕКСТ ДЛЯ АНАЛИЗА": text}
        p.task_description = "ТВОЙ ОТВЕТ (ТОЛЬКО JSON В ТЕГАХ <json>)"
        return p

    def _build_batch_dependency_prompt(self, req_a_id: str, req_a_text: str, candidates_list: str,
                                       relevant_entities: dict) -> BasePrompt:
        p = BasePrompt()
        p.role_description = "Ты — системный аналитик. Проанализируй **Требование А** и определи его связи с каждым требованием из **Списка Кандидатов**."
        p.instructions = [
            "Для КАЖДОГО кандидата из списка определи тип его связи с Требованием А.",
            "Типы связей: DEPENDS_ON, BLOCKS, RELATES_TO, DUPLICATES, NO_RELATION.",
            "Включай в список **только** те пары, между которыми есть любая связь, кроме 'NO_RELATION'."
        ]
        p.examples = """### ПРИМЕР:
*   **Требование А:** `ID: REQ-001`, `Текст: "Пользователь должен иметь возможность отменить заказ."`
*   **Список Кандидатов:**
    - `ID: REQ-005`, `Текст: "Система должна отправлять email-уведомление при отмене заказа."`
    - `ID: REQ-012`, `Текст: "Администратор может просматривать все заказы."`
*   **Результат:**
    <json>
    {
      "dependencies": [
        {"source": "REQ-005", "target": "REQ-001", "type": "DEPENDS_ON"},
        {"source": "REQ-012", "target": "REQ-001", "type": "RELATES_TO"}
      ]
    }
    </json>
"""
        p.json_structure = {
            "dependencies": [
                {"source": "REQ-001", "target": "REQ-005", "type": "DEPENDS_ON"}
            ]
        }
        p.context_data = {
            "Требование А": f"ID: {req_a_id}\nТекст: \"{req_a_text}\"",
            "Список Кандидатов для проверки": candidates_list,
            "Сущности, упоминаемые в этих требованиях (для контекста)": relevant_entities
        }
        p.task_description = "ТВОЙ ОТВЕТ (ТОЛЬКО JSON В ТЕГАХ <json>)"
        return p

    def _build_requirement_category_prompt(self, req_text: str, existing_categories: list[str]) -> BasePrompt:
        p = BasePrompt()
        p.role_description = "Проанализируй текст требования и придумай для него короткую категорию."
        p.instructions = [
            "Категория должна быть на английском языке, в верхнем регистре.",
            "Формат категории: `ГЛАГОЛ_СУЩНОСТЬ` (например, `AUTHENTICATE_USER`, `EXPORT_REPORT`).",
            "Проверь существующие категории. Если есть подходящая, используй ее. Если нет, создай новую, которая хорошо описывает суть требования."
        ]
        p.json_structure = {"category": "EXAMPLE_CATEGORY"}
        p.context_data = {
            "Текст требования": f"'{req_text}'",
            "Существующие категории": existing_categories
        }
        p.task_description = "ТВОЙ ОТВЕТ (ТОЛЬКО JSON)"
        p.output_constraints = ["Верни ТОЛЬКО JSON объект с одним ключом \"category\" без тегов."]
        return p

    def _build_ui_image_prompt(self) -> BasePrompt:
        p = BasePrompt()
        p.role_description = "Ты — AI, который преобразует изображения UI в структурированное JSON-описание."
        p.instructions = [
            "Определи основные компоненты: Найди на экране логические группы элементов (формы, таблицы, панели, шапки, модальные окна).",
            "Опиши каждый компонент: Для каждого компонента укажи его тип (`component_type`) и дочерние элементы (`children`).",
            "Детализируй элементы: Для интерактивных элементов (кнопки, поля ввода, чекбоксы) укажи их тип (`element_type`), видимый текст или `label`, и, если применимо, подтип (`subtype`)."
        ]
        p.json_structure = {
            "ui_components": [
                {
                    "component_type": "form",
                    "name": "Форма аутентификации",
                    "children": [
                        {"element_type": "input", "label": "Логин"},
                        {"element_type": "button", "text": "Войти"}
                    ]
                }
            ]
        }
        p.context_data = {"ИЗОБРАЖЕНИЕ ДЛЯ АНАЛИЗА": "предоставлено"}
        p.task_description = "ТВОЙ ОТВЕТ (ТОЛЬКО JSON В ТЕГАХ <json>)"
        return p

    def _build_qna_prompt(self, question: str, context_graph: dict, enriched_text: str) -> BasePrompt:
        p = BasePrompt()
        p.role_description = "Ты — ведущий системный аналитик, отвечающий на вопросы по базе знаний проекта. Твоя задача — дать точный и исчерпывающий ответ на вопрос пользователя, основываясь на предоставленных данных."
        p.instructions = [
            "Проанализируй все данные: Внимательно изучи структурированный граф и дополнительный текстовый контекст.",
            "Построй цепочку рассуждений: Мысленно, шаг за шагом, построй логическую цепочку от вопроса к ответу, используя связи из графа как основу.",
            "Сформулируй прямой ответ: На основе рассуждений дай четкий и прямой ответ на вопрос пользователя.",
            "Укажи источники: Перечисли ID ключевых узлов графа (требований или сущностей), которые были наиболее важны для твоего ответа.",
            "Оцени уверенность: В поле `confidence_score` поставь оценку от 0.0 до 1.0, насколько ты уверена, что предоставленный контекст позволил тебе дать точный и полный ответ на вопрос пользователя. Если контекст был нерелевантным или недостаточным, оценка должна быть низкой (например, 0.2)."
        ]
        p.json_structure = {
            "reasoning": "Здесь пошаговое объяснение, как ты пришел к выводу.",
            "answer": "Здесь прямой и чистый ответ для пользователя.",
            "source_ids": ["REQ-ID-001", "EntityName"],
            "confidence_score": 0.9
        }
        p.context_data = {
            "ВОПРОС ПОЛЬЗОВАТЕЛЯ": question,
            "СТРУКТУРИРОВАННЫЙ КОНТЕКСТ (ГРАФ ЗНАНИЙ)": context_graph,
            "ДОПОЛНИТЕЛЬНЫЙ ТЕКСТОВЫЙ КОНТЕКСТ (ИЗ ДОКУМЕНТОВ)": enriched_text
        }
        p.task_description = "ТВОЙ ОТВЕТ (ТОЛЬКО JSON В ТЕГАХ <json>)"
        return p
