import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from LLMClientManager import LLMClientManager

class ResultProcessor:
    def __init__(self, llm_manager: "LLMClientManager"):
        self.llm_manager = llm_manager

    def extract_json(self, raw_response: str) -> dict | None:
        """
        Extracts JSON from the LLM response, cleaning up common artifacts.
        Returns a dictionary if successful, None otherwise.
        """
        json_str = self._find_json_string(raw_response)
        if not json_str:
            return None

        try:
            return self._parse_cleaned_json(json_str)
        except json.JSONDecodeError as e:
            logging.warning(f"Initial JSON parsing failed: {e}. The calling method should handle fixing.")
            return None

    def _find_json_string(self, raw_response: str) -> str:
        """Finds the most likely JSON string in the raw text."""
        # 1. Look for a <json> tag first
        json_match = re.search(r"<json>\s*(\{[\s\S]*?\})\s*<\/json>", raw_response, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # 2. Look for a markdown block as a fallback
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_response, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # 3. If not found, find the main JSON object
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            return json_match.group(0)

        logging.error(f"JSON object not found in response: {raw_response}")
        return ""

    def _parse_cleaned_json(self, json_str: str) -> dict:
        """Cleans and parses a JSON string."""
        # Clean comments and trailing commas
        json_str_cleaned = re.sub(r"//.*", "", json_str)
        json_str_cleaned = re.sub(r",\s*([\}\]])", r"\1", json_str_cleaned)
        return json.loads(json_str_cleaned)
