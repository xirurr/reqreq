import json
import re
from typing import TYPE_CHECKING

from prompts.promts import FIX_JSON_PROMPT

if TYPE_CHECKING:
    from LLMClientManager import LLMClientManager


class ResultProcessor:
    def __init__(self, llm_manager: "LLMClientManager"):
        self.llm_manager = llm_manager

    def extract_json(self, raw_response: str) -> dict:
        """
        Extracts JSON from the LLM response, cleaning up common artifacts.
        If parsing fails, it attempts to fix the JSON using another LLM call.
        """
        json_str = self._find_json_string(raw_response)
        if not json_str:
            return {}

        try:
            # First attempt to parse the cleaned string
            return self._parse_cleaned_json(json_str)
        except json.JSONDecodeError as e:
            print(f"--- WARNING: Failed to decode JSON. Attempting to fix with LLM. Error: {e} ---")
            # If parsing fails, try to fix it with an LLM call
            return self._fix_json_with_llm(json_str)

    def _pre_clean_json_string(self, json_str: str) -> str:
        """Applies a series of regex fixes for common LLM-induced JSON errors."""
        # 1. Replace "::" with ":"
        cleaned_str = re.sub(r'"::"', r'":"', json_str)
        
        # 2. Remove newlines within string values
        # This is complex, so we do it carefully. This looks for a quote, then any characters
        # that are not a quote, then a newline, and replaces the newline with a space.
        # It's not perfect but can fix many common cases.
        cleaned_str = re.sub(r'("[^"\n]*)\n(["^"\n]*)', r'\1 \2', cleaned_str)

        # 3. Fix boolean values that might be in quotes
        cleaned_str = re.sub(r'"true"', 'true', cleaned_str)
        cleaned_str = re.sub(r'"false"', 'false', cleaned_str)

        return cleaned_str

    def _find_json_string(self, raw_response: str) -> str:
        """Finds the most likely JSON string in the raw text."""
        # 1. Look for a <json> tag first
        json_match = re.search(r"<json>\s*(\{[\s\S]*?\})\s*<\/json>", raw_response, re.DOTALL)
        if json_match:
            return self._pre_clean_json_string(json_match.group(1))

        # 2. Look for a markdown block as a fallback
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_response, re.DOTALL)
        if json_match:
            return self._pre_clean_json_string(json_match.group(1))

        # 3. If not found, remove <think> blocks and find the main JSON object
        cleaned_response = re.sub(r"<think>[\s\S]*?<\/think>", "", raw_response).strip()
        json_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
        if json_match:
            return self._pre_clean_json_string(json_match.group(0))

        print(f"--- ERROR: JSON not found in response after cleaning. Original response: ---\n{raw_response}\n-----------------")
        return ""

    def _parse_cleaned_json(self, json_str: str) -> dict:
        """Cleans and parses a JSON string."""
        # Clean comments and trailing commas
        json_str_cleaned = re.sub(r"//.*", "", json_str)
        json_str_cleaned = re.sub(r",\s*([\}\]])", r"\1", json_str_cleaned)
        return json.loads(json_str_cleaned)

    def _fix_json_with_llm(self, broken_json_str: str) -> dict:
        """Uses an LLM call to attempt to fix a broken JSON string."""
        # Safeguard: Don't try to fix hopelessly broken or very short strings.
        if len(broken_json_str) < 10 or '{' not in broken_json_str:
            print("--- FATAL ERROR: JSON string is too short or invalid to be fixed.")
            return {}

        print("  -> Sending to LLM for automated repair...")
        prompt = FIX_JSON_PROMPT.format(broken_json=broken_json_str)
        
        fixed_json_str = ""  # Initialize to prevent UnboundLocalError
        try:
            messages = [{"role": "user", "content": prompt}]
            fixed_response = self.llm_manager.call_text_llm(messages)
            
            fixed_json_str = self._find_json_string(fixed_response)
            if not fixed_json_str:
                 print(f"--- ERROR: LLM fixer did not return a JSON object. Response: ---\n{fixed_response}\n-----------------")
                 return {}

            return self._parse_cleaned_json(fixed_json_str)
        except json.JSONDecodeError as e:
            print(f"--- FATAL ERROR: Failed to decode JSON even after LLM fix. Error: {e} ---\nProblematic string: {fixed_json_str}")
            return {}
        except Exception as e:
            print(f"--- FATAL ERROR: An unexpected error occurred during JSON fixing: {e} ---")
            return {}