"""Handles text generation using LLMs."""

import time
import re
import json
from typing import Dict, Optional, Any
from datetime import datetime, timezone


import ollama

from .utils import resize_and_encode_image


model_confidence = {
    "qwen2.5vl:7b": 0.85,
}


class ClothingTextAnalyzer:
    """Base class combining prompt generation and response parsing for clothing text analysis."""

    def __init__(self, clothing_categories: Optional[Dict] = None):
        self.clothing_categories = clothing_categories or {}

    def create_system_prompt(self, **kwargs) -> str:
        raise NotImplementedError("Must implement create_system_prompt")

    def create_user_prompt(self, **kwargs) -> str:
        raise NotImplementedError("Must implement create_user_prompt")

    def parse_response(self, raw_content: str) -> Any:
        raise NotImplementedError("Must implement parse_response")

    def is_valid_response(self, parsed_data: Any) -> bool:
        raise NotImplementedError("Must implement is_valid_response")


class DescriptionAnalyzer(ClothingTextAnalyzer):
    """Handles description-only analysis."""

    def create_system_prompt(self, **kwargs) -> str:
        """Simple description-only system prompt for multiple clothing items."""

        return """You are a clothing description AI. Provide only the description text, no JSON or formatting.

        Your task is to provide concise, accurate descriptions of ALL clothing items visible in images.

        RULES:
        - Return ONLY the description text, nothing else
        - If multiple clothing items are visible, describe each one separately
        - Use clear tags to separate items: [ITEM 1], [ITEM 2], etc.
        - Focus on key visual elements: type, color, style, material, fit
        - Be objective and descriptive
        - Use clear, simple language
        - No JSON, no formatting, just plain description text

        Examples:
        Single item: "Blue denim jeans with straight leg cut and classic five-pocket design"

        Multiple items: "[ITEM 1] White cotton button-up shirt with long sleeves and classic collar [ITEM 2] Dark blue skinny jeans with mid-rise waist [ITEM 3] Black leather ankle boots with pointed toe" """

    def create_user_prompt(
        self, additional_context: Optional[str] = None, **kwargs
    ) -> str:
        """Simple user prompt for description only."""

        context_text = f" {additional_context}" if additional_context else ""
        return f"Analyze this clothing image and provide a concise description.{context_text}."

    def parse_response(self, raw_content: str) -> str:
        return raw_content.strip()

    def is_valid_response(self, parsed_data: str) -> bool:
        return bool(parsed_data and len(parsed_data) > 10)


class CategorizedAnalyzer(ClothingTextAnalyzer):
    """Handles categorized analysis with JSON output."""

    def __init__(self, clothing_categories: Dict):
        super().__init__(clothing_categories)

    def create_system_prompt(self, **kwargs) -> str:
        """System prompt for categorized analysis with proper category enforcement."""

        if not self.clothing_categories:
            raise ValueError(
                "clothing_categories must be provided for categorized analysis"
            )

        prompt = """You are a clothing analysis AI. Return ONLY valid JSON, no other text.

        Your task is to analyze ALL clothing items visible in the image and categorize them using predefined tags.

        AVAILABLE CATEGORIES AND TAGS:
        """

        # Add each category with its exact allowed values
        for category, items in self.clothing_categories.items():
            if isinstance(items, (set, list)):
                items_str = ", ".join(f'"{item}"' for item in sorted(items))
                prompt += f"{category}: [{items_str}]\n"

        prompt += f"""
        REQUIRED JSON FORMAT:
        {{
        "{list(self.clothing_categories.keys())[0]}": ["tag1", "tag2"],
        "{list(self.clothing_categories.keys())[1] if len(self.clothing_categories) > 1 else "category2"}": [],
        "description": "your description here"
        }}

        DESCRIPTION FIELD RULES:
        - ALWAYS include a "description" field as a string
        - Single item: "Blue denim jeans with straight leg cut and classic five-pocket design"
        - Multiple items: "[ITEM 1] White cotton button-up shirt with long sleeves [ITEM 2] Dark blue skinny jeans with mid-rise waist"
        - Focus on: type, color, style, material, fit

        STRICT REQUIREMENTS:
        - Use ONLY tags from the categories above
        - Every category must be present (use [] if empty)
        - The "description" field is MANDATORY
        - Return valid JSON only, no other text
        """

        return prompt

    def parse_response(self, raw_content: str) -> Dict:
        """Parse JSON response and normalize to lowercase."""
        try:
            match = re.search(r"(\{.*\})", raw_content, re.DOTALL)
            if not match:
                raise ValueError("JSON block not found")

            json_str = match.group(1).replace('[""]', "[]").replace('["]', "[]").lower()
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            return {}

    def is_valid_response(self, parsed_data: Dict) -> bool:
        """Check if categorized result is valid and fill missing categories with empty lists."""
        if not parsed_data:
            return False

        # Description field must always be present and valid
        if "description" not in parsed_data:
            return False

        description = parsed_data.get("description", "")
        if not description or len(description) < 10:
            return False

        valid_categories_lower = {
            cat.lower() for cat in self.clothing_categories.keys()
        }
        existing_keys_lower = {k.lower() for k in parsed_data.keys()}

        # Reject unexpected category keys
        for key in parsed_data.keys():
            if key == "description":
                continue
            if key.lower() not in valid_categories_lower:
                return False

        # Add missing categories with empty lists
        for category in self.clothing_categories.keys():
            cat_lower = category.lower()
            if cat_lower not in existing_keys_lower:
                parsed_data[cat_lower] = []

        return True

    def create_user_prompt(self, **kwargs) -> str:
        """User prompt for categorized analysis."""
        return """Analyze this clothing image. Return JSON with all categories as lists and a "description" field. Example: {"category1": ["tag"], "category2": [], "description": "your description"}"""

    def get_description(self, parsed_data: Dict) -> str:
        """Extract description from parsed data."""
        return parsed_data.get("description", "")

    def get_categories(self, parsed_data: Dict) -> Dict:
        """Extract category data without description."""
        return {k: v for k, v in parsed_data.items() if k != "description"}


def call_vision_model(
    model_name: str, encoded_image: str, system_prompt: str, user_prompt: str
) -> dict:
    """Call vision model through ollama with image and prompts.

    Args:
        model_name: Name of the LLM model to use
        encoded_image: Base64-encoded string of the image to analyze
        system_prompt: System prompt to guide the LLM's behavior
        user_prompt: User prompt with specific instructions or questions

    Returns:
        Dictionary with keys: content (str), duration (float), success (bool), error (str)
    """
    start_time = time.time()

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [encoded_image]},
            ],
        )

        return {
            "content": response.message.content,
            "duration": time.time() - start_time,
            "success": True,
        }

    except Exception as e:
        return {
            "content": None,
            "duration": time.time() - start_time,
            "error": str(e),
            "success": False,
        }


def analyze_clothing_image(
    image_path: str,
    prompt_type: str = "description_only",
    model_name: str = "qwen2.5vl:7b",
    clothing_categories: Optional[Dict] = None,
    max_width: int = 256,
    max_retries: int = 10,
) -> dict:
    """Complete clothing image analysis pipeline with multiple retries on invalid result."""

    try:
        # Create appropriate analyzer
        if prompt_type == "description_only":
            analyzer = DescriptionAnalyzer()
        elif prompt_type == "categorized":
            if not clothing_categories:
                raise ValueError(
                    "clothing_categories required for categorized analysis"
                )
            analyzer = CategorizedAnalyzer(clothing_categories)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # Prepare image encoding once
        encoded_image = resize_and_encode_image(image_path, max_width=max_width)

        # Initial prompts
        system_prompt = analyzer.create_system_prompt()
        user_prompt = analyzer.create_user_prompt()

        total_duration = 0
        raw_content = ""
        parsed_data = None
        is_valid = False

        for attempt in range(max_retries):
            if attempt == 0:
                current_user_prompt = user_prompt
            else:
                current_user_prompt = (
                    f"Previous response was invalid or incomplete. Please analyze the image and the previous output below, "
                    f"correct any errors, and provide a valid {prompt_type}.\n"
                    f"Previous output:\n{raw_content}"
                )

            model_result = call_vision_model(
                model_name, encoded_image, system_prompt, current_user_prompt
            )

            raw_content = model_result.get("content", "")
            duration = model_result.get("duration", 0)
            total_duration += duration

            parsed_data = analyzer.parse_response(raw_content)
            is_valid = analyzer.is_valid_response(parsed_data)

            if is_valid:
                break

        success = model_result.get("success", False) and is_valid

        return {
            "parsed_data": parsed_data,
            "raw_content": raw_content,
            "is_valid": is_valid,
            "success": success,
            "duration": total_duration,
            "image_path": image_path,
            "prompt_type": prompt_type,
            "model_name": model_name,
            "analyzer": analyzer,
        }

    except Exception as e:
        return {
            "parsed_data": None,
            "raw_content": "",
            "is_valid": False,
            "success": False,
            "error": str(e),
            "image_path": image_path,
            "prompt_type": prompt_type,
            "model_name": model_name,
        }


def get_tags_from_analysis(analysis: dict) -> list:
    """Extract tags from analysis result for database storage."""
    common_data = {
        "model_name": analysis["model_name"],
        "confidence": model_confidence[analysis["model_name"]],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if analysis["prompt_type"] == "categorized":
        tags = [{**analysis["parsed_data"], **common_data}]
    elif analysis["prompt_type"] == "description_only":
        tags = [
            {
                "description": analysis["parsed_data"],
                **common_data,
            }
        ]
    else:
        tags = []

    return tags


def get_processing_status(analysis: dict) -> dict:
    """Extract processing status from analysis result for database storage."""
    return {analysis["model_name"]: analysis["is_valid"] and analysis["success"]}
