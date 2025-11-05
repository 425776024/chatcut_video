import json
import os

from openai import OpenAI
from .config import LLM_MODEL_OPTIONS


class LLMProcessor:
    def __init__(self, llm_model: str, temperature: float):
        for model in LLM_MODEL_OPTIONS:
            if model['label'] == llm_model:
                self.api_key = os.getenv(model['api_key_env_name'])
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=model['base_url'],
                )
                self.model = model['model']
                self.temperature = temperature
                self.max_tokens = model.get('max_tokens', 4096)
                break

        if not hasattr(self, 'client'):
            raise ValueError(
                f"Unsupported LLM model: {llm_model}. Available models: "
                f"{', '.join([m['label'] for m in LLM_MODEL_OPTIONS])}"
            )