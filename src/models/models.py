"""Model registry: maps the ``model_id`` strings used in configs to LLM client instances.

Credentials are read from the environment (see ``.env.template``):

* OpenAI models (``gpt-4o``, ``gpt-4o-mini``, ``gpt-4.1``, ...) use ``OPENAI_API_KEY`` / ``OPENAI_API_BASE``.
* Anthropic models use ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_API_BASE``.
* Google models (``gemini-2.5-pro``) use ``GOOGLE_API_KEY`` (or application-default credentials).
* Any OpenAI-compatible endpoint (e.g. a local vLLM server) can be added with ``register_model``.
"""

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.logger import logger
from src.models.litellm import LiteLLMModel
from src.utils import Singleton

load_dotenv(verbose=True)

CUSTOM_ROLE_CONVERSIONS = {"tool-call": "assistant", "tool-response": "user"}

# (registered name, provider model id)
OPENAI_MODELS = [
    ("gpt-4o", "gpt-4o"),
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("gpt-4.1", "gpt-4.1"),
    ("o3", "o3"),
    ("gpt-4o-search-preview", "gpt-4o-search-preview"),
]
ANTHROPIC_MODELS = [
    ("claude37-sonnet", "claude-3-7-sonnet-20250219"),
]
GOOGLE_MODELS = [
    ("gemini-2.5-pro", "gemini-2.5-pro"),
]


class ModelManager(metaclass=Singleton):
    def __init__(self):
        self.registed_models: Dict[str, Any] = {}

    def init_models(self, **kwargs) -> None:
        """Register all built-in models (clients are created lazily, keys are checked at call time)."""
        self._register_group(OPENAI_MODELS, api_key_env="OPENAI_API_KEY", api_base_env="OPENAI_API_BASE")
        self._register_group(ANTHROPIC_MODELS, api_key_env="ANTHROPIC_API_KEY", api_base_env="ANTHROPIC_API_BASE")
        self._register_group(GOOGLE_MODELS, api_key_env="GOOGLE_API_KEY", api_base_env=None)
        # Optional OpenAI-compatible endpoint, e.g. a local vLLM server (served model name in LOCAL_MODEL_ID).
        if os.getenv("LOCAL_MODEL_ID") and os.getenv("LOCAL_API_BASE"):
            self.register_model(
                name=os.getenv("LOCAL_MODEL_NAME", os.getenv("LOCAL_MODEL_ID")),
                model_id=f"openai/{os.getenv('LOCAL_MODEL_ID')}",
                api_base=os.getenv("LOCAL_API_BASE"),
                api_key=os.getenv("LOCAL_API_KEY", "EMPTY"),
            )

    def _register_group(self, models, api_key_env: str, api_base_env: Optional[str]) -> None:
        api_key = os.getenv(api_key_env)
        api_base = os.getenv(api_base_env) if api_base_env else None
        if not api_key:
            logger.debug(f"{api_key_env} is not set; models {[m[0] for m in models]} will fail when called.")
        for name, model_id in models:
            self.register_model(name=name, model_id=model_id, api_key=api_key, api_base=api_base)

    def register_model(
        self, name: str, model_id: str, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs
    ) -> None:
        """Register a LiteLLM-backed model under ``name`` (the ``model_id`` used in configs)."""
        model_kwargs = dict(model_id=model_id, custom_role_conversions=CUSTOM_ROLE_CONVERSIONS, **kwargs)
        if api_key:
            model_kwargs["api_key"] = api_key
        if api_base:
            model_kwargs["api_base"] = api_base
        self.registed_models[name] = LiteLLMModel(**model_kwargs)
