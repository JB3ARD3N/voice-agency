"""
LLM Router for Voice Agency
Handles routing between XAI (Grok) and Groq LLM providers with intelligent fallback.
"""

import os
import requests
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    """Available LLM providers."""
    XAI = "xai"
    GROQ = "groq"
    AUTO = "auto"


class Message(BaseModel):
    """Message format for LLM conversations."""
    role: str
    content: str


class LLMRequest(BaseModel):
    """Request model for LLM calls."""
    messages: List[Message]
    provider: LLMProvider = LLMProvider.AUTO
    temperature: float = 0.7
    max_tokens: int = 1024


class LLMResponse(BaseModel):
    """Response model from LLM."""
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None


class LLMRouter:
    """
    Intelligent LLM router that handles requests across multiple providers.
    Supports automatic fallback and load balancing.
    """

    def __init__(self):
        self.xai_api_key = os.getenv("XAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # XAI configuration
        self.xai_endpoint = "https://api.x.ai/v1/chat/completions"
        self.xai_model = "grok-beta"

        # Groq configuration
        self.groq_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_model = "mixtral-8x7b-32768"

        # Provider availability
        self.providers_available = {
            LLMProvider.XAI: bool(self.xai_api_key and self.xai_api_key != "your_key"),
            LLMProvider.GROQ: bool(self.groq_api_key and self.groq_api_key != "your_key")
        }

    def _call_xai(self, messages: List[Dict], temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Call XAI (Grok) API."""
        headers = {
            "Authorization": f"Bearer {self.xai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.xai_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(
            self.xai_endpoint,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def _call_groq(self, messages: List[Dict], temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Call Groq API."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(
            self.groq_endpoint,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def _select_provider(self, preferred: LLMProvider) -> LLMProvider:
        """
        Select an available provider based on preference.
        Falls back to available provider if preferred is unavailable.
        """
        if preferred == LLMProvider.AUTO:
            # Default preference order: XAI -> Groq
            if self.providers_available[LLMProvider.XAI]:
                return LLMProvider.XAI
            elif self.providers_available[LLMProvider.GROQ]:
                return LLMProvider.GROQ
            else:
                raise ValueError("No LLM providers are available. Please check API keys.")

        if self.providers_available[preferred]:
            return preferred

        # Fallback logic
        for provider, available in self.providers_available.items():
            if available and provider != preferred:
                print(f"Warning: {preferred} unavailable, falling back to {provider}")
                return provider

        raise ValueError(f"Provider {preferred} is not available and no fallback found.")

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Execute LLM completion with intelligent routing and fallback.

        Args:
            request: LLMRequest containing messages and configuration

        Returns:
            LLMResponse with completion and metadata
        """
        # Convert Pydantic messages to dicts
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Select provider
        provider = self._select_provider(request.provider)

        # Try primary provider
        try:
            if provider == LLMProvider.XAI:
                result = self._call_xai(messages, request.temperature, request.max_tokens)
                return LLMResponse(
                    content=result["choices"][0]["message"]["content"],
                    provider="xai",
                    model=self.xai_model,
                    usage=result.get("usage")
                )
            elif provider == LLMProvider.GROQ:
                result = self._call_groq(messages, request.temperature, request.max_tokens)
                return LLMResponse(
                    content=result["choices"][0]["message"]["content"],
                    provider="groq",
                    model=self.groq_model,
                    usage=result.get("usage")
                )
        except Exception as e:
            print(f"Error with {provider}: {e}")

            # Try fallback provider
            for fallback_provider, available in self.providers_available.items():
                if available and fallback_provider != provider:
                    print(f"Attempting fallback to {fallback_provider}")
                    try:
                        if fallback_provider == LLMProvider.XAI:
                            result = self._call_xai(messages, request.temperature, request.max_tokens)
                            return LLMResponse(
                                content=result["choices"][0]["message"]["content"],
                                provider="xai",
                                model=self.xai_model,
                                usage=result.get("usage")
                            )
                        elif fallback_provider == LLMProvider.GROQ:
                            result = self._call_groq(messages, request.temperature, request.max_tokens)
                            return LLMResponse(
                                content=result["choices"][0]["message"]["content"],
                                provider="groq",
                                model=self.groq_model,
                                usage=result.get("usage")
                            )
                    except Exception as fallback_error:
                        print(f"Fallback to {fallback_provider} also failed: {fallback_error}")

            raise Exception("All LLM providers failed")

    def check_health(self) -> Dict[str, bool]:
        """Check health status of all providers."""
        return self.providers_available.copy()


# Singleton instance
_router_instance = None


def get_router() -> LLMRouter:
    """Get or create singleton LLM router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance
