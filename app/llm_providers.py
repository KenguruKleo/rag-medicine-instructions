"""
LLM Provider abstraction for supporting multiple LLM providers.

Supports:
- OpenAI (default)
- Anthropic (Claude)
- Google (Gemini)
- Microsoft Azure OpenAI
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel


def get_model_name(llm: BaseChatModel) -> str:
    """
    Extract a short model name from an LLM instance.
    
    Args:
        llm: BaseChatModel instance
        
    Returns:
        Short model name string (e.g., "gpt-5-nano", "claude-3-5-sonnet-20241022")
    """
    if isinstance(llm, ChatOpenAI):
        return getattr(llm, "model_name", "unknown")
    elif isinstance(llm, AzureChatOpenAI):
        return getattr(llm, "azure_deployment", "unknown")
    elif isinstance(llm, ChatAnthropic):
        return getattr(llm, "model", "unknown")
    elif isinstance(llm, ChatGoogleGenerativeAI):
        return getattr(llm, "model", "unknown")
    else:
        # Fallback: try to get model_name attribute
        return getattr(llm, "model_name", getattr(llm, "model", str(type(llm).__name__)))


def get_llm_provider() -> str:
    """Get the configured LLM provider from environment variables."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    # Check if Azure is configured (takes precedence)
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    return provider


def create_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    provider: Optional[str] = None,
) -> BaseChatModel:
    """
    Create an LLM instance based on the configured provider.
    
    Args:
        model_name: Model name to use (overrides default for provider)
        temperature: Temperature setting (if supported by model)
        provider: Provider name (overrides LLM_PROVIDER env var)
    
    Returns:
        BaseChatModel instance
    """
    provider = provider or get_llm_provider()
    
    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set")
        
        model = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=model,
            api_key=api_key,
            api_version=api_version,
            temperature=temperature or float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.7")),
        )
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        model = model_name or os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
        
        # Some OpenAI models don't support custom temperature
        if model in ["gpt-5-nano"]:
            return ChatOpenAI(
                model=model,
                api_key=api_key,
            )
        else:
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature or float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            )
    
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature or float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
        )
    
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        model = model_name or os.getenv("GOOGLE_MODEL", "gemini-pro")
        
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature or float(os.getenv("GOOGLE_TEMPERATURE", "0.7")),
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: openai, azure, anthropic, google")


def get_embedding_provider() -> str:
    """Get the configured embedding provider from environment variables."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    # Check if Azure is configured (takes precedence)
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    return provider


def create_embedding_model():
    """
    Create an embedding model instance based on the configured provider.
    
    Returns:
        Embedding model instance
    """
    provider = get_embedding_provider()
    
    if provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set")
        
        model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        return AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            azure_deployment=model,
            api_key=api_key,
            api_version=api_version,
        )
    
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
        )
    
    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        model = os.getenv("GOOGLE_EMBED_MODEL", "models/embedding-001")
        
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key,
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Supported: openai, azure, google")


