"""
models/__init__.py — FigureVault models package
"""
from .ollama_client import OllamaClient, OllamaError

__all__ = ["OllamaClient", "OllamaError"]
