"""
models/ollama_client.py — Ollama API wrapper for FigureVault

Provides:
  • OllamaClient.query_text()         – text-only inference
  • OllamaClient.query_multimodal()   – vision + text inference (Gemma 4 E4B)
  • OllamaClient.is_available()       – health check

The multimodal method encodes the image as base64 and passes it in the
``images`` field of the Ollama /api/generate payload, which is what the
Ollama API requires for vision models.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MAX_RETRIES, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when the Ollama API returns an unexpected response."""


class OllamaClient:
    """Thin, retry-capable wrapper around the Ollama REST API.

    Parameters
    ----------
    base_url : str
        Root URL of the Ollama server, e.g. ``"http://localhost:11434"``.
    model : str
        Model tag to use for inference, e.g. ``"gemma4:4b"``.
    timeout : int
        Per-request timeout in seconds.
    max_retries : int
        Number of retry attempts on transient failures.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
        max_retries: int = OLLAMA_MAX_RETRIES,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable and the model is loaded.

        Performs a lightweight GET to ``/api/tags`` and checks whether
        ``self.model`` appears in the tag list.
        """
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            tags = [m.get("name", "") for m in resp.json().get("models", [])]
            if not any(self.model in t for t in tags):
                logger.warning(
                    "Model '%s' not found in Ollama.  Available: %s",
                    self.model,
                    tags,
                )
                return False
            return True
        except requests.RequestException as exc:
            logger.error("Ollama health-check failed: %s", exc)
            return False

    def query_text(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        """Send a text-only prompt to Ollama and return the response string.

        Parameters
        ----------
        prompt : str
            The user turn content.
        system : str
            Optional system instruction prepended to the conversation.
        temperature : float
            Sampling temperature (lower = more deterministic).

        Returns
        -------
        str
            The model's response text.
        """
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = self._call_with_retry(
            endpoint="/api/generate",
            payload=payload,
        )
        return response.get("response", "")

    def query_multimodal(
        self,
        prompt: str,
        image_path: str | Path,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        """Send a vision + text prompt to Ollama and return the response string.

        The image file at ``image_path`` is read from disk, base64-encoded,
        and submitted in the ``images`` field of the Ollama API payload — the
        format required by Ollama's multimodal models (LLaVA, Gemma4, etc.).

        Parameters
        ----------
        prompt : str
            The text part of the user turn.
        image_path : str | Path
            Absolute or relative path to a PNG/JPEG image file.
        system : str
            Optional system instruction.
        temperature : float
            Sampling temperature.

        Returns
        -------
        str
            The model's response text.
        """
        image_b64 = self._encode_image(image_path)
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "images": [image_b64],   # Ollama expects a list of base64 strings
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = self._call_with_retry(
            endpoint="/api/generate",
            payload=payload,
        )
        return response.get("response", "")

    def query_multimodal_chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
    ) -> str:
        """Send a chat-format multimodal conversation to Ollama.

        Each message in ``messages`` follows the format::

            {
                "role": "user" | "assistant",
                "content": "text here",
                "images": ["<base64>", ...]   # optional, for user turns
            }

        Returns
        -------
        str
            The assistant's response content.
        """
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = self._call_with_retry(
            endpoint="/api/chat",
            payload=payload,
        )
        return response.get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image_path: str | Path) -> str:
        """Read an image file and return its base64-encoded bytes as a string."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with path.open("rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    def _call_with_retry(self, endpoint: str, payload: dict) -> dict:
        """POST ``payload`` to ``endpoint`` with exponential-backoff retry logic.

        Parameters
        ----------
        endpoint : str
            API path, e.g. ``"/api/generate"``.
        payload : dict
            JSON-serialisable request body.

        Returns
        -------
        dict
            Parsed JSON response.

        Raises
        ------
        OllamaError
            If all retries are exhausted or the server returns a non-2xx status.
        """
        url = f"{self.base_url}{endpoint}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "Ollama timeout on attempt %d/%d — retrying in %ds",
                    attempt, self.max_retries, wait,
                )
                time.sleep(wait)
            except requests.exceptions.HTTPError as exc:
                # Surface server-side errors immediately (no point retrying 400s)
                body = exc.response.text if exc.response is not None else ""
                raise OllamaError(
                    f"Ollama HTTP {exc.response.status_code}: {body}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "Ollama request error on attempt %d/%d (%s) — retrying in %ds",
                    attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise OllamaError(
            f"Ollama unreachable after {self.max_retries} attempts: {last_exc}"
        )
