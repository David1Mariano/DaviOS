"""WebAccess do DaviOS.

Modulo OPCIONAL e DESATIVADO por padrao (config.web_enabled=False).
Solicitudes HTTP apenas quando explicitamente habilitado; todo conteudo
baixado e sanitizado (html removido, tamanho limitado) antes de ser usado.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("davios.web")

ALLOWED_SCHEMES = {"http", "https"}
MAX_REDIRECTS = 3


@dataclass
class WebResult:
    """Resultado de uma busca web sanitizada."""

    ok: bool
    url: str = ""
    text: str = ""
    title: str = ""
    status: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "url": self.url,
            "text": self.text[:2000],
            "title": self.title,
            "status": self.status,
            "error": self.error,
        }


class WebAccess:
    """Busca paginas HTTP con sanitizacion y limites de seguridad."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        timeout_seconds: float = 5.0,
        max_content_bytes: int = 200_000,
    ):
        self.enabled = enabled
        self.timeout = timeout_seconds
        self.max_content = max_content_bytes

    def fetch(self, url: str) -> WebResult:
        """Busca y sanitiza el contenido de una URL. Nunca lanza excepcion."""
        if not self.enabled:
            return WebResult(
                ok=False,
                url=url,
                error=(
                    "El acceso a internet esta deshabilitado. "
                    "Activa en config (web_enabled) y sal del modo offline."
                ),
            )

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ALLOWED_SCHEMES:
            return WebResult(
                ok=False, url=url, error=f"Esquema '{parsed.scheme}' no permitido."
            )
        if parsed.hostname is None:
            return WebResult(ok=False, url=url, error="URL invalida.")

        try:
            return self._fetch_with_redirects(url, MAX_REDIRECTS)
        except urllib.error.HTTPError as e:
            return WebResult(ok=False, url=url, status=e.code, error=f"HTTP {e.code}")
        except Exception as e:
            return WebResult(ok=False, url=url, error=f"{type(e).__name__}: {e}")

    def _fetch_with_redirects(self, url: str, redirects_left: int) -> WebResult:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "DaviOS/1.0 (assistente local offline-first)",
                "Accept": "text/html, text/plain;q=0.9, */*;q=0.5",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            status = resp.getcode() or 200
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(self.max_content + 1)

            if status in (301, 302, 303, 307, 308) and redirects_left > 0:
                location = resp.headers.get("Location")
                if not location:
                    return WebResult(
                        ok=False, url=url, status=status, error="Redirect sin Location."
                    )
                next_url = urllib.parse.urljoin(url, location)
                return self._fetch_with_redirects(next_url, redirects_left - 1)

            if len(raw) > self.max_content:
                raw = raw[: self.max_content]

            text = raw.decode("utf-8", errors="replace")

            if not self._is_text_content(content_type, text):
                return WebResult(
                    ok=False,
                    url=url,
                    status=status,
                    error=f"Contenido no textual ({content_type or 'desconocido'}).",
                )

            return WebResult(
                ok=True,
                url=url,
                status=status,
                text=self._sanitize_text(text),
                title=self._extract_title(text),
            )

    # ------------------------------------------------------------------
    # Sanitizacion
    # ------------------------------------------------------------------

    @staticmethod
    def _is_text_content(content_type: str, body: str) -> bool:
        if not content_type:
            return True
        ct = content_type.lower()
        if any(marker in ct for marker in ("json", "xml", "html", "text/")):
            return True
        if body.startswith("\x89PNG") or body.startswith("GIF8"):
            return False
        return True

    @staticmethod
    def _sanitize_text(raw_html: str) -> str:
        """Elimina tags HTML y normaliza espacios; segmentos vacios se descartan."""
        text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2000]

    @staticmethod
    def _extract_title(raw_html: str) -> str:
        m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html)
        if not m:
            return ""
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        return title[:120]