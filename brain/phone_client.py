"""PhoneClient do DaviOS — cliente HTTP para o servidor Termux no celular.

Opt-in, desabilitado por padrão. Token lido de variável de ambiente
DAVIOS_PHONE_TOKEN (nunca no davios.json). Nunca levanta exceção.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger("davios.phone")


class PhoneClient:
    """Cliente HTTP tipado para o servidor Termux (Galaxy A30S + Termux:API).

    - base_url: ex. "http://192.168.1.50:8080" (sem barra final)
    - token: Bearer token; se None, lê de os.environ["DAVIOS_PHONE_TOKEN"]
    - timeout: segundos por requisição (padrão 10.0)
    - opener: urllib.request.OpenerDirector opcional para injeção de transporte
      (usado em testes para apontar para servidor local). Default: urllib padrão.

    Todos os métodos retornam dict padronizado:
        {"ok": True, ...dados...}  em sucesso
        {"ok": False, "error": "mensagem amigável em PT-BR"}  em falha
    NUNCA levanta exceção. Nunca vaza token, código HTTP ou stack trace.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout: float = 10.0,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("DAVIOS_PHONE_TOKEN", "")
        self.timeout = timeout
        self._opener = opener or urllib.request.build_opener()

    # ------------------------------------------------------------------
    # Método interno de requisição
    # ------------------------------------------------------------------

    def _request(
        self, method: str, path: str, json_body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Executa requisição HTTP e retorna dict padronizado (ok/error)."""
        if not self.base_url:
            return {
                "ok": False,
                "error": "URL base do celular não configurada (phone_base_url).",
            }

        # Valida esquema da URL base
        parsed_base = urllib.parse.urlparse(self.base_url)
        if parsed_base.scheme not in ("http", "https"):
            return {
                "ok": False,
                "error": "phone_base_url deve usar http:// ou https://.",
            }

        url = f"{self.base_url}{path}"
        headers = {
            "User-Agent": "DaviOS/1.0 (assistente local offline-first)",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        data = None
        if json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                status = resp.getcode() or 200
                # Limita leitura a 64 KB para evitar respostas gigantes
                raw = resp.read(64 * 1024 + 1)
                if len(raw) > 64 * 1024:
                    return {
                        "ok": False,
                        "error": "Resposta do celular excede o limite de 64 KB.",
                    }
        except urllib.error.HTTPError as e:
            # Não vazar código HTTP nem corpo de erro técnico
            logger.warning("[PHONE] HTTP %s ao acessar %s", e.code, path)
            return {
                "ok": False,
                "error": f"Servidor do celular respondeu com erro ({e.code}).",
            }
        except urllib.error.URLError as e:
            logger.warning("[PHONE] Erro de rede ao acessar %s: %s", path, e.reason)
            return {
                "ok": False,
                "error": "Não foi possível conectar ao celular. Verifique IP/porta e rede.",
            }
        except TimeoutError:
            logger.warning("[PHONE] Timeout ao acessar %s", path)
            return {
                "ok": False,
                "error": "Tempo esgotado ao falar com o celular.",
            }
        except Exception as e:  # noqa: BLE001 — captura deliberada
            logger.warning("[PHONE] Erro inesperado em %s: %s", path, e)
            return {
                "ok": False,
                "error": "Erro interno ao comunicar com o celular.",
            }

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            logger.warning("[PHONE] Resposta não-JSON de %s", path)
            return {
                "ok": False,
                "error": "Resposta inválida do celular.",
            }

        if not isinstance(payload, dict):
            return {"ok": False, "error": "Resposta inválida do celular."}

        # O servidor do celular deve retornar {"ok": true, ...} ou {"ok": false, "error": "..."}
        if not payload.get("ok", False):
            err = payload.get("error", "Erro desconhecido do celular.")
            return {"ok": False, "error": err}

        return payload

    # ------------------------------------------------------------------
    # Sensores seguros (category: phone_read)
    # ------------------------------------------------------------------

    def get_battery(self) -> dict[str, Any]:
        """Retorna nível de bateria e status de carregamento.

        Sucesso: {"ok": True, "level": int, "charging": bool}
        """
        return self._request("GET", "/battery")

    def get_location(self) -> dict[str, Any]:
        """Retorna localização GPS atual.

        Sucesso: {"ok": True, "lat": float, "lon": float, "accuracy": float}
        """
        return self._request("GET", "/location")

    def get_device_info(self) -> dict[str, Any]:
        """Retorna informações do dispositivo.

        Sucesso: {"ok": True, "model": str, "android_version": str, ...}
        """
        return self._request("GET", "/device")

    # ------------------------------------------------------------------
    # Ações sensíveis (category: phone_act)
    # ------------------------------------------------------------------

    def speak(self, text: str) -> dict[str, Any]:
        """Faz o celular falar via TTS (Termux:TTS).

        Validação: máximo 300 caracteres, remove caracteres de controle.
        Sucesso: {"ok": True, "spoken": True}
        """
        if not isinstance(text, str):
            return {"ok": False, "error": "Texto para fala deve ser string."}

        # Remove caracteres de controle (exceto \n, \t, \r)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        cleaned = cleaned.strip()
        if not cleaned:
            return {"ok": False, "error": "Texto vazio após limpeza."}
        if len(cleaned) > 300:
            return {"ok": False, "error": "Texto muito longo (máx. 300 caracteres)."}

        return self._request("POST", "/tts/speak", {"text": cleaned})

    def vibrate(self, ms: int) -> dict[str, Any]:
        """Vibra o celular por ms milissegundos (Termux:Vibrate).

        Validação: 50–2000 ms.
        Sucesso: {"ok": True, "vibrated": True}
        """
        if not isinstance(ms, int):
            return {"ok": False, "error": "Duração deve ser inteiro (ms)."}
        if ms < 50 or ms > 2000:
            return {"ok": False, "error": "Duração deve ser entre 50 e 2000 ms."}

        return self._request("POST", "/vibrate", {"duration": ms})