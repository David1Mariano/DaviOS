"""ActionManager do DaviOS.

Permite executar APENAS acoes de uma allowlist explicita e curta.
Nenhuma execucao arbitraria: comandos nao listados sao recusados.

Seguranca:
- allowlist estatica (comandos simples, sem shell, sem pipes);
- parametros validados por regex no proprio comando;
- acoes marcadas como destructive requerem confirmacao explicita;
- subprocess SEM shell=True (nunca herda shell aberto).
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("davios.actions")


@dataclass
class ActionResult:
    """Resultado de uma acao executada ou recusada."""

    ok: bool
    message: str
    command: str = ""
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout[-2000:],
            "stderr": self.stderr[-2000:],
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class AllowedCommand:
    """Comando permitido na allowlist."""

    name: str
    argv: list[str]
    description: str
    destructive: bool = False
    args_pattern: str = r"^[a-zA-Z0-9_\-\s.,/\\]+$"
    max_args_length: int = 80
    handler: Optional[Callable[[str], str]] = None

    def validate_args(self, args: str) -> Optional[str]:
        """Valida os parametros do usuario. Retorna erro ou None."""
        if not args:
            return None
        if len(args) > self.max_args_length:
            return "Parametros muito longos."
        if not re.match(self.args_pattern, args):
            return "Parametros contem caracteres nao permitidos."
        return None



def _read_file_handler(path: str, max_bytes: int = 200_000) -> str:
    """Le um arquivo de texto com validacoes de seguranca."""
    from pathlib import Path

    if not path:
        raise ValueError("Caminho do arquivo nao informado.")

    file_path = Path(path)

    try:
        abs_path = file_path.resolve()
    except OSError as e:
        raise ValueError(f"Caminho invalido: {e}")

    project_root = Path(__file__).resolve().parent.parent
    try:
        abs_path.relative_to(project_root)
    except ValueError:
        raise PermissionError(f"Acesso negado: fora do diretorio permitido.")

    if not abs_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: '{path}'")

    if not abs_path.is_file():
        raise ValueError(f"O caminho nao e um arquivo: '{path}'")

    file_size = abs_path.stat().st_size
    if file_size > max_bytes:
        content = abs_path.read_bytes()[:max_bytes].decode('utf-8', errors='replace')
        return f"{content}\n\n[AVISO: arquivo truncado em {max_bytes} bytes de {file_size} bytes totais]"

    return abs_path.read_text(encoding='utf-8', errors='replace')


class ActionManager:
    """Executa acoes da allowlist com confirmacion quando necesario."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        confirm: Optional[Callable[[str], bool]] = None,
        extra_commands: Optional[list[AllowedCommand]] = None,
    ):
        self.enabled = enabled
        self.confirm = confirm
        self._commands: dict[str, AllowedCommand] = {}
        self._register_builtins()
        for cmd in extra_commands or []:
            self.register(cmd)


    def _register_builtins(self) -> None:
        self.register(
            AllowedCommand(
                name="datetime",
                argv=["cmd", "/c", "date", "/t"],
                description="Mostra a data atual do sistema.",
            )
        )
        self.register(
            AllowedCommand(
                name="time",
                argv=["cmd", "/c", "time", "/t"],
                description="Mostra a hora atual do sistema.",
            )
        )
        self.register(
            AllowedCommand(
                name="read_file",
                argv=[],
                description="Le o conteudo de um arquivo de texto (somente leitura).",
                args_pattern=r"^[a-zA-Z0-9_.\\\-/]+$",
                max_args_length=200,
                handler=lambda args: _read_file_handler(args),
            )
        )
        self.register(
            AllowedCommand(
                name="echo",
                argv=["cmd", "/c", "echo", "{args}"],
                description=(
                    "Repete exatamente o texto informado. Use APENAS quando o usuario "
                    "pedir explicitamente para repetir, ecoar ou devolver um texto "
                    "especifico — nunca como resposta padrao a mensagens casuais, "
                    "agradecimentos ou reacoes."
                ),
                args_pattern=r"^[a-zA-Z0-9_\s.,!?\"'áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ\-]+$",
            )
        )
        self.register(
            AllowedCommand(
                name="read_file",
                argv=[],
                description="Le o conteudo de um arquivo de texto (somente leitura).",
                args_pattern=r"^[a-zA-Z0-9_.\\\-/]+$",
                max_args_length=200,
                handler=lambda args: _read_file_handler(args),
            )
        )
        self.register(
            AllowedCommand(
                name="read_file",
                argv=[],
                description="Le o conteudo de um arquivo de texto (somente leitura).",
                args_pattern=r"^[a-zA-Z0-9_.\\\-/]+$",
                max_args_length=200,
                handler=lambda args: _read_file_handler(args),
            )
        )
        self.register(
            AllowedCommand(
                name="list_dir",
                argv=["cmd", "/c", "dir", "{args}"],
                description="Lista arquivos de um diretorio simples.",
                destructive=False,
                args_pattern=r"^[a-zA-Z]:\\[a-zA-Z0-9_\\\-.]*$|^[a-zA-Z0-9_\\\-.]{1,40}$",
            )
        )

    def register(self, cmd: AllowedCommand) -> None:
        self._commands[cmd.name] = cmd

    def list_commands(self) -> list[dict[str, Any]]:
        return [
            {
                "name": c.name,
                "description": c.description,
                "destructive": c.destructive,
            }
            for c in self._commands.values()
        ]

    def is_allowed(self, name: str) -> bool:
        return name in self._commands

    def execute(self, name: str, args: str = "") -> ActionResult:
        """Tenta executar uma acao da allowlist. Nunca lanca excecao."""
        if not self.enabled:
            return ActionResult(
                ok=False,
                message=(
                    "As acoes de sistema estan deshabilitadas. "
                    "Active en config (actions_enabled)."
                ),
                command=name,
            )
        cmd = self._commands.get(name)
        if cmd is None:
            allowed = ", ".join(self._commands) or "ninguna"
            return ActionResult(
                ok=False,
                message=(
                    f"Acao '{name}' no esta en la allowlist. "
                    f"Acoes permitidas: {allowed}"
                ),
                command=name,
            )

        if cmd.destructive:
            if self.confirm is None:
                return ActionResult(
                    ok=False,
                    message=(
                        f"A acao '{name}' requiere confirmacion y no hay mecanismo "
                        "de confirmacion configurado."
                    ),
                    command=name,
                    requires_confirmation=True,
                )
            try:
                confirmed = self.confirm(
                    f"A acao '{name}' pode modificar o sistema. Confirmar? (s/n)"
                )
            except Exception:
                confirmed = False
            if not confirmed:
                return ActionResult(
                    ok=False,
                    message=f"Acao '{name}' cancelada pelo usuario.",
                    command=name,
                    requires_confirmation=True,
                )

        error = cmd.validate_args(args)
        if error:
            return ActionResult(ok=False, message=error, command=name)

        if cmd.handler is not None:
            try:
                result_text = cmd.handler(args)
                return ActionResult(ok=True, message=result_text, command=name, stdout=result_text)
            except Exception as e:
                return ActionResult(ok=False, message=f"Erro ao executar '{name}': {e}", command=name)

        parts = [p.replace("{args}", args) if "{args}" in p else p for p in cmd.argv]
        parts = [p for p in parts if p != ""]
        try:
            proc = subprocess.run(
                parts, capture_output=True, text=True, shell=False, timeout=15
            )
        except FileNotFoundError:
            return ActionResult(
                ok=False,
                message=f"Comando '{parts[0]}' no encontrado en el sistema.",
                command=name,
            )
        except subprocess.TimeoutExpired:
            return ActionResult(
                ok=False, message=f"Acao '{name}' excedio el tiempo limite.", command=name
            )

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        ok = proc.returncode == 0
        msg = out or err or (f"Acao '{name}' executada." if ok else "Falha ao executar.")
        return ActionResult(
            ok=ok,
            message=msg,
            command=name,
            returncode=proc.returncode,
            stdout=out,
            stderr=err,
        )