"""Script: baixa um modelo GGUF do catalogo do DaviOS.

Uso:
    .venv\\Scripts\\python.exe scripts\\download_model.py
    .venv\\Scripts\\python.exe scripts\\download_model.py --model qwen3-8b-q4km
    .venv\\Scripts\\python.exe scripts\\download_model.py --model "qwen3 8b" --yes
    .venv\\Scripts\\python.exe scripts\\download_model.py --list

O script NAO aceita URL. A origem do arquivo e exclusivamente o catalogo
(`config/model_catalog.json`), e o destino e derivado de la -- nunca de um
caminho informado na linha de comando. Isso e deliberado: o downloader
canonico existe para impedir que uma URL arbitraria vire um `.gguf` em um
diretorio escolhido por quem chamou.

Quem resolve tudo, do `model_id` ao `.gguf` final, e
`brain.model_downloader.ModelDownloader`. Este arquivo so traduz argumentos
de linha de comando em chamadas a ele.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.model_downloader import (  # noqa: E402
    STATUS_ALREADY_INSTALLED,
    STATUS_CONFIRMATION_REQUIRED,
    STATUS_DECLINED,
    STATUS_DOWNLOADED,
    STATUS_LOCKED,
    STATUS_REFUSED,
    DownloadPreview,
    ModelDownloader,
)

# Modelo padrao: apenas um valor padrao de CLI. A fonte da verdade continua
# sendo o catalogo.
DEFAULT_MODEL = "qwen3-4b-q4km"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baixa um modelo GGUF do catalogo do DaviOS.",
        epilog="A origem e o destino vem sempre do catalogo; nao ha opcao de URL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "ID, alias ou nome do modelo no catalogo (padrao: %s). "
            "Ex.: qwen3-8b-q4km, 'qwen3 8b'." % DEFAULT_MODEL
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista os modelos com download disponivel e sai.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirma automaticamente. Sem isto, o script mostra a previa e "
             "espera uma resposta.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Nao exibe o progresso do download.",
    )
    return parser


def _make_downloader() -> ModelDownloader:
    """Downloader canonico, com catalogo e config do projeto."""
    return ModelDownloader(project_root=ROOT)


def _list_available(downloader: ModelDownloader) -> int:
    catalog = downloader.catalog
    available = catalog.get_downloadable_models()
    if not available:
        print("Nenhum modelo com origem de download confirmada no catalogo.")
        return 1

    installed = {m.id for m in catalog.get_installed_models()}
    print("Modelos com download disponivel no catalogo:\n")
    for model in sorted(available, key=lambda m: m.size_bytes):
        mark = "[instalado]" if model.id in installed else "[para baixar]"
        print(
            "  %-18s %-8s ~%6.2f GiB  %s"
            % (model.id, model.quantization, model.size_bytes / (1024 ** 3), mark)
        )
    return 0


def _progress_printer(quiet: bool):
    if quiet:
        return None

    state = {"last": -1}

    def report(downloaded: int, total: int) -> None:
        # `total=0` significa "servidor nao declarou Content-Length": mostrar
        # porcentagem ali seria inventar um denominador.
        if total:
            percent = int(downloaded * 100 / total)
            bucket = percent // 5
            if bucket != state["last"]:
                state["last"] = bucket
                print(
                    "  %3d%%  %6.1f MiB / %6.1f MiB"
                    % (percent, downloaded / (1024 ** 2), total / (1024 ** 2)),
                    flush=True,
                )
        else:
            bucket = downloaded // (50 * 1024 * 1024)
            if bucket != state["last"]:
                state["last"] = bucket
                print("  %6.1f MiB baixados" % (downloaded / (1024 ** 2),), flush=True)

    return report


def _confirm_interactive(preview: DownloadPreview) -> bool:
    """Pergunta ao usuario. So e chamada quando `require_confirmation=True`."""
    print()
    print("Confirmar o download?")
    try:
        answer = input("  [s/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("s", "sim", "y", "yes")


def _report_integrity(result) -> list[str]:
    """Linhas sobre a verificação criptográfica, se houver algo a dizer.

    A CLI NÃO calcula hash: ela só traduz o que o downloader já verificou.
    Isso mantém uma única fonte de verdade (o downloader) e impede a CLI de
    "confirmar" algo que ninguém checou.
    """
    lines: list[str] = []

    if result.hash_verified:
        lines.append("Integridade verificada: SHA-256 confere.")
        return lines

    if result.actual_sha256 and result.expected_sha256:
        # Os dois hashes estão preenchidos e são diferentes: houve mismatch.
        lines.append("")
        lines.append("O arquivo baixado não confere com o SHA-256 esperado.")
        lines.append("  esperado: %s" % result.expected_sha256)
        lines.append("  obtido  : %s" % result.actual_sha256)
        lines.append("Nada foi instalado.")
        return lines

    if result.expected_sha256 is None:
        lines.append(
            "Aviso: este modelo não possui SHA-256 no catálogo. "
            "A integridade criptográfica não foi verificada."
        )
    return lines


def _report(result) -> int:
    """Traduz o DownloadResult em texto e codigo de saida."""
    if result.status == STATUS_DOWNLOADED:
        print()
        print("Modelo pronto.")
        print("  Arquivo : %s" % result.destination)
        print("  Tamanho : %d bytes" % result.downloaded_bytes)
        for line in _report_integrity(result):
            print("  %s" % line if line else "")
        return 0

    if result.status == STATUS_ALREADY_INSTALLED:
        print()
        print("O modelo ja estava instalado: %s" % result.destination)
        for warning in result.validation_warnings:
            print("  aviso: %s" % warning)
        for line in _report_integrity(result):
            print("  %s" % line if line else "")
        return 0

    if result.status == STATUS_LOCKED:
        # Conflito de lock NÃO é falha nem recusa: nada foi tentado. A CLI não
        # espera — disser ao usuário o que fazer é mais útil do que prendê-lo.
        print()
        print("Ja existe um download em andamento para este modelo.")
        if result.lock_holder:
            print("  processo: %s" % result.lock_holder)
        print("Nenhum download novo foi iniciado.")
        print("Tente novamente quando o download atual terminar.")
        return 5

    if result.status == STATUS_CONFIRMATION_REQUIRED:
        print()
        print(result.summary())
        print("Use --yes para confirmar automaticamente.")
        return 2

    if result.status == STATUS_DECLINED:
        print()
        print("Download cancelado. Nada foi baixado.")
        return 3

    if result.status == STATUS_REFUSED:
        print()
        print(result.summary())
        return 4

    print()
    print(result.summary())
    for message in result.validation_errors:
        print("  erro: %s" % message)
    for line in _report_integrity(result):
        print("  %s" % line if line else "")
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        downloader = _make_downloader()
    except Exception as exc:  # noqa: BLE001 - erro de config vira saida limpa
        print("Nao foi possivel preparar o downloader: %s" % exc)
        return 1

    if args.list:
        return _list_available(downloader)

    # A previa e read-only: mostra o que aconteceria e nao toca a rede nem
    # escreve nada. Se ela disser que nao pode, nao ha nem o que perguntar.
    try:
        preview = downloader.preview(args.model)
    except Exception as exc:  # noqa: BLE001
        print("Nao foi possivel consultar o catalogo: %s" % exc)
        return 1

    if not preview.can_download:
        print(preview.summary())
        if preview.blocked_reason:
            print()
            print("Motivo: %s" % preview.blocked_reason)
        return 4

    print()
    print(preview.summary())
    for warning in preview.warnings:
        print("  aviso: %s" % warning)
    if preview.already_installed:
        print()
        print("O arquivo ja existe e parece valido; nada sera sobrescrito.")

    # `--yes` nao pula validacao nenhuma: ele apenas responde `True` a
    # confirmacao, depois que a previa e as protecoes ja passaram.
    confirm = (lambda _preview: True) if args.yes else _confirm_interactive

    if not args.yes:
        print()

    try:
        result = downloader.download(
            args.model,
            progress_callback=_progress_printer(args.quiet),
            confirm=confirm,
            require_confirmation=True,
        )
    except KeyboardInterrupt:
        print()
        print("Interrompido pelo usuario.")
        return 1
    except Exception as exc:  # noqa: BLE001 - nada de traceback cru
        print()
        print("Falha inesperada no download: %s" % exc)
        return 1

    return _report(result)


if __name__ == "__main__":
    sys.exit(main())
