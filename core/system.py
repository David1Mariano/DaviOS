from rich.console import Console
from rich.panel import Panel

from memory.memory_manager import MemoryManager


class System:

    def __init__(self):
        self.console = Console()
        self.memory_manager = MemoryManager()

    def boot(self):

        self.console.print(
            Panel.fit(
                "[bold cyan]DaviOS[/bold cyan]\n\n"
                "Olá, Davi. Bem-vindo ao DaviOS.",
                title="Boot"
            )
        )

        memories = self.memory_manager.recall()

        self.console.print(memories)