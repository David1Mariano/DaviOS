from rich.console import Console
from rich.panel import Panel

from memory.memory_manager import MemoryManager


class System:

    def __init__(self):

        self.console = Console()
        self.memory = MemoryManager()

    def boot(self):

        self.memory.remember("O David acordou pela primeira vez.")

        memories = self.memory.recall()

        self.console.print(
            Panel.fit(
                "[bold cyan]DaviOS[/bold cyan]\n\nOlá, Davi. Bem-vindo ao DaviOS.",
                title="Boot"
            )
        )

        self.console.print(memories)