from rich.console import Console
from rich.panel import Panel

from memory.database import Database


class System:

    def __init__(self):

        self.console = Console()
        self.database = Database()

    def boot(self):

        self.database.initialize()

        self.database.save_memory("O David acordou pela primeira vez.")

        memories = self.database.get_memories()

        self.console.print(
            Panel.fit(
                "[bold cyan]DaviOS[/bold cyan]\n\nOlá, Davi. Bem-vindo ao DaviOS.",
                title="Boot"
            )
        )

        self.console.print(memories)