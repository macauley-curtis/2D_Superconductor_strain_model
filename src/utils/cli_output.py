from rich.console import Console
from rich.panel import Panel

console = Console()


def toast(message, color="white", title=None):
    """
    Display a reusable toast-style console message.

    Parameters
    ----------
    message : str
        Message to display.

    color : str
        Rich colour name, e.g.:
        "green", "red", "yellow", "cyan", "blue".

    title : str, optional
        Optional title displayed in the panel.
    """

    console.print(
        Panel(
            f"[{color}]{message}[/{color}]",
            title=title,
            border_style=color,
            expand=False,
        )
    )
