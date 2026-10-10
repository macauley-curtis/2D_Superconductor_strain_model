from unittest.mock import patch

from rich.panel import Panel

from src.utils.cli_output import toast


def test_toast_prints_panel():
    """Test that toast prints a Rich Panel."""

    with patch("src.utils.cli_output.console.print") as mock_print:
        toast("Test message")

    mock_print.assert_called_once()

    panel = mock_print.call_args.args[0]

    assert isinstance(panel, Panel)


def test_toast_with_title_and_color():
    """Test that toast accepts a message, colour, and title."""

    with patch("src.utils.cli_output.console.print") as mock_print:
        toast(
            "Data saved successfully",
            color="green",
            title="Save Complete",
        )

    mock_print.assert_called_once()

    panel = mock_print.call_args.args[0]

    assert isinstance(panel, Panel)
    assert panel.title == "Save Complete"
    assert panel.border_style == "green"
    assert "Data saved successfully" in str(panel.renderable)


def test_toast_without_title():
    """Test that toast works when no title is provided."""

    with patch("src.utils.cli_output.console.print") as mock_print:
        toast("Warning message", color="yellow")

    mock_print.assert_called_once()

    panel = mock_print.call_args.args[0]

    assert isinstance(panel, Panel)
    assert panel.title is None
    assert panel.border_style == "yellow"
    assert "Warning message" in str(panel.renderable)


def test_toast_uses_non_expanding_panel():
    """Test that the panel does not expand to the console width."""

    with patch("src.utils.cli_output.console.print") as mock_print:
        toast("Compact message")

    panel = mock_print.call_args.args[0]

    assert panel.expand is False