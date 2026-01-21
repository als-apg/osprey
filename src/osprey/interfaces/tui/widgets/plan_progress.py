"""Plan progress bar widget for the TUI.

A persistent widget that shows execution plan progress above the input area.
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static

from osprey.interfaces.tui.widgets.blocks import TodoList


class PlanProgressBar(Vertical):
    """Persistent plan progress bar above input area.

    Shows todo list with current execution progress.
    Hidden initially, appears when plan is created.
    Can be toggled with Ctrl+O.
    """

    can_focus = False  # Prevent focus stealing when toggled visible

    class PlanUpdated(Message):
        """Posted when plan progress changes (for auto-scroll)."""

        pass

    def __init__(self, **kwargs):
        """Initialize the progress bar.

        Args:
            **kwargs: Additional arguments passed to Vertical.
        """
        super().__init__(**kwargs)
        self._plan_steps: list[dict] = []
        self._states: list[str] = []
        self._mounted = False

    @property
    def is_streaming(self) -> bool:
        """Check if execution is actively streaming.

        Returns:
            True if any step has 'current' state, False otherwise.
        """
        return "current" in self._states

    def _build_hint(self) -> str:
        """Build the hint string with styled Ctrl-O (matches StatusPanel style)."""
        return "[dim]([/dim][$text bold]Ctrl-O[/$text bold][dim] to hide)[/dim]"

    def _build_header(self, status_text: str) -> str:
        """Build header text with dynamic styling based on streaming state.

        Args:
            status_text: The status text (e.g., "0/5 complete", "No active plan").

        Returns:
            Rich markup string with appropriate styling.
        """
        hint = self._build_hint()

        if self.is_streaming:
            # Active streaming - bold + accent color (matches ProcessingBlock active style)
            return f"[bold][$accent]Plan:[/$accent][/bold] {status_text}  {hint}"
        else:
            # Idle state - bold, regular color
            return f"[bold]Plan:[/bold] {status_text}  {hint}"

    def compose(self) -> ComposeResult:
        """Compose the progress bar with header and todo list."""
        yield Static("", id="progress-header")
        yield TodoList(id="progress-todos")

    def on_mount(self) -> None:
        """Handle mount event."""
        self._mounted = True
        # Hide todo list initially (will be shown when set_plan is called)
        todo_list = self.query_one("#progress-todos", TodoList)
        todo_list.display = False
        # Initialize header with empty state
        header = self.query_one("#progress-header", Static)
        header.update(self._build_header("No active plan"))

    def set_plan(self, steps: list[dict]) -> None:
        """Initialize with plan steps (all pending).

        Shows the progress bar and populates the todo list.
        Clears any existing plan from previous queries.

        Args:
            steps: List of step dicts with 'task_objective' key.
        """
        self._plan_steps = steps
        self._states = ["pending"] * len(steps)

        # Hide command dropdown (mutual exclusivity with overlays)
        try:
            from osprey.interfaces.tui.widgets.input import CommandDropdown

            dropdown = self.app.query_one("#command-dropdown", CommandDropdown)
            if dropdown.is_visible:
                dropdown.hide()
        except Exception:
            pass  # Dropdown may not exist

        if self._mounted:
            # Clear and rebuild todo list for new plan
            todo_list = self.query_one("#progress-todos", TodoList)
            todo_list.set_todos(self._plan_steps, self._states)
            todo_list.display = True

            # Update header
            header = self.query_one("#progress-header", Static)
            header.update(self._build_header(f"0/{len(steps)} complete"))

        self.display = True  # Show the bar

    def update_progress(self, states: list[str]) -> None:
        """Update states in-place.

        Called when a capability starts/completes to update the visual progress.

        Args:
            states: List of states ("pending", "current", "done").
        """
        if len(states) != len(self._states):
            return  # Mismatch, ignore

        self._states = states.copy()
        self._update_display()
        self.post_message(self.PlanUpdated())

    def _update_display(self) -> None:
        """Update the header and todo list display."""
        if not self._mounted:
            return

        # Update header with progress summary
        done = self._states.count("done")
        total = len(self._states)
        header = self.query_one("#progress-header", Static)
        header.update(self._build_header(f"{done}/{total} complete"))

        # Update todo list
        todo_list = self.query_one("#progress-todos", TodoList)
        if not todo_list.display:
            # First time - need to set todos
            todo_list.set_todos(self._plan_steps, self._states)
        else:
            # Already displayed - update states in-place
            todo_list.update_states(self._states)

    def clear(self) -> None:
        """Hide and reset for new query (clears all data)."""
        self.display = False
        self._plan_steps = []
        self._states = []
        if self._mounted:
            todo_list = self.query_one("#progress-todos", TodoList)
            todo_list.display = False
            # Update header to show empty state
            header = self.query_one("#progress-header", Static)
            header.update(self._build_header("No active plan"))
        self.refresh()  # Force immediate UI update

    def mark_complete(self) -> None:
        """Mark all items as done and hide bar (keeps data for later viewing)."""
        if self._plan_steps:
            self._states = ["done"] * len(self._plan_steps)
            if self._mounted:
                # Update header to show finished state
                total = len(self._plan_steps)
                header = self.query_one("#progress-header", Static)
                header.update(self._build_header(f"{total}/{total} complete ✓"))
                # Update todo list states
                todo_list = self.query_one("#progress-todos", TodoList)
                todo_list.update_states(self._states)
        self.display = False
        self.refresh()  # Force immediate UI update

    def has_plan(self) -> bool:
        """Check if there's an active plan.

        Returns:
            True if there's a plan, False otherwise.
        """
        return len(self._plan_steps) > 0
