import tkinter as tk
from typing import Callable, Sequence


###############################################################################
# Custom Entry + Popup Listbox Autocomplete
###############################################################################

class AutocompleteEntryPopup(tk.Frame):
    """
    A custom widget with:
      - a tk.Entry for user input
      - a popup tk.Toplevel with a tk.Listbox of suggestions
    """

    def __init__(
        self,
        master,
        suggestion_list: Sequence[str] | None = None,
        width: int = 30,
        font=("Helvetica", 10),
        callback: Callable[[], None] | None = None,
        max_popup_rows: int = 6,
        *args,
        **kwargs,
    ):
        super().__init__(master, *args, **kwargs)

        self.callback = callback
        self.suggestion_list = list(suggestion_list or [])
        self.current_suggestions: list[str] = []
        self.current_index = -1
        self.max_popup_rows = max_popup_rows

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(
            self,
            textvariable=self.entry_var,
            width=width,
            font=font,
        )
        self.entry.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)

        self.popup: tk.Toplevel | None = None
        self.listbox: tk.Listbox | None = None

        self.entry.bind("<KeyRelease>", self._on_keyrelease)
        self.entry.bind("<Down>", self._on_down_arrow)
        self.entry.bind("<Up>", self._on_up_arrow)
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Tab>", self._on_tab_press)
        self.entry.bind("<Escape>", self._on_escape)
        self.entry.bind("<FocusOut>", self._on_focus_out)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_text(self) -> str:
        return self.entry_var.get()

    def set_text(self, text: str, trigger_callback: bool = False) -> None:
        self.entry_var.set(str(text))

        if trigger_callback and self.callback:
            self.callback()

    def clear(self, trigger_callback: bool = False) -> None:
        self.set_text("", trigger_callback=trigger_callback)
        self._hide_popup()

    def set_suggestions(self, suggestions: Sequence[str]) -> None:
        self.suggestion_list = list(suggestions)
        self._hide_popup()

    def focus(self) -> None:
        self.entry.focus_set()

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def _on_keyrelease(self, event):
        if event.keysym in {
            "Up",
            "Down",
            "Left",
            "Right",
            "Return",
            "Tab",
            "Escape",
        }:
            return

        text = self.entry_var.get().strip()

        if not text:
            matches = list(self.suggestion_list)
        else:
            matches = self._filter_suggestions(text)

        if matches:
            self._show_popup(matches)
        else:
            self._hide_popup()

    def _on_listbox_click(self, event):
        if not self.listbox:
            return "break"

        idx = self.listbox.curselection()

        if idx:
            self.current_index = idx[0]
            self._select_current()

        self._hide_popup()
        self.entry.focus_set()

        return "break"

    def _on_down_arrow(self, event):
        if not self.popup:
            self._show_popup(list(self.suggestion_list))
            return "break"

        self.current_index += 1
        self._wrap_index()
        self._update_listbox_selection()

        return "break"

    def _on_up_arrow(self, event):
        if not self.popup:
            self._show_popup(list(self.suggestion_list))
            return "break"

        self.current_index -= 1
        self._wrap_index()
        self._update_listbox_selection()

        return "break"

    def _on_return(self, event):
        if self.popup:
            self._select_current()
            self._hide_popup()
            self.entry.focus_set()
            self._move_to_next_widget()
            return "break"

        typed = self.entry_var.get().strip()
        matches = self._filter_suggestions(typed)

        if len(matches) == 1:
            self._set_text(matches[0])
            self._move_to_next_widget()
            return "break"

        return "break"

    def _on_tab_press(self, event):
        if self.popup:
            self._select_current()
            self._hide_popup()

        self._move_to_next_widget()
        return "break"

    def _on_escape(self, event):
        self._hide_popup()
        return "break"

    def _on_focus_out(self, event):
        if not self.popup:
            return

        widget_under_pointer = self.winfo_containing(
            self.winfo_pointerx(),
            self.winfo_pointery(),
        )

        if widget_under_pointer and str(widget_under_pointer).startswith(str(self.popup)):
            return

        self._hide_popup()

    # -------------------------------------------------------------------------
    # Popup helpers
    # -------------------------------------------------------------------------

    def _show_popup(self, suggestions: Sequence[str]) -> None:
        suggestions = list(suggestions)

        if not suggestions:
            self._hide_popup()
            return

        self._hide_popup()

        self.current_suggestions = suggestions
        self.current_index = 0

        self.popup = tk.Toplevel(self)
        self.popup.wm_overrideredirect(True)

        x = self.entry.winfo_rootx()
        y = self.entry.winfo_rooty() + self.entry.winfo_height()
        self.popup.geometry(f"+{x}+{y}")

        self.listbox = tk.Listbox(
            self.popup,
            selectmode=tk.SINGLE,
            height=min(self.max_popup_rows, len(suggestions)),
        )
        self.listbox.pack(fill="both", expand=True)

        for item in suggestions:
            self.listbox.insert(tk.END, item)

        self._update_listbox_selection()

        self.listbox.bind("<Button-1>", self._on_listbox_click)
        self.listbox.bind("<Return>", self._on_return)
        self.listbox.bind("<Down>", self._on_down_arrow)
        self.listbox.bind("<Up>", self._on_up_arrow)
        self.listbox.bind("<Escape>", self._on_escape)

    def _hide_popup(self) -> None:
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()

        self.popup = None
        self.listbox = None
        self.current_suggestions = []
        self.current_index = -1

    def _select_current(self) -> None:
        self._wrap_index()

        if 0 <= self.current_index < len(self.current_suggestions):
            self._set_text(self.current_suggestions[self.current_index])

    def _update_listbox_selection(self) -> None:
        if not self.listbox:
            return

        self.listbox.select_clear(0, tk.END)

        if 0 <= self.current_index < len(self.current_suggestions):
            self.listbox.select_set(self.current_index)
            self.listbox.activate(self.current_index)
            self.listbox.see(self.current_index)

    def _wrap_index(self) -> None:
        count = len(self.current_suggestions)

        if count == 0:
            self.current_index = -1
        else:
            self.current_index %= count

    # -------------------------------------------------------------------------
    # Text/filter helpers
    # -------------------------------------------------------------------------

    def _filter_suggestions(self, typed: str) -> list[str]:
        low = typed.lower()

        prefix_matches = [
            s for s in self.suggestion_list
            if s.lower().startswith(low)
        ]

        substring_matches = [
            s for s in self.suggestion_list
            if low in s.lower() and not s.lower().startswith(low)
        ]

        return prefix_matches + substring_matches

    def _set_text(self, text: str) -> None:
        self.entry_var.set(text)

        if self.callback:
            self.callback()

    def _move_to_next_widget(self) -> None:
        next_widget = self.entry.tk_focusNext()

        if next_widget:
            next_widget.focus_set()