import tkinter as tk

###############################################################################
# Custom Entry + Popup Listbox Autocomplete
###############################################################################
class AutocompleteEntryPopup(tk.Frame):
    """
    A custom widget with:
      - A tk.Entry for user input
      - A popup tk.Toplevel with a tk.Listbox of suggestions
    """
    def __init__(self, master, suggestion_list=None, width=30, font=("Helvetica", 10), callback=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.callback = callback
        self.suggestion_list = suggestion_list or []
        self.current_suggestions = []
        self.current_index = -1

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(self, textvariable=self.entry_var, width=width, font=font)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)

        # Bind entry events
        self.entry.bind("<KeyRelease>", self._on_keyrelease)
        self.entry.bind("<Down>", self._on_down_arrow)
        self.entry.bind("<Up>", self._on_up_arrow)
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Tab>", self._on_tab_press)
        self.entry.bind("<FocusOut>", self._on_focus_out)

        self.popup = None

    def _on_keyrelease(self, event):
        if event.keysym in ("Up","Down","Left","Right","Return","Tab","Escape"):
            return

        text = self.entry_var.get().strip()

        if not text:
            # Show the entire suggestion_list when the box is blank
            matches = list(self.suggestion_list)
            if matches:
                self._show_popup(matches)
            else:
                self._hide_popup()
            return

        matches = self._filter_suggestions(text)
        if matches:
            self._show_popup(matches)
        else:
            self._hide_popup()

    def _show_popup(self, suggestions):
        self._hide_popup()
        self.current_suggestions = suggestions
        self.current_index = 0

        self.popup = tk.Toplevel(self)
        self.popup.wm_overrideredirect(True)
        x = self.entry.winfo_rootx()
        y = self.entry.winfo_rooty() + self.entry.winfo_height()
        self.popup.geometry(f"+{x}+{y}")

        self.listbox = tk.Listbox(self.popup, selectmode=tk.SINGLE, height=min(6, len(suggestions)))
        self.listbox.pack(fill="both", expand=True)
        for item in suggestions:
            self.listbox.insert(tk.END, item)

        self._wrap_index()
        self._update_listbox_selection()

        self.listbox.bind("<Button-1>", self._on_listbox_click)
        self.listbox.bind("<Return>", self._on_return)
        self.listbox.bind("<Down>", self._on_down_arrow)
        self.listbox.bind("<Up>", self._on_up_arrow)

    def _hide_popup(self):
        if self.popup and tk.Toplevel.winfo_exists(self.popup):
            self.popup.destroy()
        self.popup = None
        self.current_suggestions = []
        self.current_index = -1

    def _on_listbox_click(self, event):
        idx = self.listbox.curselection()
        if idx:
            self.current_index = idx[0]
            self._select_current()
        self._hide_popup()
        self.entry.focus_set()
        return "break"

    def _on_down_arrow(self, event):
        if not self.popup:
            return
        self.current_index += 1
        self._wrap_index()
        self._update_listbox_selection()
        return "break"

    def _on_up_arrow(self, event):
        if not self.popup:
            return
        self.current_index -= 1
        self._wrap_index()
        self._update_listbox_selection()
        return "break"

    def _on_return(self, event):
        if self.popup:
            self._select_current()
            self._hide_popup()
            self.entry.focus_set()  # Focus back on the entry widget
            self._move_to_next_widget()
            return "break"
        else:
            matches = self._filter_suggestions(self.entry_var.get().strip())
            if len(matches) == 1:
                self._set_text(matches[0])
                self._move_to_next_widget()

    def _on_tab_press(self, event):
        if self.popup:
            self._select_current()
            self._hide_popup()
            self._move_to_next_widget()
            return "break"

    def _move_to_next_widget(self):
        next_widget = self.entry.tk_focusNext()
        if next_widget:
            next_widget.focus_set()

    def _on_focus_out(self, event):
        if self.popup:
            widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
            if not (widget and str(widget).startswith(str(self.popup))):
                self._hide_popup()

    def _select_current(self):
        self._wrap_index()
        if 0 <= self.current_index < len(self.current_suggestions):
            self._set_text(self.current_suggestions[self.current_index])

    def _filter_suggestions(self, typed):
        low = typed.lower()
        return [s for s in self.suggestion_list if low in s.lower()]

    def _set_text(self, text):
        self.entry_var.set(text)
        if self.callback:
            self.callback()

    def _wrap_index(self):
        """
        Wrap current_index around if it goes out of bounds
        """
        count = len(self.current_suggestions)
        if count == 0:
            self.current_index = -1
        else:
            self.current_index %= count

    def _update_listbox_selection(self):
        """
        Reflect current_index in the listbox UI
        """
        self.listbox.select_clear(0, tk.END)
        if 0 <= self.current_index < len(self.current_suggestions):
            self.listbox.select_set(self.current_index)
            self.listbox.activate(self.current_index)
            self.listbox.see(self.current_index)  # Scroll to make the selection visible

    def get_text(self):
        return self.entry_var.get()