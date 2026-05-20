#!/usr/bin/env python3
"""GTK3 overlay window for streaming STT partial results."""

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, Gdk, Pango

import threading


class GTKOverlay:
    def __init__(self, opacity: float = 0.85, font_size: int = 18):
        self._opacity = opacity
        self._font_size = font_size
        self._window = None
        self._label = None
        self._thread = None
        self._visible = False

    def start(self):
        self._thread = threading.Thread(target=self._run_gtk, daemon=True)
        self._thread.start()

    def _run_gtk(self):
        self._window = Gtk.Window(type=Gtk.WindowType.POPUP)
        self._window.set_keep_above(True)
        self._window.set_decorated(False)
        self._window.set_skip_taskbar_hint(True)
        self._window.set_skip_pager_hint(True)

        screen = self._window.get_screen()
        visual = screen.get_rgba_visual()
        if visual and self._window.is_composited():
            self._window.set_visual(visual)

        self._window.set_app_paintable(True)
        self._window.connect("draw", self._on_draw)
        self._window.connect("screen-changed", self._on_screen_changed)

        self._label = Gtk.Label()
        self._label.set_text("")
        self._label.set_ellipsize(Pango.EllipsizeMode.END)
        self._label.set_max_width_chars(80)
        self._label.set_margin_top(12)
        self._label.set_margin_bottom(12)
        self._label.set_margin_start(20)
        self._label.set_margin_end(20)

        css_provider = Gtk.CssProvider()
        css = f"""
        label {{
            color: white;
            font-size: {self._font_size}px;
            font-weight: bold;
        }}
        """
        css_provider.load_from_data(css.encode())
        self._label.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self._window.add(self._label)
        self._window.set_default_size(500, 50)
        self._window.set_size_request(300, 50)
        self._window.show_all()
        self._window.hide()

        display = Gdk.Display.get_default()
        if display:
            monitor = display.get_primary_monitor()
            if monitor is None:
                monitor = display.get_monitor(0)
            if monitor:
                geom = monitor.get_geometry()
                win_w, win_h = self._window.get_size()
                x = geom.x + (geom.width - win_w) // 2
                y = geom.y + geom.height - win_h - 80
                self._window.move(x, y)

        Gtk.main()

    def _on_draw(self, widget, cr):
        cr.set_source_rgba(30 / 255, 30 / 255, 30 / 255, self._opacity)
        cr.set_operator(0)
        cr.paint()
        return False

    def _on_screen_changed(self, widget, old_screen):
        screen = widget.get_screen()
        visual = screen.get_rgba_visual()
        if visual and widget.is_composited():
            widget.set_visual(visual)
        else:
            widget.set_visual(screen.get_system_visual())

    def show(self, text: str = "🎤 ..."):
        self._visible = True
        if self._window:
            GLib.idle_add(self._do_show, text)

    def hide(self):
        self._visible = False
        if self._window:
            GLib.idle_add(self._do_hide)

    def update(self, text: str):
        if self._visible and self._window:
            GLib.idle_add(self._do_update, text)

    def _do_show(self, text):
        if self._label:
            self._label.set_text(text)
        if self._window:
            self._window.show_all()
        return False

    def _do_hide(self):
        if self._window:
            self._window.hide()
        return False

    def _do_update(self, text):
        if self._label:
            self._label.set_text(text)
        display = Gdk.Display.get_default()
        if display and self._window:
            monitor = display.get_primary_monitor()
            if monitor is None:
                monitor = display.get_monitor(0)
            if monitor:
                geom = monitor.get_geometry()
                win_w, win_h = self._window.get_size()
                x = geom.x + (geom.width - win_w) // 2
                y = geom.y + geom.height - win_h - 80
                self._window.move(x, y)
                self._window.resize(win_w, win_h)
        return False
