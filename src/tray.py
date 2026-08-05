import threading
from PIL import Image, ImageDraw
import pystray


def _make_icon_image(color="green") -> Image.Image:
    """
    Generates a simple colored circle as the tray icon, so the project
    doesn't depend on an external image file. Color reflects state:
    green = actively detecting, gray = paused.
    """
    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, size - 8, size - 8), fill=color)
    return image


class SystemTray:
    def __init__(self, on_quit):
        """
        on_quit: a callback invoked when the user clicks Quit in the
        tray menu — lets app.py decide how to shut down cleanly
        (stopping the camera thread, releasing windows, etc.) rather
        than this module reaching into app internals directly.
        """
        self.paused = threading.Event()  # set = paused, clear = running
        self._on_quit = on_quit
        self._icon = pystray.Icon(
            "person_detection",
            _make_icon_image("green"),
            "Person Detection",
            menu=pystray.Menu(
                pystray.MenuItem("Pause", self._toggle_pause, checked=lambda item: self.paused.is_set()),
                pystray.MenuItem("Quit", self._quit),
            ),
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True)

    def start(self):
        self._thread.start()

    def _toggle_pause(self, icon, item):
        if self.paused.is_set():
            self.paused.clear()
            icon.icon = _make_icon_image("green")
        else:
            self.paused.set()
            icon.icon = _make_icon_image("gray")

    def _quit(self, icon, item):
        icon.stop()
        self._on_quit()

    def stop(self):
        self._icon.stop()
