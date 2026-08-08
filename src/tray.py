import threading
from PIL import Image, ImageDraw
import pystray


def _make_icon_image(color="green") -> Image.Image:
    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, size - 8, size - 8), fill=color)
    return image


class SystemTray:
    def __init__(self, on_quit):
        self.paused = threading.Event()  # set = paused, clear = running
        self._on_quit = on_quit
        self.show_masks = True
        self.show_windows = True

        #track active status for each camera feed
        self.enabled_cameras = {1: True, 2:True, 3:True, 4:True}
        
        self._icon = pystray.Icon(
            "person_detection",
            _make_icon_image("green"),
            "Person Detection",
            menu=pystray.Menu(
                pystray.MenuItem("Pause All", self._toggle_pause, checked=lambda item: self.paused.is_set()),
                pystray.Menu.SEPARATOR,

                pystray.MenuItem("Show Camera Views", self._toggle_windows, checked=lambda item: self.show_windows),
                pystray.MenuItem("Show Mask Views", self._toggle_masks, checked=lambda item: self.show_masks),
                pystray.Menu.SEPARATOR,

                pystray.MenuItem("Camera 1", self._toggle_cam(1), checked=lambda item: self.enabled_cameras[1]),
                pystray.MenuItem("Camera 2", self._toggle_cam(2), checked=lambda item: self.enabled_cameras[2]),
                pystray.MenuItem("Camera 3", self._toggle_cam(3), checked=lambda item: self.enabled_cameras[3]),
                pystray.MenuItem("Camera 4", self._toggle_cam(4), checked=lambda item: self.enabled_cameras[4]),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self._quit),
            ),
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True)

    def _toggle_windows(self, icon, item):
        self.show_windows = not self.show_windows
        icon.update_menu()

    def is_window_enabled(self) -> bool:
        return self.show_windows

    def _toggle_masks(self, icon, item):
        self.show_masks = not self.show_masks
        icon.update_menu()

    def is_mask_enabled(self) -> bool:
        return self.show_masks

    def _toggle_cam(self, cam_id: int):
        def handler(icon, item):
            self.enabled_cameras[cam_id] = not self.enabled_cameras[cam_id]
            icon.update_menu() # <--- Forces the tray to redraw the checkmark!
        return handler

    def is_camera_enabled(self, cam_id: int) -> bool:
        return self.enabled_cameras.get(cam_id, True)

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
