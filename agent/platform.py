from __future__ import annotations

import base64
import importlib.util
import io
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image

pyautogui_spec = importlib.util.find_spec("pyautogui")
pyautogui = pyautogui_spec and pyautogui_spec.loader is not None
if pyautogui:
    import pyautogui as pag
else:
    pag = None

mss_spec = importlib.util.find_spec("mss")
mss_available = mss_spec and mss_spec.loader is not None
if mss_available:
    import mss
else:
    mss = None


@dataclass
class Screenshot:
    image: Image.Image
    width: int
    height: int

    def to_base64(self) -> str:
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class PlatformAdapter:
    def capture(self) -> Screenshot:
        raise NotImplementedError

    def screen_size(self) -> Tuple[int, int]:
        raise NotImplementedError

    def move(self, x: int, y: int) -> None:
        raise NotImplementedError

    def click(self, x: int, y: int, clicks: int = 1, button: str = "left") -> None:
        raise NotImplementedError

    def drag(self, start: Tuple[int, int], end: Tuple[int, int], duration: float = 0.2) -> None:
        raise NotImplementedError

    def scroll(self, dx: int, dy: int) -> None:
        raise NotImplementedError

    def type_text(self, text: str) -> None:
        raise NotImplementedError

    def keypress(self, keys: Optional[list[str]]) -> None:
        raise NotImplementedError


class PyAutoGUIAdapter(PlatformAdapter):
    def __init__(self) -> None:
        if not pag:
            raise RuntimeError("pyautogui is not available")

    def capture(self) -> Screenshot:
        if mss:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.rgb)
                return Screenshot(img, shot.width, shot.height)
        shot = pag.screenshot()
        return Screenshot(shot, shot.width, shot.height)

    def screen_size(self) -> Tuple[int, int]:
        size = pag.size()
        return size.width, size.height

    def move(self, x: int, y: int) -> None:
        pag.moveTo(x, y, duration=0.1)

    def click(self, x: int, y: int, clicks: int = 1, button: str = "left") -> None:
        pag.click(x=x, y=y, clicks=clicks, button=button)

    def drag(self, start: Tuple[int, int], end: Tuple[int, int], duration: float = 0.2) -> None:
        pag.moveTo(start[0], start[1])
        pag.dragTo(end[0], end[1], duration=duration)

    def scroll(self, dx: int, dy: int) -> None:
        if dy:
            pag.scroll(dy)
        if dx:
            pag.hscroll(dx)

    def type_text(self, text: str) -> None:
        pag.typewrite(text, interval=0.02)

    def keypress(self, keys: Optional[list[str]]) -> None:
        if not keys:
            return
        for key in keys:
            pag.press(key)


class NullAdapter(PlatformAdapter):
    def __init__(self, size: Tuple[int, int] = (1280, 720)) -> None:
        self._size = size

    def capture(self) -> Screenshot:
        image = Image.new("RGB", self._size, color=(24, 24, 24))
        return Screenshot(image, self._size[0], self._size[1])

    def screen_size(self) -> Tuple[int, int]:
        return self._size

    def move(self, x: int, y: int) -> None:
        return None

    def click(self, x: int, y: int, clicks: int = 1, button: str = "left") -> None:
        return None

    def drag(self, start: Tuple[int, int], end: Tuple[int, int], duration: float = 0.2) -> None:
        return None

    def scroll(self, dx: int, dy: int) -> None:
        return None

    def type_text(self, text: str) -> None:
        return None

    def keypress(self, keys: Optional[list[str]]) -> None:
        return None


def default_adapter() -> PlatformAdapter:
    if pag:
        return PyAutoGUIAdapter()
    return NullAdapter()
