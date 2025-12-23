from __future__ import annotations

import base64
import importlib.util
import io
import time
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
    dpi: tuple[float, float] = (96.0, 96.0)

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

    # Updated signature: repeat + hold_ms are optional and default-safe.
    def keypress(self, keys: Optional[list[str]], repeat: int = 1, hold_ms: int = 0) -> None:
        raise NotImplementedError


def _normalize_key_name(key: str) -> str:
    """
    Normalize common key synonyms to what pyautogui expects.
    This stays intentionally small to avoid breaking existing behavior.
    """
    k = (key or "").strip()
    if not k:
        return k
    k_upper = k.upper()

    mapping = {
        "CTRL": "ctrl",
        "CONTROL": "ctrl",
        "ALT": "alt",
        "SHIFT": "shift",
        "ENTER": "enter",
        "RETURN": "enter",
        "ESC": "esc",
        "ESCAPE": "esc",
        "TAB": "tab",
        "WIN": "win",
        "WINDOWS": "win",
        "CMD": "command",   # mac
        "COMMAND": "command",
        "META": "command",  # mac-ish synonym
        "BACKSPACE": "backspace",
        "DEL": "delete",
        "DELETE": "delete",
        "SPACE": "space",
        "UP": "up",
        "DOWN": "down",
        "LEFT": "left",
        "RIGHT": "right",
        "HOME": "home",
        "END": "end",
        "PGUP": "pageup",
        "PAGEUP": "pageup",
        "PGDN": "pagedown",
        "PAGEDOWN": "pagedown",
    }

    # Letters/numbers should be lower-case for pyautogui, e.g. "L" -> "l"
    if len(k) == 1:
        return k.lower()

    return mapping.get(k_upper, k.lower())


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
                dpi = img.info.get("dpi", (96.0, 96.0))
                return Screenshot(img, shot.width, shot.height, dpi=dpi)
        shot = pag.screenshot()
        dpi = shot.info.get("dpi", (96.0, 96.0))
        return Screenshot(shot, shot.width, shot.height, dpi=dpi)

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

    def keypress(self, keys: Optional[list[str]], repeat: int = 1, hold_ms: int = 0) -> None:
        """
        Supports combos like ["ctrl","l"] using pag.hotkey.
        - repeat: press the combo N times
        - hold_ms: optional small hold before releasing modifiers (best-effort)
        """
        if not keys:
            return

        # Clamp repeat defensively
        if repeat < 1:
            repeat = 1
        if repeat > 10:
            repeat = 10
        if hold_ms < 0:
            hold_ms = 0
        if hold_ms > 2000:
            hold_ms = 2000

        norm = [_normalize_key_name(k) for k in keys if k and str(k).strip()]
        norm = [k for k in norm if k]
        if not norm:
            return

        for _ in range(repeat):
            if len(norm) == 1:
                pag.press(norm[0])
            else:
                # If hold_ms requested, do a manual keyDown/keyUp sequence.
                if hold_ms > 0:
                    # press down all but last key as modifiers, then press last, then release
                    mods = norm[:-1]
                    last = norm[-1]
                    for m in mods:
                        pag.keyDown(m)
                    pag.press(last)
                    time.sleep(hold_ms / 1000.0)
                    for m in reversed(mods):
                        pag.keyUp(m)
                else:
                    pag.hotkey(*norm)

            time.sleep(0.03)  # tiny spacing to avoid OS dropping events


class NullAdapter(PlatformAdapter):
    def __init__(self, size: Tuple[int, int] = (1280, 720)) -> None:
        self._size = size

    def capture(self) -> Screenshot:
        image = Image.new("RGB", self._size, color=(24, 24, 24))
        return Screenshot(image, self._size[0], self._size[1], dpi=(96.0, 96.0))

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

    def keypress(self, keys: Optional[list[str]], repeat: int = 1, hold_ms: int = 0) -> None:
        return None


def default_adapter() -> PlatformAdapter:
    if pag:
        return PyAutoGUIAdapter()
    return NullAdapter()
