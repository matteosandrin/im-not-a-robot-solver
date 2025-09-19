"""
Din Don Dan Bot

This script solves the 47th level of https://neal.fun/not-a-robot/

The script captures a configurable rectangle of the screen, tracks average RGB
changes for the full area and each cell of an evenly divided grid, and
optionally shows a live OpenCV preview while sending debounced keyboard input
(arrow keys by default) whenever a region exceeds a tolerance threshold; run it
via `python screen_monitor.py --left <x> --top <y> --width <w> --height <h>
[options]` and use `--help` to review CLI flags for tuning rows/cols, tolerance,
debounce, FPS, and preview scaling.

Usage:
python din_don_dan.py \
    --preview \
    --left=70 \
    --top=700 \
    --width=440 \
    --height=42 \
    --cols=4


Dependencies:
 * mss
 * numpy
 * pillow
 * opencv-python
 * pynput

"""

from __future__ import annotations

import argparse
import asyncio
import math
import signal
import string
import sys
import threading
import time
from pathlib import Path
import numpy as np
import mss
import cv2
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pynput import keyboard

@dataclass(frozen=True)
class Region:
    name: str
    x: int
    y: int
    width: int
    height: int

    def as_slice(self) -> Tuple[slice, slice]:
        return (
            slice(self.y, self.y + self.height),
            slice(self.x, self.x + self.width),
        )

    def center(self) -> Tuple[int, int]:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )


class ScreenMonitor:
    WINDOW_NAME = "DIN DON DAN BOT"

    def __init__(self, *, left: int, top: int, width: int, height: int,
                 rows: int, cols: int, tolerance: float, trigger_delay: float,
                 debounce_interval: float, target_fps: float,
                 show_frame_stats: bool, preview: bool) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("Both rows and cols must be at least 1")
        if width % cols != 0 or height % rows != 0:
            raise ValueError("Capture width and height must be divisible by cols and rows")

        self.monitor = {"left": left, "top": top, "width": width, "height": height}
        self.regions = self._build_grid(rows, cols)
        self.tolerance = tolerance
        if self.tolerance < 0:
            raise ValueError("Tolerance must be non-negative")
        self.trigger_delay = trigger_delay
        if self.trigger_delay < 0:
            raise ValueError("Trigger delay must be non-negative")
        self.debounce_interval = debounce_interval
        if self.debounce_interval < 0:
            raise ValueError("Debounce interval must be non-negative")
        self.target_frame_time = 1.0 / target_fps if target_fps > 0 else 0
        self.show_frame_stats = show_frame_stats
        self.preview = preview

        self.reference_colors: Dict[str, Tuple[float, float, float]] = {}
        self.key_controller = keyboard.Controller()
        self.region_keys = self._build_region_keys()
        self.icon_directory = Path(__file__).resolve().parent / "icons"
        self.icon_cache: Dict[str, np.ndarray] = {}

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()
        self.last_trigger_times: Dict[str, float] = {}
        self.keypress_indicator_until: Dict[str, float] = {}
        self.keypress_indicator_duration = 0.25

        self._is_first_draw = False


    def _build_grid(self, rows: int, cols: int) -> List[Region]:
        regions: List[Region] = []
        cell_width = self.monitor["width"] // cols
        cell_height = self.monitor["height"] // rows

        for row in range(rows):
            for col in range(cols):
                regions.append(
                    Region(
                        name=f"r{row + 1}c{col + 1}",
                        x=col * cell_width,
                        y=row * cell_height,
                        width=cell_width,
                        height=cell_height,
                    )
                )
        return regions

    def _build_region_keys(self) -> Dict[str, "keyboard.KeyCode"]:
        mapping: Dict[str, "keyboard.KeyCode"] = {}
        arrows = [
            keyboard.Key.left,
            keyboard.Key.down,
            keyboard.Key.up,
            keyboard.Key.right,
        ]
        for idx, region in enumerate(self.regions):
            mapping[region.name] = arrows[idx]
        return mapping

    def _capture_reference(self, sct) -> None:
        raw = sct.grab(self.monitor)
        frame = self._to_numpy(raw)
        self.reference_colors.clear()
        self.reference_colors["full"] = average_rgb(frame)
        for region in self.regions:
            sub_frame = frame[region.as_slice()]
            self.reference_colors[region.name] = average_rgb(sub_frame)
            print(region, self.reference_colors[region.name])

    def _index_to_char(self, idx: int) -> str:
        if 1 <= idx <= 9:
            return str(idx)

        idx -= 9
        lowercase = string.ascii_lowercase
        if idx <= len(lowercase):
            return lowercase[idx - 1]

        idx -= len(lowercase)
        uppercase = string.ascii_uppercase
        if idx <= len(uppercase):
            return uppercase[idx - 1]

        raise ValueError(
            "Too many regions to assign unique keys automatically; reduce rows*cols."
        )

    def run(self) -> None:
        with mss.mss() as sct:
            self._capture_reference(sct)
            frame_counter = 0
            last_report = time.perf_counter()
            while True:
                start_time = time.perf_counter()
                raw = sct.grab(self.monitor)
                frame = self._to_numpy(raw)
                self.handle_frame(frame)
                if self.preview:
                    self._show_preview(frame)
                frame_counter += 1

                if self.show_frame_stats and frame_counter % 30 == 0:
                    now = time.perf_counter()
                    elapsed = now - last_report
                    fps = frame_counter / elapsed if elapsed else 0.0
                    print(f"[stats] {frame_counter} frames captured @ {fps:.2f} fps", flush=True)

                if self.target_frame_time:
                    elapsed = time.perf_counter() - start_time
                    remaining = self.target_frame_time - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

    def handle_frame(self, frame) -> None:
        full_avg = average_rgb(frame)
        full_change = self._detect_change("full", full_avg)
        print(f"full={int(full_change)}", end="")

        per_region: List[str] = []
        for region in self.regions:
            sub_frame = frame[region.as_slice()]
            avg = average_rgb(sub_frame)
            change = self._detect_change(region.name, avg)
            if change:
                per_region.append(f"{region.name}={int(change)}")
            else:
                per_region.append(f"{region.name}= ")
            if change:
                self._maybe_trigger_action(region.name)

        if per_region:
            print(" | " + ", ".join(per_region), end="")

        print("", flush=True)

    def _to_numpy(self, raw_capture):
        frame = np.asarray(raw_capture)
        frame = frame[..., :3][:, :, ::-1]
        return convert_to_black(frame)

    def _show_preview(self, frame) -> None:

        preview_frame = np.ascontiguousarray(frame.copy())
        x_scale = 1.0
        y_scale = 1.0

        now = time.perf_counter()
        active_regions = [
            region for region in self.regions
            if now < self.keypress_indicator_until.get(region.name, 0.0)
        ]

        icon_data: List[Tuple[Region, np.ndarray]] = []
        if active_regions:
            self._draw_keypress_indicators(preview_frame, active_regions, x_scale, y_scale)
            icon_data = self._collect_icon_data(active_regions, x_scale, y_scale)

        composed_frame = self._compose_preview_with_icons(preview_frame, icon_data, x_scale)

        bgr_frame = composed_frame[:, :, ::-1]
        cv2.imshow(self.WINDOW_NAME, bgr_frame)
        if not self._is_first_draw:
            cv2.moveWindow(self.WINDOW_NAME, 1200, 280 )
            self._is_first_draw = True
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt

    def close(self) -> None:
        if self.preview:
            cv2.destroyWindow(self.WINDOW_NAME)
        if hasattr(self, "loop"):
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join(timeout=1.0)
            if not self.loop.is_running():
                self.loop.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _detect_change(self, key: str, current: Tuple[float, float, float]) -> bool:
        reference = self.reference_colors.get(key)
        if reference is None:
            return False

        delta = color_distance(current, reference)
        return delta >= self.tolerance

    def _maybe_trigger_action(self, region_key: str) -> None:
        now = time.perf_counter()
        last = self.last_trigger_times.get(region_key, 0.0)
        if now - last < self.debounce_interval:
            return
        self.last_trigger_times[region_key] = now
        keycode = self.region_keys.get(region_key)
        if keycode is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._delayed_keypress(region_key, keycode),
            self.loop,
        )

    async def _delayed_keypress(self, region_key: str, keycode: "keyboard.KeyCode") -> None:
        await asyncio.sleep(self.trigger_delay)
        self._press_key(region_key, keycode)

    def _press_key(self, region_key: str, keycode: "keyboard.KeyCode") -> None:
        self.key_controller.press(keycode)
        self.key_controller.release(keycode)
        self.keypress_indicator_until[region_key] = (
            time.perf_counter() + self.keypress_indicator_duration
        )

    def _draw_keypress_indicators(
        self,
        frame,
        regions: List[Region],
        x_scale: float,
        y_scale: float,
    ) -> None:
        indicator_color = (0, 255, 0)
        scale_factor = max(x_scale, y_scale)
        rect_thickness = max(2, int(round(scale_factor)))
        margin = rect_thickness + 2
        for region in regions:
            top_left = (
                int(region.x * x_scale + margin),
                int(region.y * y_scale + margin),
            )
            bottom_right = (
                int((region.x + region.width) * x_scale - margin),
                int((region.y + region.height) * y_scale - margin),
            )
            cv2.rectangle(frame, top_left, bottom_right, indicator_color, rect_thickness)

    def _collect_icon_data(
        self,
        regions: List[Region],
        x_scale: float,
        y_scale: float,
    ) -> List[Tuple[Region, np.ndarray]]:
        icon_data: List[Tuple[Region, np.ndarray]] = []
        for region in regions:
            icon = self._region_key_icon(region.name)
            if icon is None:
                continue
            scaled = self._scale_icon_for_region(icon, region, x_scale, y_scale)
            icon_data.append((region, scaled))
        return icon_data

    def _compose_preview_with_icons(
        self,
        preview_frame: np.ndarray,
        icon_data: List[Tuple[Region, np.ndarray]],
        x_scale: float,
    ) -> np.ndarray:
        preview_height, preview_width = preview_frame.shape[:2]
        icon_strip_height = preview_height * 2
        combined_height = preview_height + icon_strip_height
        combined_frame = np.zeros((combined_height, preview_width, 3), dtype=preview_frame.dtype)
        combined_frame[:preview_height] = preview_frame

        if icon_strip_height > 0:
            separator_color = (0, 180, 0)
            cv2.line(
                combined_frame,
                (0, preview_height),
                (preview_width - 1, preview_height),
                separator_color,
                1,
            )

        if icon_data:
            self._draw_icon_strip(
                combined_frame,
                preview_height,
                icon_strip_height,
                icon_data,
                x_scale,
            )

        return combined_frame

    def _draw_icon_strip(
        self,
        frame: np.ndarray,
        preview_height: int,
        icon_strip_height: int,
        icon_data: List[Tuple[Region, np.ndarray]],
        x_scale: float,
    ) -> None:
        padding = 12
        strip_top = preview_height
        strip_bottom = preview_height + icon_strip_height
        frame_width = frame.shape[1]
        for region, icon in icon_data:
            icon_height, icon_width = icon.shape[:2]
            center_x = (region.x + region.width / 2.0) * x_scale
            icon_left = int(center_x - icon_width / 2.0)
            icon_left = max(padding, min(icon_left, frame_width - icon_width - padding))

            available_height = icon_strip_height - 2 * padding
            if available_height <= 0:
                icon_top = strip_top
            else:
                icon_top = strip_top + padding + max(0, (available_height - icon_height) // 2)
            icon_top = min(icon_top, strip_bottom - icon_height - padding)
            icon_top = max(strip_top + padding, icon_top)

            self._overlay_icon(frame, icon, icon_top, icon_left)

    def _region_key_icon(self, region_name: str) -> Optional[np.ndarray]:
        keycode = self.region_keys.get(region_name)
        path = self._icon_path_for_keycode(keycode)
        if path is None:
            return None
        try:
            return self._load_icon(path)
        except FileNotFoundError:
            return None

    def _icon_path_for_keycode(self, keycode) -> Optional[Path]:
        if keycode is None:
            return None

        if isinstance(keycode, keyboard.Key):
            filename_map = {
                keyboard.Key.left: "left.png",
                keyboard.Key.right: "right.png",
                keyboard.Key.up: "up.png",
                keyboard.Key.down: "down.png",
            }
            filename = filename_map.get(keycode)
            if not filename:
                return None
            return self.icon_directory / filename

        if hasattr(keycode, "char") and keycode.char:
            filename = f"{keycode.char.lower()}.png"
            return self.icon_directory / filename

        return None

    def _load_icon(self, path: Path) -> np.ndarray:
        cache_key = str(path)
        cached = self.icon_cache.get(cache_key)
        if cached is not None:
            return cached

        if not path.exists():
            raise FileNotFoundError(f"Icon not found: {path}")

        icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if icon is None:
            raise FileNotFoundError(f"Unable to load icon: {path}")

        if icon.shape[-1] == 3:
            rgb = cv2.cvtColor(icon, cv2.COLOR_BGR2RGB)
            alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
            icon = np.concatenate([rgb, alpha], axis=2)
        else:
            icon = cv2.cvtColor(icon, cv2.COLOR_BGRA2RGBA)

        self.icon_cache[cache_key] = icon
        return icon

    def _scale_icon_for_region(
        self,
        icon: np.ndarray,
        region: Region,
        x_scale: float,
        y_scale: float,
    ) -> np.ndarray:
        target_width = int(region.width * x_scale * 0.4)
        if target_width <= 0:
            return icon
        icon_height, icon_width = icon.shape[:2]
        if icon_width == 0:
            return icon
        scale = target_width / icon_width
        target_height = max(1, int(round(icon_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(icon, (target_width, target_height), interpolation=interpolation)

    def _overlay_icon(self, frame: np.ndarray, icon: np.ndarray, top: int, left: int) -> None:
        icon_height, icon_width = icon.shape[:2]
        bottom = min(frame.shape[0], top + icon_height)
        right = min(frame.shape[1], left + icon_width)
        icon_height = bottom - top
        icon_width = right - left
        if icon_height <= 0 or icon_width <= 0:
            return

        icon_slice = icon[:icon_height, :icon_width].astype(np.float32)
        rgb = icon_slice[..., :3]
        alpha = icon_slice[..., 3:] / 255.0

        roi = frame[top:bottom, left:right].astype(np.float32)
        blended = rgb * alpha + roi * (1.0 - alpha)
        frame[top:bottom, left:right] = blended.astype(np.uint8)


def convert_to_black(frame):
    if np is not None and isinstance(frame, np.ndarray):
        white_mask = np.all(frame == 255, axis=-1)
        frame[~white_mask] = 0
        frame[white_mask] = 255
        return frame

    try:
        iterator = frame.getdata()
        pixels = [
            (255, 255, 255) if tuple(pixel[:3]) == (255, 255, 255) else (0, 0, 0)
            for pixel in iterator
        ]
        frame.putdata(pixels)
        return frame
    except AttributeError:
        pass

    try:
        return [
            (255, 255, 255) if tuple(pixel[:3]) == (255, 255, 255) else (0, 0, 0)
            for pixel in frame
        ]
    except TypeError:
        return frame


def average_rgb(frame) -> Tuple[float, float, float]:
    if np is not None and isinstance(frame, np.ndarray):
        return tuple(frame.mean(axis=(0, 1)))

    if hasattr(frame, "getdata"):
        pixels = list(frame.getdata())
    else:
        pixels = frame

    r_total = g_total = b_total = 0
    count = 0
    for r, g, b in pixels:
        r_total += r
        g_total += g
        b_total += b
        count += 1

    if count == 0:
        return (0.0, 0.0, 0.0)

    return (r_total / count, g_total / count, b_total / count)


def np_from_image(image):
    return np.asarray(image)


def color_distance(color: Tuple[float, float, float],
                   other: Tuple[float, float, float]) -> float:
    dr = float(color[0]) - other[0]
    dg = float(color[1]) - other[1]
    db = float(color[2]) - other[2]
    return math.sqrt(dr * dr + dg * dg + db * db)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor a rectangular screen region, indicating whether the full\n"
            "area and each cell in an evenly divided grid deviate from their\n"
            "reference startup color beyond a tolerance."
        )
    )
    parser.add_argument("--left", type=int, required=True, help="Left coordinate of the capture area")
    parser.add_argument("--top", type=int, required=True, help="Top coordinate of the capture area")
    parser.add_argument("--width", type=int, required=True, help="Width of the capture area")
    parser.add_argument("--height", type=int, required=True, help="Height of the capture area")
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of rows to divide the capture area",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=1,
        help="Number of columns to divide the capture area",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Euclidean distance in RGB space that counts as a deviation",
    )
    parser.add_argument(
        "--trigger-delay",
        type=float,
        default=0.2,
        help="Delay in seconds before sending the keypress after a trigger",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=0.2,
        help="Minimum seconds between keypress schedules per region",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target frames per second; set to 0 to disable frame pacing",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print periodic capture statistics",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live OpenCV window of the captured region",
    )
    return parser.parse_args(argv)


def install_signal_handlers() -> None:
    def handle_sigint(signum, frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    install_signal_handlers()

    monitor = ScreenMonitor(
        left=args.left,
        top=args.top,
        width=args.width,
        height=args.height,
        rows=args.rows,
        cols=args.cols,
        tolerance=args.tolerance,
        trigger_delay=args.trigger_delay,
        debounce_interval=args.debounce,
        target_fps=args.fps,
        show_frame_stats=args.stats,
        preview=args.preview,
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        monitor.close()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
