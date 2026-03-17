from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


LEFT_KEYS = {2424832, 81, ord("a"), ord("A")}
RIGHT_KEYS = {2555904, 83, ord("d"), ord("D")}
UP_KEYS = {2490368, 82, ord("w"), ord("W")}
DOWN_KEYS = {2621440, 84, ord("s"), ord("S")}
HOME_KEYS = {2359296, ord("h"), ord("H")}
END_KEYS = {2293760, ord("e"), ord("E")}


@dataclass
class FrameSelection:
    frame: Optional[np.ndarray]
    frame_index: int
    total_frames: int


@dataclass
class MultiFrameSelection:
    frames: list[Optional[np.ndarray]]
    indices: list[int]


def load_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    return cap


def select_frame(video_path: str, window_name: str = "Frame Selection") -> FrameSelection:
    cap = load_video(video_path)
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    idx = 0
    selected = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        canvas = frame.copy()
        cv2.putText(canvas, f"Frame {idx + 1}/{total_frames} ({idx / fps:.2f}s)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 255, 20), 2)
        cv2.putText(canvas, "Arrows/A-D/W-S : navigate | H/E : start/end | Enter/Space : validate | ESC : cancel", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKeyEx(0)

        if key == 27:
            selected = None
            break
        if key in (13, 32):
            selected = frame.copy()
            break
        if key in LEFT_KEYS:
            idx = max(0, idx - 1)
            continue
        if key in RIGHT_KEYS:
            idx = min(total_frames - 1, idx + 1)
            continue
        if key in UP_KEYS:
            idx = min(total_frames - 1, idx + 10)
            continue
        if key in DOWN_KEYS:
            idx = max(0, idx - 10)
            continue
        if key in HOME_KEYS:
            idx = 0
            continue
        if key in END_KEYS:
            idx = total_frames - 1
            continue

    cap.release()
    cv2.destroyWindow(window_name)
    return FrameSelection(frame=selected, frame_index=idx, total_frames=total_frames)


def read_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    cap = load_video(video_path)
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    safe_idx = max(0, min(total_frames - 1, frame_index))
    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def _fit_for_grid(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    interpolation = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(frame, (nw, nh), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _fit_for_grid_with_meta(frame: np.ndarray, target_w: int, target_h: int) -> tuple[np.ndarray, dict[str, float]]:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        return canvas, {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "src_w": float(w), "src_h": float(h)}

    scale = min(target_w / w, target_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    interpolation = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(frame, (nw, nh), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas, {"x": float(x), "y": float(y), "w": float(nw), "h": float(nh), "src_w": float(w), "src_h": float(h)}


def _draw_text_with_bg(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.8, color: tuple[int, int, int] = (255, 255, 255), thickness: int = 2) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def select_frames_grid(video_paths: list[str], titles: list[str], step_name: str) -> MultiFrameSelection:
    if len(video_paths) != 4 or len(titles) != 4:
        raise ValueError("select_frames_grid expects exactly 4 videos and 4 titles")

    caps = [load_video(path) for path in video_paths]
    totals = [max(1, int(c.get(cv2.CAP_PROP_FRAME_COUNT))) for c in caps]
    indices = [0, 0, 0, 0]
    active = 0

    win = f"Selection 2x2 - {step_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1820, 1020)

    tile_w = 860
    tile_h = 460
    header_h = 56
    current_grid_w = tile_w * 2
    current_grid_h = header_h + tile_h * 2 + 64

    def _window_to_grid(mx: int, my: int) -> tuple[int, int]:
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(win)
        except Exception:
            win_w, win_h = current_grid_w, current_grid_h
        if win_w <= 0 or win_h <= 0:
            return mx, my
        gx = int((mx / float(win_w)) * current_grid_w)
        gy = int((my / float(win_h)) * current_grid_h)
        return gx, gy

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal active
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        gx, gy = _window_to_grid(x, y)
        y_core = gy - header_h
        if y_core < 0 or y_core >= tile_h * 2:
            return
        if gx < 0 or gx >= tile_w * 2:
            return

        col = 0 if gx < tile_w else 1
        row = 0 if y_core < tile_h else 1
        panel = row * 2 + col
        if 0 <= panel <= 3:
            active = panel

    cv2.setMouseCallback(win, on_mouse)

    def build_tile(i: int) -> np.ndarray:
        cap = caps[i]
        idx = max(0, min(totals[i] - 1, indices[i]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        border = (0, 255, 255) if i == active else (90, 90, 90)
        tile = _fit_for_grid(frame, tile_w, tile_h)
        cv2.rectangle(tile, (1, 1), (tile.shape[1] - 2, tile.shape[0] - 2), border, 3)
        _draw_text_with_bg(tile, f"[{i + 1}] {titles[i]}", (12, 34), scale=0.85)
        _draw_text_with_bg(tile, f"Frame {idx + 1}/{totals[i]}", (12, 72), scale=0.75, color=(180, 255, 180))
        return tile

    while True:
        tiles = [build_tile(i) for i in range(4)]
        top = np.hstack([tiles[0], tiles[1]])
        bottom = np.hstack([tiles[2], tiles[3]])
        core = np.vstack([top, bottom])

        header = np.zeros((56, core.shape[1], 3), dtype=np.uint8)
        footer = np.zeros((64, core.shape[1], 3), dtype=np.uint8)
        _draw_text_with_bg(header, f"{step_name} - frame selection", (16, 40), scale=0.95, color=(0, 255, 255))
        _draw_text_with_bg(
            footer,
            "Click on a video to make it active | Arrows/A-D: +/-1 | W/S: +/-10 | H/E: start/end | +/-: zoom | Enter: validate",
            (16, 44),
            scale=0.68,
            color=(0, 255, 0),
            thickness=2,
        )
        grid = np.vstack([header, core, footer])
        current_grid_h, current_grid_w = grid.shape[:2]
        cv2.imshow(win, grid)

        key = cv2.waitKeyEx(20)
        if key == -1:
            continue
        if key == 27:
            for c in caps:
                c.release()
            cv2.destroyWindow(win)
            return MultiFrameSelection(frames=[None, None, None, None], indices=indices)
        if key in (13, 32):
            break
        if key in (ord("+"), ord("=")):
            tile_w = min(1120, int(tile_w * 1.08))
            tile_h = min(640, int(tile_h * 1.08))
            continue
        if key in (ord("-"), ord("_")):
            tile_w = max(520, int(tile_w * 0.92))
            tile_h = max(300, int(tile_h * 0.92))
            continue
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            active = int(chr(key)) - 1
            continue
        if key in LEFT_KEYS:
            indices[active] = max(0, indices[active] - 1)
            continue
        if key in RIGHT_KEYS:
            indices[active] = min(totals[active] - 1, indices[active] + 1)
            continue
        if key in UP_KEYS:
            indices[active] = min(totals[active] - 1, indices[active] + 10)
            continue
        if key in DOWN_KEYS:
            indices[active] = max(0, indices[active] - 10)
            continue
        if key in HOME_KEYS:
            indices[active] = 0
            continue
        if key in END_KEYS:
            indices[active] = totals[active] - 1
            continue

    frames: list[Optional[np.ndarray]] = []
    for i, cap in enumerate(caps):
        idx = max(0, min(totals[i] - 1, indices[i]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames.append(frame.copy() if ok else None)
        cap.release()

    cv2.destroyWindow(win)
    return MultiFrameSelection(frames=frames, indices=indices)


def measure_lengths_grid(
    frames: list[np.ndarray],
    titles: list[str],
    labels: list[str],
    step_name: str,
    scales_cm_per_px: list[float] | None = None,
) -> list[float] | None:
    if len(frames) != 4 or len(titles) != 4 or len(labels) != 4:
        raise ValueError("measure_lengths_grid expects exactly 4 frames, 4 titles and 4 labels")

    win = f"Measurements 2x2 - {step_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1900, 1080)

    tile_w = 900
    tile_h = 500
    points: list[list[tuple[int, int]]] = [[], [], [], []]
    active = 0
    warning = ""

    latest_meta: list[dict[str, float]] = []
    header_h = 62
    footer_h = 78

    def _point_in_panel(mouse_x: int, mouse_y: int) -> tuple[int, int, int] | None:
        if not latest_meta:
            return None
        y_core = mouse_y - header_h
        if y_core < 0 or y_core >= tile_h * 2:
            return None
        col = 0 if mouse_x < tile_w else 1
        row = 0 if y_core < tile_h else 1
        panel = row * 2 + col
        x_local = mouse_x - (col * tile_w)
        y_local = y_core - (row * tile_h)
        if x_local < 0 or y_local < 0 or x_local >= tile_w or y_local >= tile_h:
            return None
        return panel, x_local, y_local

    def _local_to_frame(panel: int, x_local: int, y_local: int) -> tuple[int, int] | None:
        meta = latest_meta[panel]
        x0 = int(meta["x"])
        y0 = int(meta["y"])
        nw = int(meta["w"])
        nh = int(meta["h"])
        if nw <= 0 or nh <= 0:
            return None
        if not (x0 <= x_local < x0 + nw and y0 <= y_local < y0 + nh):
            return None

        fx = (x_local - x0) * (meta["src_w"] / meta["w"])
        fy = (y_local - y0) * (meta["src_h"] / meta["h"])
        frame = frames[panel]
        px = int(max(0, min(frame.shape[1] - 1, round(fx))))
        py = int(max(0, min(frame.shape[0] - 1, round(fy))))
        return px, py

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal active, warning
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pos = _point_in_panel(x, y)
        if pos is None:
            return
        panel, x_local, y_local = pos
        mapped = _local_to_frame(panel, x_local, y_local)
        if mapped is None:
            return
        active = panel
        warning = ""
        if len(points[panel]) == 0:
            points[panel] = [mapped]
        elif len(points[panel]) == 1:
            points[panel].append(mapped)
        else:
            points[panel] = [mapped]

    cv2.setMouseCallback(win, on_mouse)

    while True:
        tiles: list[np.ndarray] = []
        latest_meta = []
        for i in range(4):
            tile, meta = _fit_for_grid_with_meta(frames[i], tile_w, tile_h)
            latest_meta.append(meta)
            border = (0, 255, 255) if i == active else (120, 120, 120)
            cv2.rectangle(tile, (2, 2), (tile.shape[1] - 3, tile.shape[0] - 3), border, 4)
            _draw_text_with_bg(tile, f"[{i + 1}] {titles[i]} - {labels[i]}", (12, 44), scale=1.0, thickness=3)

            pts = points[i]
            if len(pts) >= 1:
                cv2.circle(tile, (int(meta["x"] + pts[0][0] * (meta["w"] / max(1.0, meta["src_w"]))), int(meta["y"] + pts[0][1] * (meta["h"] / max(1.0, meta["src_h"]))),), 8, (0, 255, 255), -1)
            if len(pts) >= 2:
                p1 = (
                    int(meta["x"] + pts[0][0] * (meta["w"] / max(1.0, meta["src_w"]))),
                    int(meta["y"] + pts[0][1] * (meta["h"] / max(1.0, meta["src_h"]))),
                )
                p2 = (
                    int(meta["x"] + pts[1][0] * (meta["w"] / max(1.0, meta["src_w"]))),
                    int(meta["y"] + pts[1][1] * (meta["h"] / max(1.0, meta["src_h"]))),
                )
                cv2.circle(tile, p2, 8, (0, 255, 255), -1)
                cv2.line(tile, p1, p2, (0, 255, 255), 3)
                dist_px = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
                text = f"{dist_px:.1f} px"
                if scales_cm_per_px is not None and i < len(scales_cm_per_px):
                    text += f" | {dist_px * scales_cm_per_px[i]:.2f} cm"
                _draw_text_with_bg(tile, text, (12, 88), scale=0.88, color=(160, 255, 160), thickness=2)

            tiles.append(tile)

        top = np.hstack([tiles[0], tiles[1]])
        bottom = np.hstack([tiles[2], tiles[3]])
        core = np.vstack([top, bottom])
        header = np.zeros((header_h, core.shape[1], 3), dtype=np.uint8)
        footer = np.zeros((footer_h, core.shape[1], 3), dtype=np.uint8)

        _draw_text_with_bg(header, f"{step_name} - simultaneous measurements (2 clicks per video)", (14, 44), scale=1.0, color=(0, 255, 255), thickness=2)
        line1 = "Click directly in each video to place 2 points"
        line2 = "1-4: active | R: reset active | C: reset all | +/-: zoom | Enter: validate | ESC: cancel"
        _draw_text_with_bg(footer, line1, (12, 34), scale=0.8, color=(0, 255, 0), thickness=2)
        _draw_text_with_bg(footer, line2, (12, 68), scale=0.76, color=(0, 255, 0), thickness=2)
        if warning:
            _draw_text_with_bg(footer, warning, (970, 34), scale=0.8, color=(0, 120, 255), thickness=2)

        cv2.imshow(win, np.vstack([header, core, footer]))
        key = cv2.waitKeyEx(20)

        if key == 27:
            cv2.destroyWindow(win)
            return None
        if key in (13, 32):
            if all(len(p) == 2 for p in points):
                break
            warning = "Incomplete measure: 2 points needed on each video."
            continue
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            active = int(chr(key)) - 1
            warning = ""
            continue
        if key in (ord("r"), ord("R")):
            points[active] = []
            warning = ""
            continue
        if key in (ord("c"), ord("C")):
            points = [[], [], [], []]
            warning = ""
            continue
        if key in (ord("+"), ord("=")):
            tile_w = min(1200, int(tile_w * 1.08))
            tile_h = min(700, int(tile_h * 1.08))
            continue
        if key in (ord("-"), ord("_")):
            tile_w = max(540, int(tile_w * 0.92))
            tile_h = max(320, int(tile_h * 0.92))
            continue

    cv2.destroyWindow(win)
    return [float(np.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1])) for p in points]
