from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import numpy as np

import cv2

from heartvolume.core.calculations import (
    build_doppler_result,
    build_volume_result,
    calculate_volume_simple,
    calculate_volume_simpson,
)
from heartvolume.data.discovery import auto_detect_images, auto_detect_videos
from heartvolume.imaging.measurement import measure_distance
from heartvolume.imaging.scale_detection import (
    detect_scale_on_frame,
    detect_scale_with_details,
    draw_scale_detection,
)
from heartvolume.imaging.video_tools import read_frame, select_frame, select_frames_grid, measure_lengths_grid


class HeartVolumeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HeartVolume - Echocardiography Analyzer")
        self.root.geometry("980x720")

        self._build_header()
        self._build_tabs()
        self._build_output()
        self._autofill_paths()

    def _build_header(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Echocardiographic Analysis (Simple / Simpson / Doppler)", font=("Segoe UI", 13, "bold")).pack(side="left")
        ttk.Button(top, text="Auto-detect data/", command=self._autofill_paths).pack(side="right")

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=6)

        self.simple_tab = ttk.Frame(self.notebook, padding=10)
        self.simpson_tab = ttk.Frame(self.notebook, padding=10)
        self.doppler_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.simple_tab, text="Simple Method")
        self.notebook.add(self.simpson_tab, text="Modified Simpson")
        self.notebook.add(self.doppler_tab, text="Doppler")

        self._build_simple_tab()
        self._build_simpson_tab()
        self._build_doppler_tab()

    def _build_output(self) -> None:
        block = ttk.LabelFrame(self.root, text="Results", padding=8)
        block.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        self.output = tk.Text(block, height=14, wrap="word")
        self.output.pack(fill="both", expand=True)
        self._log("Ready. Select a method and click 'Run'.")

    def _row_file_selector(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
        ttk.Label(parent, text=label, width=18).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var, width=75).grid(row=row, column=1, sticky="ew", pady=4, padx=4)
        ttk.Button(parent, text="Browse", command=lambda: self._browse_into(var, filetypes)).grid(row=row, column=2, padx=4, pady=4)

    def _build_simple_tab(self) -> None:
        self.simple_video_var = tk.StringVar()
        self.simple_hr_var = tk.StringVar(value="60")

        self.simple_tab.columnconfigure(1, weight=1)
        self._row_file_selector(
            self.simple_tab,
            0,
            "Apical Video:",
            self.simple_video_var,
            [("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")],
        )
        ttk.Label(self.simple_tab, text="HR (bpm, optional):", width=18).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(self.simple_tab, textvariable=self.simple_hr_var, width=20).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Button(self.simple_tab, text="Run Simple Method", command=self.run_simple).grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)

    def _build_simpson_tab(self) -> None:
        self.simpson_apical_var = tk.StringVar()
        self.simpson_mv_var = tk.StringVar()
        self.simpson_pm_var = tk.StringVar()
        self.simpson_apex_var = tk.StringVar()
        self.simpson_hr_var = tk.StringVar(value="60")
        self.simpson_tick_cm_var = tk.StringVar(value="1.0")

        self.simpson_tab.columnconfigure(1, weight=1)
        self._row_file_selector(self.simpson_tab, 0, "Apical Video:", self.simpson_apical_var, [("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")])
        self._row_file_selector(self.simpson_tab, 1, "MV Video:", self.simpson_mv_var, [("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")])
        self._row_file_selector(self.simpson_tab, 2, "PM Video:", self.simpson_pm_var, [("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")])
        self._row_file_selector(self.simpson_tab, 3, "Apex Video:", self.simpson_apex_var, [("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")])

        ttk.Label(self.simpson_tab, text="HR (bpm, optional):").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(self.simpson_tab, textvariable=self.simpson_hr_var, width=20).grid(row=4, column=1, sticky="w", pady=4)
        ttk.Label(self.simpson_tab, text="Scale tick dist (cm):").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(self.simpson_tab, textvariable=self.simpson_tick_cm_var, width=20).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Button(self.simpson_tab, text="Run Modified Simpson", command=self.run_simpson).grid(row=6, column=0, columnspan=3, sticky="ew", pady=10)

    def _build_doppler_tab(self) -> None:
        self.doppler_image_var = tk.StringVar()
        self.doppler_video_var = tk.StringVar()
        self.doppler_vti_var = tk.StringVar()
        self.doppler_hr_var = tk.StringVar(value="60")

        self.doppler_tab.columnconfigure(1, weight=1)
        self._row_file_selector(self.doppler_tab, 0, "D_AO Image:", self.doppler_image_var, [("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")])
        self._row_file_selector(self.doppler_tab, 1, "D_AO Video:", self.doppler_video_var, [("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")])

        ttk.Label(self.doppler_tab, text="VTI (cm):", width=18).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(self.doppler_tab, textvariable=self.doppler_vti_var, width=20).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(self.doppler_tab, text="HR (bpm):", width=18).grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(self.doppler_tab, textvariable=self.doppler_hr_var, width=20).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Button(self.doppler_tab, text="Run Doppler", command=self.run_doppler).grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)

    def _browse_into(self, var: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _autofill_paths(self) -> None:
        videos = auto_detect_videos()
        images = auto_detect_images()

        if videos.get("apical"):
            self.simple_video_var.set(videos["apical"])
            self.simpson_apical_var.set(videos["apical"])
        if videos.get("mv"):
            self.simpson_mv_var.set(videos["mv"])
        if videos.get("pm"):
            self.simpson_pm_var.set(videos["pm"])
        if videos.get("apex"):
            self.simpson_apex_var.set(videos["apex"])
        if images.get("d_ao"):
            self.doppler_image_var.set(images["d_ao"])
        if videos.get("d_ao"):
            self.doppler_video_var.set(videos["d_ao"])

        self._log("Auto-detection finished.")

    def _log(self, text: str) -> None:
        self.output.insert("end", text + "\n")
        self.output.see("end")

    @staticmethod
    def _optional_float(value: str) -> float | None:
        value = value.strip()
        if not value:
            return None
        return float(value)

    def _calibrate_scale(self, frame) -> float | None:
        detected = detect_scale_on_frame(frame, cm_per_tick=1.0)
        if detected is not None:
            px_per_cm = 1.0 / detected if detected > 0 else 0
            use_auto = messagebox.askyesno(
                "Scale Detected",
                f"Automatically detected scale: {detected:.6f} cm/pixel\n({px_per_cm:.1f} pixels per cm)\n\nUse this value?",
            )
            if use_auto:
                return detected

            tick_cm = simpledialog.askfloat("Tick Distance", "Real distance between two ticks (cm):", initialvalue=1.0, minvalue=0.01)
            if tick_cm is not None:
                corrected = detect_scale_on_frame(frame, cm_per_tick=tick_cm)
                if corrected is not None:
                    return corrected
                messagebox.showwarning("Scale", "Automatic detection failed with this tick distance. Manual calibration.")

        messagebox.showinfo("Manual Calibration", "Click 2 points of known distance, then enter the distance in cm.")
        m = measure_distance(frame, "Calibration")
        if m is None or not m.is_complete():
            return None
        real_cm = simpledialog.askfloat("Real Distance", "Distance between the 2 points (cm):", minvalue=0.01)
        if real_cm is None:
            return None
        if m.distance_pixels() <= 0:
            return None
        return real_cm / m.distance_pixels()

    def _measure_cm_on_video(self, video_path: str, label: str, scale_cm_per_pixel: float | None = None) -> tuple[float | None, float | None]:
        if not os.path.isfile(video_path):
            messagebox.showerror("File", f"Video not found:\n{video_path}")
            return None, scale_cm_per_pixel

        selected = select_frame(video_path, window_name=f"Frame Selection - {label}")
        if selected.frame is None:
            return None, scale_cm_per_pixel

        scale = scale_cm_per_pixel
        if scale is None:
            scale = self._calibrate_scale(selected.frame)
            if scale is None:
                return None, scale_cm_per_pixel

        m = measure_distance(selected.frame, label=label, window_name=f"Measure - {label}")
        if m is None or not m.is_complete():
            return None, scale

        return m.distance_pixels() * scale, scale

    def _require_float(self, value: str, field_name: str) -> float:
        try:
            return float(value.strip())
        except Exception as exc:
            raise ValueError(f"Invalid value for {field_name}") from exc

    def run_simple(self) -> None:
        self._log("--- Simple Method ---")
        try:
            path = self.simple_video_var.get().strip()
            if not path:
                raise ValueError("Select the apical video")

            l_ed, scale = self._measure_cm_on_video(path, "Lendo_ED", None)
            if l_ed is None:
                return
            d_ed, scale = self._measure_cm_on_video(path, "Dendo_ED", scale)
            if d_ed is None:
                return
            l_es, scale = self._measure_cm_on_video(path, "Lendo_ES", scale)
            if l_es is None:
                return
            d_es, _ = self._measure_cm_on_video(path, "Dendo_ES", scale)
            if d_es is None:
                return

            edv = calculate_volume_simple(l_ed, d_ed)
            esv = calculate_volume_simple(l_es, d_es)
            hr = self._optional_float(self.simple_hr_var.get())
            result = build_volume_result(edv, esv, hr)

            self._log(f"Lendo ED={l_ed:.2f} cm | Dendo ED={d_ed:.2f} cm")
            self._log(f"Lendo ES={l_es:.2f} cm | Dendo ES={d_es:.2f} cm")
            self._log(f"EDV={result.edv:.2f} ml | ESV={result.esv:.2f} ml | SV={result.sv:.2f} ml | EF={result.ef:.1f}%")
            if result.co is not None:
                self._log(f"CO={result.co:.2f} L/min")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    @staticmethod
    def _default_scale_roi(frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        # Keep GUI default ROI aligned with the new core detector: fixed 50px on right edge.
        return (max(0, w - 50), int(h * 0.03), w, int(h * 0.95))

    @staticmethod
    def _clamp_roi(roi: tuple[int, int, int, int], frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi
        x0 = max(0, min(w - 2, x0))
        y0 = max(0, min(h - 2, y0))
        x1 = max(x0 + 2, min(w, x1))
        y1 = max(y0 + 2, min(h, y1))
        return (x0, y0, x1, y1)

    def _update_roi(self, roi: tuple[int, int, int, int], key: int, frame: np.ndarray) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = roi
        dx = max(2, int(frame.shape[1] * 0.01))
        dy = max(2, int(frame.shape[0] * 0.01))

        # Movement
        if key in (ord("j"), ord("J")):
            x0 -= dx
            x1 -= dx
        elif key in (ord("l"), ord("L")):
            x0 += dx
            x1 += dx
        elif key in (ord("i"), ord("I")):
            y0 -= dy
            y1 -= dy
        elif key in (ord("k"), ord("K")):
            y0 += dy
            y1 += dy
        # Resizing
        elif key in (ord("u"), ord("U")):
            x1 = max(x0 + 20, x1 - dx)
        elif key in (ord("o"), ord("O")):
            x1 += dx
        elif key in (ord("y"), ord("Y")):
            y1 = max(y0 + 20, y1 - dy)
        elif key in (ord("h"), ord("H")):
            y1 += dy

        return self._clamp_roi((x0, y0, x1, y1), frame)

    def _preview_scales_grid(self, video_paths: list[str], titles: list[str], step_name: str, cm_per_tick: float) -> list[float] | None:
        frames: list[np.ndarray] = []
        for i, path in enumerate(video_paths):
            frame = read_frame(path, 0)
            if frame is None:
                raise ValueError(f"Unable to read frame 1 of {titles[i]}")
            frames.append(frame)

        rois = [self._default_scale_roi(frame) for frame in frames]
        active = 0
        zoomed_panel: int | None = None

        def fit_for_grid(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                return np.zeros((target_h, target_w, 3), dtype=np.uint8)
            ratio = min(target_w / w, target_h / h)
            nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
            interpolation = cv2.INTER_LINEAR if ratio >= 1.0 else cv2.INTER_AREA
            resized = cv2.resize(frame, (nw, nh), interpolation=interpolation)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x0 = (target_w - nw) // 2
            y0 = (target_h - nh) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized
            return canvas

        def draw_text_with_bg(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.9, color: tuple[int, int, int] = (255, 255, 255), thickness: int = 2) -> None:
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            x, y = org
            cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + baseline + 6), (0, 0, 0), -1)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        tile_w = 900
        tile_h = 500
        win = f"Auto scale 2x2 - {step_name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1900, 1080)

        header_h = 62
        footer_h = 74
        current_grid_w = tile_w * 2
        current_grid_h = header_h + tile_h * 2 + footer_h

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

        def _panel_from_grid(gx: int, gy: int) -> int | None:
            y_core = gy - header_h
            if y_core < 0 or y_core >= tile_h * 2:
                return None
            if gx < 0 or gx >= tile_w * 2:
                return None
            if zoomed_panel is not None:
                return zoomed_panel
            col = 0 if gx < tile_w else 1
            row = 0 if y_core < tile_h else 1
            panel = row * 2 + col
            return panel if 0 <= panel <= 3 else None

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            nonlocal active, zoomed_panel
            if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
                return
            gx, gy = _window_to_grid(x, y)
            panel = _panel_from_grid(gx, gy)
            if panel is None:
                return

            active = panel
            if event == cv2.EVENT_LBUTTONDBLCLK:
                # Toggle single-panel fullscreen inside the same window.
                if zoomed_panel == panel:
                    zoomed_panel = None
                else:
                    zoomed_panel = panel

        cv2.setMouseCallback(win, on_mouse)

        final_details = [None, None, None, None]
        warning = ""
        while True:
            overlays: list[np.ndarray] = []
            details_list = []
            for i in range(4):
                # Main app now uses exactly the core method implemented in scale_detection.py
                details = detect_scale_with_details(frames[i], cm_per_tick=cm_per_tick, roi_override=rois[i])
                details_list.append(details)
                overlay = draw_scale_detection(frames[i], details, title=f"{titles[i]} - frame1")
                overlays.append(overlay)

            tiles = [fit_for_grid(img, tile_w, tile_h) for img in overlays]
            for i, tile in enumerate(tiles):
                details = details_list[i]
                ok = details.cm_per_pixel is not None
                color = (0, 255, 0) if ok else (0, 0, 255)
                border = (0, 255, 255) if i == active else (120, 120, 120)
                cv2.rectangle(tile, (2, 2), (tile.shape[1] - 3, tile.shape[0] - 3), border, 4)
                draw_text_with_bg(tile, f"[{i + 1}] {titles[i]}", (12, 40), scale=1.05, color=(255, 255, 255), thickness=3)

                if ok and details.cm_per_pixel is not None:
                    px_per_cm = 1.0 / details.cm_per_pixel
                    draw_text_with_bg(tile, f"{details.cm_per_pixel:.6f} cm/px  ({px_per_cm:.1f} px/cm)", (12, 84), scale=0.9, color=(150, 255, 150), thickness=2)
                else:
                    draw_text_with_bg(tile, "Auto scale: not detected", (12, 84), scale=0.9, color=(120, 120, 255), thickness=2)

                visible = details.visible_cm if details.visible_cm is not None else 0.0
                draw_text_with_bg(tile, f"Estimated visible ruler length: {visible:.2f} cm", (12, 122), scale=0.88, color=color, thickness=2)

            top = np.hstack([tiles[0], tiles[1]])
            bottom = np.hstack([tiles[2], tiles[3]])
            core = np.vstack([top, bottom])
            if zoomed_panel is not None:
                core = cv2.resize(tiles[zoomed_panel], (tile_w * 2, tile_h * 2), interpolation=cv2.INTER_LINEAR)

            header = np.zeros((header_h, core.shape[1], 3), dtype=np.uint8)
            footer = np.zeros((footer_h, core.shape[1], 3), dtype=np.uint8)
            draw_text_with_bg(header, f"{step_name} - auto scale on frame 1 (ROI modifiable)", (14, 44), scale=1.0, color=(0, 255, 255), thickness=2)
            draw_text_with_bg(
                footer,
                f"cm between ticks={cm_per_tick:.3f} | Click active video | Double-click fullscreen/return | IJKL move | U/O width -/+ | Y/H height -/+ | R reset | +/- zoom | Enter",
                (12, 50),
                scale=0.74,
                color=(0, 255, 0),
                thickness=2,
            )
            if warning:
                draw_text_with_bg(footer, warning, (12, 22), scale=0.8, color=(0, 120, 255), thickness=2)
            grid = np.vstack([header, core, footer])
            current_grid_h, current_grid_w = grid.shape[:2]
            cv2.imshow(win, grid)

            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            if key in (13, 32):
                if all(d.cm_per_pixel is not None and d.cm_per_pixel > 0 for d in details_list):
                    final_details = details_list
                    break
                warning = "Auto KO on at least one video: adjust ROIs then revalidate."
                continue
            if key == 27:
                cv2.destroyWindow(win)
                return None
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                active = int(chr(key)) - 1
                warning = ""
                continue
            if key in (ord("r"), ord("R")):
                rois[active] = self._default_scale_roi(frames[active])
                warning = ""
                continue
            if key in (ord("+"), ord("=")):
                tile_w = min(1200, int(tile_w * 1.08))
                tile_h = min(700, int(tile_h * 1.08))
                warning = ""
                continue
            if key in (ord("-"), ord("_")):
                tile_w = max(540, int(tile_w * 0.92))
                tile_h = max(320, int(tile_h * 0.92))
                warning = ""
                continue
            if key in (ord("i"), ord("j"), ord("k"), ord("l"), ord("u"), ord("o"), ord("y"), ord("h"), ord("I"), ord("J"), ord("K"), ord("L"), ord("U"), ord("O"), ord("Y"), ord("H")):
                rois[active] = self._update_roi(rois[active], key, frames[active])
                warning = ""
                continue

        cv2.destroyWindow(win)
        out_scales: list[float] = []
        for d in final_details:
            if d is None:
                continue
            cm_val = getattr(d, "cm_per_pixel", None)
            if cm_val is not None:
                out_scales.append(float(cm_val))
        return out_scales

    def _measure_simpson_step(self, step_name: str, video_paths: list[str], titles: list[str], labels: list[str], scales: list[float]) -> dict[str, float] | None:
        selected = select_frames_grid(video_paths, titles, step_name=f"{step_name} - frame selection")
        if any(frame is None for frame in selected.frames):
            return None

        frames = [frame for frame in selected.frames if frame is not None]
        distances_px = measure_lengths_grid(frames, titles, labels, step_name=f"{step_name} - measures", scales_cm_per_px=scales)
        if distances_px is None:
            return None

        values: dict[str, float] = {}
        for i, dist_px in enumerate(distances_px):
            values[labels[i]] = dist_px * scales[i]

        self._log(
            f"{step_name} frames: "
            + " | ".join([f"{titles[i]}={selected.indices[i] + 1}" for i in range(4)])
        )
        return values

    def run_simpson(self) -> None:
        self._log("--- Modified Simpson (2x2 grid EDV/ESV) ---")
        try:
            apical = self.simpson_apical_var.get().strip()
            mv = self.simpson_mv_var.get().strip()
            pm = self.simpson_pm_var.get().strip()
            apex = self.simpson_apex_var.get().strip()
            if not all([apical, mv, pm, apex]):
                raise ValueError("Select the 4 videos: Apical, MV, PM, Apex")

            video_paths = [apical, mv, pm, apex]
            titles = ["Apical", "MV", "PM", "Apex"]
            for path in video_paths:
                if not os.path.isfile(path):
                    raise ValueError(f"Video not found: {path}")

            tick_cm = self._require_float(self.simpson_tick_cm_var.get(), "Scale tick dist (cm)")
            if tick_cm <= 0:
                raise ValueError("Scale tick distance must be > 0")

            scales_edv = self._preview_scales_grid(video_paths, titles, step_name="EDV", cm_per_tick=tick_cm)
            if scales_edv is None:
                return
            labels_edv = ["L_ED", "D_MV_ED", "D_PM_ED", "D_APEX_ED"]
            values_edv = self._measure_simpson_step("EDV", video_paths, titles, labels_edv, scales_edv)
            if values_edv is None:
                return

            scales_esv = self._preview_scales_grid(video_paths, titles, step_name="ESV", cm_per_tick=tick_cm)
            if scales_esv is None:
                return
            labels_esv = ["L_ES", "D_MV_ES", "D_PM_ES", "D_APEX_ES"]
            values_esv = self._measure_simpson_step("ESV", video_paths, titles, labels_esv, scales_esv)
            if values_esv is None:
                return

            edv = calculate_volume_simpson(
                values_edv["L_ED"], values_edv["D_APEX_ED"], values_edv["D_PM_ED"], values_edv["D_MV_ED"]
            )
            esv = calculate_volume_simpson(
                values_esv["L_ES"], values_esv["D_APEX_ES"], values_esv["D_PM_ES"], values_esv["D_MV_ES"]
            )
            hr = self._optional_float(self.simpson_hr_var.get())
            result = build_volume_result(edv, esv, hr)

            self._log(
                "EDV measures (cm): "
                + f"L={values_edv['L_ED']:.2f} | MV={values_edv['D_MV_ED']:.2f} | PM={values_edv['D_PM_ED']:.2f} | Apex={values_edv['D_APEX_ED']:.2f}"
            )
            self._log(
                "ESV measures (cm): "
                + f"L={values_esv['L_ES']:.2f} | MV={values_esv['D_MV_ES']:.2f} | PM={values_esv['D_PM_ES']:.2f} | Apex={values_esv['D_APEX_ES']:.2f}"
            )
            self._log(f"EDV={result.edv:.2f} ml | ESV={result.esv:.2f} ml | SV={result.sv:.2f} ml | EF={result.ef:.1f}%")
            if result.co is not None:
                self._log(f"CO={result.co:.2f} L/min")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def run_doppler(self) -> None:
        self._log("--- Doppler ---")
        try:
            frame = None
            img_path = self.doppler_image_var.get().strip()
            video_path = self.doppler_video_var.get().strip()

            if img_path and os.path.isfile(img_path):
                frame = cv2.imread(img_path)
                if frame is None:
                    raise ValueError("Unable to read the D_AO image")
            elif video_path and os.path.isfile(video_path):
                selected = select_frame(video_path, window_name="Frame Selection - D_AO")
                if selected.frame is None:
                    return
                frame = selected.frame
            else:
                raise ValueError("Select a D_AO image or a D_AO video")

            scale = self._calibrate_scale(frame)
            if scale is None:
                return

            m = measure_distance(frame, "D_AO", window_name="Measure - D_AO")
            if m is None or not m.is_complete():
                return
            d_ao = m.distance_pixels() * scale

            vti = self._require_float(self.doppler_vti_var.get(), "VTI")
            hr = self._require_float(self.doppler_hr_var.get(), "HR")

            result = build_doppler_result(d_ao, vti, hr)
            self._log(f"D_AO={result.d_ao:.2f} cm | VTI={result.vti:.2f} cm | HR={result.hr:.0f} bpm")
            self._log(f"Aortic area={result.area_ao:.2f} cm^2 | SV={result.sv:.2f} ml | CO={result.co:.2f} L/min")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


def run_gui() -> None:
    root = tk.Tk()
    app = HeartVolumeApp(root)
    root.mainloop()
