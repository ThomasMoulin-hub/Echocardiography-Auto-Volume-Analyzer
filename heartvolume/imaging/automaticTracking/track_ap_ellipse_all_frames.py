from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from heartvolume.imaging.automaticTracking.fit_ap_ellipse_manual import (
    FreehandEllipseTool,
    Ellipse,
    ellipse_to_dict,
    fit_ellipse_from_points,
    refine_ellipse_with_image,
)

# Path resolution to be independent of current working directory or script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_VIDEO = str(PROJECT_ROOT / "data" / "2nd Session" / "Dendo_video_AP.mp4")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "automatic_ellipse_tracking"


@dataclass
class TrackPoint:
    frame_index: int
    ellipse: Ellipse
    source: str


def read_video_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)

    cap.release()
    if not frames:
        raise ValueError("Aucune frame lisible dans la video")
    return frames, fps


def collect_initial_freehand_points(frame0: np.ndarray) -> list[tuple[int, int]]:
    tool = FreehandEllipseTool(frame0)
    win = "AP Tracking - Trace initiale (Frame 1)"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1400, 900)
    cv2.setMouseCallback(win, tool.on_mouse)

    while True:
        canvas = tool.display.copy()
        cv2.putText(canvas, "Trace initial de la zone a suivre", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Entree: valider | R: reset | ESC: quitter", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        key = cv2.waitKeyEx(20)
        if key == 27:
            cv2.destroyWindow(win)
            return []
        if key in (ord("r"), ord("R")):
            tool.reset()
            continue
        if key in (13, 10):
            if len(tool.points) < 5:
                print("Trace trop court, continue le contour.")
                continue
            break

    cv2.destroyWindow(win)
    return tool.points


def circle_seed_points_from_ellipse(ellipse: Ellipse, n_pts: int = 64) -> list[tuple[int, int]]:
    (cx, cy), (major, minor), _ = ellipse
    radius = max(4.0, min(float(major), float(minor)) / 2.0)
    pts: list[tuple[int, int]] = []
    for a in np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False):
        x = int(round(cx + radius * float(np.cos(a))))
        y = int(round(cy + radius * float(np.sin(a))))
        pts.append((x, y))
    return pts


def _angle_lerp_deg(a_prev: float, a_new: float, alpha: float) -> float:
    delta = ((a_new - a_prev + 90.0) % 180.0) - 90.0
    return float((a_prev + alpha * delta) % 180.0)


def smooth_ellipse(prev: Ellipse, current: Ellipse, frame_shape: tuple[int, int, int], alpha: float = 0.52) -> Ellipse:
    h, w = frame_shape[:2]
    # More freedom on center motion while keeping axis stability guardrails.
    max_center_jump = 0.26 * float(min(h, w))
    max_axis_ratio_delta = 0.55
    center_alpha = 0.72
    axis_alpha = 0.60

    (pcx, pcy), (pmaj, pmin), pang = prev
    (ccx, ccy), (cmaj, cmin), cang = current

    dx = ccx - pcx
    dy = ccy - pcy
    dist = float(np.hypot(dx, dy))
    if dist > max_center_jump and dist > 1e-6:
        scale = max_center_jump / dist
        ccx = pcx + dx * scale
        ccy = pcy + dy * scale

    maj_low = pmaj * (1.0 - max_axis_ratio_delta)
    maj_high = pmaj * (1.0 + max_axis_ratio_delta)
    min_low = pmin * (1.0 - max_axis_ratio_delta)
    min_high = pmin * (1.0 + max_axis_ratio_delta)
    cmaj = float(np.clip(cmaj, maj_low, maj_high))
    cmin = float(np.clip(cmin, min_low, min_high))

    scx = (1.0 - center_alpha) * pcx + center_alpha * ccx
    scy = (1.0 - center_alpha) * pcy + center_alpha * ccy
    smaj = (1.0 - axis_alpha) * pmaj + axis_alpha * cmaj
    smin = (1.0 - axis_alpha) * pmin + axis_alpha * cmin
    sang = _angle_lerp_deg(float(pang), float(cang), axis_alpha)

    return ((float(scx), float(scy)), (float(smaj), float(smin)), float(sang))


def draw_track_overlay(frame: np.ndarray, ellipse: Ellipse, frame_idx: int, total: int, source: str, pct: int) -> np.ndarray:
    out = frame.copy()
    cv2.ellipse(out, ellipse, (255, 120, 0), 2, cv2.LINE_AA)
    cv2.circle(out, (int(round(ellipse[0][0])), int(round(ellipse[0][1]))), 3, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(out, f"Frame {frame_idx + 1}/{total}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Source: {source}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(out, f"Seuil transition: +{pct}%", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    return out


def run_tracking(video_path: str, output_dir: Path, increase_pct: int, show_preview: bool) -> None:
    frames, fps = read_video_frames(video_path)
    initial_points = collect_initial_freehand_points(frames[0])
    if len(initial_points) < 5:
        print("Tracking annule: aucun trace valide.")
        cv2.destroyAllWindows()
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking.csv"
    json_path = output_dir / "tracking.json"
    video_path_out = output_dir / "tracking_overlay.mp4"

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path_out),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Impossible de creer la video de sortie: {video_path_out}")

    track: list[TrackPoint] = []

    first_detected = refine_ellipse_with_image(frames[0], initial_points, transition_increase_pct=increase_pct)
    if first_detected is None:
        first_detected = fit_ellipse_from_points(initial_points)
    if first_detected is None:
        writer.release()
        raise RuntimeError("Impossible d'initialiser l'ellipse sur la frame 1")

    current = first_detected
    track.append(TrackPoint(frame_index=0, ellipse=current, source="manual+refine"))

    overlay0 = draw_track_overlay(frames[0], current, 0, len(frames), "manual+refine", increase_pct)
    writer.write(overlay0)

    if show_preview:
        cv2.namedWindow("AP Tracking Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AP Tracking Preview", 1200, 800)
        cv2.imshow("AP Tracking Preview", overlay0)
        cv2.waitKey(1)

    for idx in range(1, len(frames)):
        seed_points = circle_seed_points_from_ellipse(current)
        detected = refine_ellipse_with_image(frames[idx], seed_points, transition_increase_pct=increase_pct)

        if detected is None:
            source = "fallback_previous"
            candidate = current
        else:
            source = "refined"
            candidate = smooth_ellipse(current, detected, frames[idx].shape)

        current = candidate
        track.append(TrackPoint(frame_index=idx, ellipse=current, source=source))

        overlay = draw_track_overlay(frames[idx], current, idx, len(frames), source, increase_pct)
        writer.write(overlay)

        if show_preview:
            cv2.imshow("AP Tracking Preview", overlay)
            key = cv2.waitKey(1)
            if key == 27:
                break

    writer.release()
    cv2.destroyAllWindows()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            "frame_index",
            "center_x",
            "center_y",
            "axis_major_px",
            "axis_minor_px",
            "angle_deg",
            "source",
        ])
        for tp in track:
            e = tp.ellipse
            writer_csv.writerow([
                tp.frame_index,
                f"{e[0][0]:.3f}",
                f"{e[0][1]:.3f}",
                f"{e[1][0]:.3f}",
                f"{e[1][1]:.3f}",
                f"{e[2]:.3f}",
                tp.source,
            ])

    payload = {
        "video": video_path,
        "fps": fps,
        "transition_increase_pct": int(increase_pct),
        "n_frames_processed": len(track),
        "tracks": [
            {
                "frame_index": tp.frame_index,
                "ellipse": ellipse_to_dict(tp.ellipse),
                "source": tp.source,
            }
            for tp in track
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Tracking termine. Sorties: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracking d'ellipse AP sur toutes les frames d'une video.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Chemin de la video AP")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Dossier de sortie")
    parser.add_argument("--increase-pct", type=int, default=15, help="Seuil de transition relatif en pourcentage")
    parser.add_argument("--no-preview", action="store_true", help="Desactive la fenetre de preview pendant le tracking")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pct = max(1, min(1000, int(args.increase_pct)))
    run_tracking(args.video, Path(args.output), pct, show_preview=not args.no_preview)

