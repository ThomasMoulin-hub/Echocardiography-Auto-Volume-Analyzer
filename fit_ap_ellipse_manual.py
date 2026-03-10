from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

DEFAULT_VIDEO = "data/Dendo_video_AP.mp4"
OUTPUT_DIR = Path("output/ap_ellipse_fit")
Ellipse = tuple[tuple[float, float], tuple[float, float], float]
DEFAULT_TRANSITION_INCREASE_PCT = 20
MAX_TRANSITION_INCREASE_PCT = 1000
MIN_BASE_FOR_PERCENT = 6.0


class FreehandEllipseTool:
    def __init__(self, frame: np.ndarray) -> None:
        self.base_frame = frame
        self.display = frame.copy()
        self.points: list[tuple[int, int]] = []
        self.is_drawing = False

    def reset(self) -> None:
        self.display = self.base_frame.copy()
        self.points = []
        self.is_drawing = False

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.points = [(x, y)]
            self.display = self.base_frame.copy()
            cv2.circle(self.display, (x, y), 2, (0, 255, 255), -1)
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            last_x, last_y = self.points[-1]
            self.points.append((x, y))
            cv2.line(self.display, (last_x, last_y), (x, y), (0, 255, 255), 2, cv2.LINE_AA)
        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.is_drawing = False
            self.points.append((x, y))


def read_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError("Impossible de lire la premiere frame")
    return frame


def _ellipse_from_raw(raw: tuple[Sequence[float], Sequence[float], float]) -> Ellipse:
    return (
        (float(raw[0][0]), float(raw[0][1])),
        (float(raw[1][0]), float(raw[1][1])),
        float(raw[2]),
    )


def fit_ellipse_from_points(points: list[tuple[int, int]]) -> Optional[Ellipse]:
    if len(points) < 5:
        return None
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    return _ellipse_from_raw(cv2.fitEllipse(contour))


def _ellipse_mask(shape: tuple[int, int], ellipse: Ellipse, scale: float) -> np.ndarray:
    h, w = shape
    (cx, cy), (ax, by), angle = ellipse
    ax2 = max(4, int((ax * scale) / 2.0))
    by2 = max(4, int((by * scale) / 2.0))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (ax2, by2), angle, 0, 360, 255, -1)
    return mask


def _polygon_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) < 3:
        return mask
    poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 255)
    return mask


def refine_ellipse_with_image(
    frame: np.ndarray,
    freehand_points: list[tuple[int, int]],
    transition_increase_pct: int = DEFAULT_TRANSITION_INCREASE_PCT,
) -> Optional[Ellipse]:
    if len(freehand_points) < 5:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    seed_mask = _polygon_mask(gray.shape, freehand_points)
    if int(np.count_nonzero(seed_mask)) < 50:
        return None

    pct = float(max(1, min(MAX_TRANSITION_INCREASE_PCT, int(transition_increase_pct))))

    # Start from the full traced area, then grow only through near-dark neighbors.
    region = seed_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    last_valid: Optional[Ellipse] = None

    for _ in range(220):
        ys, xs = np.where(region > 0)
        if len(xs) < 25:
            return None

        pts = np.column_stack((xs, ys)).astype(np.int32).reshape((-1, 1, 2))
        ellipse = _ellipse_from_raw(cv2.fitEllipse(pts))
        last_valid = ellipse

        inner = _ellipse_mask(gray.shape, ellipse, 1.00)
        outer = _ellipse_mask(gray.shape, ellipse, 1.12)
        ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
        ring_vals = gray[ring > 0]
        region_vals = gray[region > 0]

        if ring_vals.size >= 120 and region_vals.size >= 120:
            inner_ref = float(np.percentile(region_vals, 60))
            outer_ref = float(np.percentile(ring_vals, 40))
            base = max(inner_ref, MIN_BASE_FOR_PERCENT)
            change_pct = ((outer_ref - inner_ref) / base) * 100.0

            # Stop when the detected global transition exceeds the chosen threshold.
            if change_pct >= pct and outer_ref > inner_ref:
                return ellipse

            frontier_limit = int(max(4.0, min(40.0, inner_ref + 3.0)))
        else:
            frontier_limit = 8

        dilated = cv2.dilate(region, kernel, iterations=1)
        growth = np.zeros_like(region)
        growth[(dilated > 0) & (region == 0) & (gray <= frontier_limit)] = 255
        if int(np.count_nonzero(growth)) == 0:
            break

        region = cv2.bitwise_or(region, growth)

    return last_valid


def draw_overlay(
    frame: np.ndarray,
    freehand_points: list[tuple[int, int]],
    freehand_ellipse: Ellipse,
    refined_ellipse: Optional[Ellipse],
    transition_increase_pct: int,
) -> np.ndarray:
    out = frame.copy()
    if len(freehand_points) >= 2:
        poly = np.array(freehand_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [poly], False, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.ellipse(out, freehand_ellipse, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "Ellipse ajustee sur trace", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    if refined_ellipse is not None:
        cv2.ellipse(out, refined_ellipse, (255, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(out, "Ellipse recherchee", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 120, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "Ellipse recherchee non trouvee", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2, cv2.LINE_AA)

    cv2.putText(
        out,
        f"Seuil transition: +{transition_increase_pct}% (arret quand variation relative interieur->exterieur depasse ce seuil)",
        (20, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(out, "Jaune: trace libre", (20, out.shape[0] - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "R: relancer recherche | +/- ou T/G: changer seuil", (20, out.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180, 255, 180), 2, cv2.LINE_AA)
    return out


def ellipse_to_dict(ellipse: Ellipse) -> dict[str, float]:
    (cx, cy), (major, minor), angle = ellipse
    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "axis_major_px": float(major),
        "axis_minor_px": float(minor),
        "angle_deg": float(angle),
    }


def save_results(
    frame: np.ndarray,
    overlay: np.ndarray,
    freehand_ellipse: Ellipse,
    refined_ellipse: Optional[Ellipse],
    output_dir: Path,
    transition_increase_pct: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "frame1.png"), frame)
    cv2.imwrite(str(output_dir / "frame1_overlay.png"), overlay)

    payload = {
        "manual_fit": ellipse_to_dict(freehand_ellipse),
        "image_refined_fit": ellipse_to_dict(refined_ellipse) if refined_ellipse is not None else None,
        "refined_found": refined_ellipse is not None,
        "transition_increase_pct": int(transition_increase_pct),
    }
    (output_dir / "ellipse_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(video_path: str, output_dir: Path) -> None:
    frame = read_first_frame(video_path)
    tool = FreehandEllipseTool(frame)

    win = "AP - Trace libre ellipse (Frame 1)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1400, 900)
    cv2.setMouseCallback(win, tool.on_mouse)

    while True:
        canvas = tool.display.copy()
        cv2.putText(canvas, "Trace une ellipse a main levee (souris)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Entree: valider | R: reset | ESC: quitter", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        key = cv2.waitKeyEx(20)
        if key == 27:
            cv2.destroyAllWindows()
            return
        if key in (ord("r"), ord("R")):
            tool.reset()
            continue
        if key in (13, 10):
            if len(tool.points) < 5:
                print("Pas assez de points: trace plus longuement le contour.")
                continue
            break

    cv2.destroyWindow(win)

    freehand_ellipse = fit_ellipse_from_points(tool.points)
    if freehand_ellipse is None:
        raise RuntimeError("Impossible d'ajuster une ellipse depuis la trace libre")

    threshold = DEFAULT_TRANSITION_INCREASE_PCT
    refined = refine_ellipse_with_image(frame, tool.points, transition_increase_pct=threshold)
    overlay = draw_overlay(frame, tool.points, freehand_ellipse, refined, threshold)

    preview = "AP - Resultat ellipse"
    cv2.namedWindow(preview, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview, 1400, 900)

    while True:
        canvas = overlay.copy()
        cv2.putText(canvas, "S: sauvegarder | R: relancer | +/- ou T/G: seuil | ESC: fermer", (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.imshow(preview, canvas)
        key = cv2.waitKeyEx(20)
        if key == 27:
            cv2.destroyAllWindows()
            return
        if key in (ord("r"), ord("R")):
            refined = refine_ellipse_with_image(frame, tool.points, transition_increase_pct=threshold)
            overlay = draw_overlay(frame, tool.points, freehand_ellipse, refined, threshold)
            continue
        if key in (ord("+"), ord("="), ord("t"), ord("T")):
            threshold = min(MAX_TRANSITION_INCREASE_PCT, threshold + 1)
            refined = refine_ellipse_with_image(frame, tool.points, transition_increase_pct=threshold)
            overlay = draw_overlay(frame, tool.points, freehand_ellipse, refined, threshold)
            continue
        if key in (ord("-"), ord("_"), ord("g"), ord("G")):
            threshold = max(1, threshold - 1)
            refined = refine_ellipse_with_image(frame, tool.points, transition_increase_pct=threshold)
            overlay = draw_overlay(frame, tool.points, freehand_ellipse, refined, threshold)
            continue
        if key in (ord("s"), ord("S"), 13, 10):
            save_results(frame, overlay, freehand_ellipse, refined, output_dir, threshold)
            cv2.destroyAllWindows()
            print(f"Resultats sauvegardes dans: {output_dir}")
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace libre puis fitting d'ellipse sur la video AP (frame 1).")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Chemin de la video AP")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Dossier de sortie")
    parser.add_argument("--increase-pct", type=int, default=15, help=f"Pourcentage d'augmentation interieur->exterieur initial (1-{MAX_TRANSITION_INCREASE_PCT})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DEFAULT_TRANSITION_INCREASE_PCT = max(1, min(MAX_TRANSITION_INCREASE_PCT, int(args.increase_pct)))
    run(args.video, Path(args.output))
