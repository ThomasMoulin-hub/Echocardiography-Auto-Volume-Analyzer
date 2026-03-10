from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from heartvolume.data.discovery import auto_detect_videos
from heartvolume.imaging.scale_detection import detect_scale_with_details, draw_scale_detection


SENSITIVITY_ORDER = ["low", "medium", "high", "ultra", "ultra_plus", "ultra_max"]


def _default_lendo_path() -> str | None:
    videos = auto_detect_videos()
    return videos.get("apical")


def _score_detection(details) -> int:
    score = 0
    if details.cm_per_pixel is not None and details.cm_per_pixel > 0:
        score += 10000
    majors = details.major_tick_positions_y or []
    ticks = details.tick_positions_y or []
    score += 110 * len(majors)
    score += len(ticks)

    # Reward regular major spacing to favor stable ruler interpretations.
    if len(majors) >= 3:
        diffs = [majors[i] - majors[i - 1] for i in range(1, len(majors))]
        positive = [d for d in diffs if d > 0]
        if positive:
            mean = sum(positive) / float(len(positive))
            spread = max(positive) - min(positive)
            if mean > 0:
                rel_spread = spread / mean
                if rel_spread <= 0.30:
                    score += 350
                elif rel_spread <= 0.45:
                    score += 180

    if details.visible_cm is not None:
        score += int(min(500.0, max(0.0, float(details.visible_cm) * 10.0)))
    if details.ocr_cm_per_pixel is not None:
        score += 300
    return score


def _build_sensitive_variants(frame, sensitivity: str):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    clahe_soft = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(gray)

    sharp_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sharp = cv2.addWeighted(gray, 1.35, cv2.GaussianBlur(gray, (0, 0), 1.0), -0.35, 0)
    sharp = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, sharp_kernel)

    variants = [
        frame,
        cv2.convertScaleAbs(frame, alpha=1.12, beta=6),
        cv2.convertScaleAbs(frame, alpha=1.18, beta=12),
        cv2.convertScaleAbs(frame, alpha=1.3, beta=0),
        cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        cv2.convertScaleAbs(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), alpha=1.1, beta=0),
        cv2.cvtColor(clahe_soft, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),
    ]

    if sensitivity in {"ultra", "ultra_plus", "ultra_max"}:
        gamma = np.clip(np.power(gray.astype("float32") / 255.0, 0.78) * 255.0, 0, 255).astype("uint8")
        denoise = cv2.bilateralFilter(gray, 7, 40, 40)
        variants.extend(
            [
                cv2.cvtColor(gamma, cv2.COLOR_GRAY2BGR),
                cv2.convertScaleAbs(cv2.cvtColor(gamma, cv2.COLOR_GRAY2BGR), alpha=1.1, beta=8),
                cv2.cvtColor(denoise, cv2.COLOR_GRAY2BGR),
                cv2.convertScaleAbs(cv2.cvtColor(denoise, cv2.COLOR_GRAY2BGR), alpha=1.25, beta=4),
            ]
        )

    if sensitivity in {"ultra_plus", "ultra_max"}:
        eq = cv2.equalizeHist(gray)
        variants.extend(
            [
                cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR),
                cv2.convertScaleAbs(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR), alpha=1.2, beta=10),
            ]
        )

    if sensitivity == "ultra_max":
        gamma_strong = np.clip(np.power(gray.astype("float32") / 255.0, 0.68) * 255.0, 0, 255).astype("uint8")
        variants.append(cv2.cvtColor(gamma_strong, cv2.COLOR_GRAY2BGR))

    return variants


def _candidate_rois(frame, sensitivity: str):
    h, w = frame.shape[:2]
    rois = [
        None,
        (int(w * 0.84), int(h * 0.01), w, int(h * 0.76)),
        (int(w * 0.82), int(h * 0.02), w, int(h * 0.76)),
        (int(w * 0.80), int(h * 0.02), w, int(h * 0.78)),
        (int(w * 0.76), int(h * 0.02), w, int(h * 0.80)),
        (int(w * 0.72), int(h * 0.02), w, int(h * 0.82)),
        (int(w * 0.68), int(h * 0.01), w, int(h * 0.84)),
    ]
    if sensitivity in {"ultra", "ultra_plus", "ultra_max"}:
        rois.extend(
            [
                (int(w * 0.64), int(h * 0.01), w, int(h * 0.86)),
                (int(w * 0.60), int(h * 0.00), w, int(h * 0.88)),
            ]
        )
    if sensitivity in {"ultra_plus", "ultra_max"}:
        rois.extend(
            [
                (int(w * 0.56), int(h * 0.00), w, int(h * 0.90)),
                (int(w * 0.52), int(h * 0.00), w, int(h * 0.92)),
            ]
        )
    if sensitivity == "ultra_max":
        rois.append((int(w * 0.48), int(h * 0.00), w, int(h * 0.95)))
    return rois


def _resolve_sensitivity_for_frame(
    base_sensitivity: str,
    progressive: bool,
    processed_idx: int,
    ramp_frames: int,
) -> str:
    if not progressive:
        return base_sensitivity

    if ramp_frames <= 1:
        return SENSITIVITY_ORDER[-1]

    ratio = min(1.0, max(0.0, processed_idx / float(ramp_frames - 1)))
    level = int(round(ratio * (len(SENSITIVITY_ORDER) - 1)))
    return SENSITIVITY_ORDER[level]


def _detect_scale_sensitivity(
    frame,
    cm_per_tick: float,
    sensitivity: str,
    roi_override: tuple[int, int, int, int] | None = None,
):
    if sensitivity == "low":
        return detect_scale_with_details(frame, cm_per_tick=cm_per_tick, roi_override=roi_override)

    best = detect_scale_with_details(frame, cm_per_tick=cm_per_tick, roi_override=roi_override)
    best_score = _score_detection(best)

    rois = [roi_override] if roi_override is not None else []
    rois.extend(_candidate_rois(frame, sensitivity))
    variants = [frame] if sensitivity == "medium" else _build_sensitive_variants(frame, sensitivity)

    for variant in variants:
        for roi in rois:
            details = detect_scale_with_details(variant, cm_per_tick=cm_per_tick, roi_override=roi)
            score = _score_detection(details)
            if score > best_score:
                best = details
                best_score = score

    return best


def detect_scale_with_sensitivity(
    frame,
    cm_per_tick: float = 1.0,
    sensitivity: str = "high",
    roi_override: tuple[int, int, int, int] | None = None,
):
    """Public helper reused by GUI to match export detection settings."""
    if sensitivity not in set(SENSITIVITY_ORDER):
        sensitivity = "high"
    return _detect_scale_sensitivity(frame, cm_per_tick=cm_per_tick, sensitivity=sensitivity, roi_override=roi_override)


def export_frames_with_scale_overlay(
    video_path: str,
    output_dir: str,
    cm_per_tick: float = 1.0,
    max_frames: int | None = None,
    every_n: int = 1,
    sensitivity: str = "high",
    progressive_sensitivity: bool = False,
    ramp_frames: int = 300,
) -> tuple[int, int]:
    if not os.path.isfile(video_path):
        raise ValueError(f"Video introuvable: {video_path}")
    if cm_per_tick <= 0:
        raise ValueError("cm_per_tick doit etre > 0")
    if every_n <= 0:
        raise ValueError("every_n doit etre > 0")
    if sensitivity not in set(SENSITIVITY_ORDER):
        raise ValueError(f"sensitivity doit etre dans {SENSITIVITY_ORDER}")
    if ramp_frames <= 0:
        raise ValueError("ramp_frames doit etre > 0")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "scale_detection_summary.csv"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la video: {video_path}")

    written = 0
    processed = 0
    frame_idx = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "sensitivity_used",
            "score",
            "detected",
            "cm_per_pixel",
            "visible_cm",
            "major_ticks",
            "total_ticks",
        ])

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % every_n != 0:
                frame_idx += 1
                continue

            sensitivity_used = _resolve_sensitivity_for_frame(
                base_sensitivity=sensitivity,
                progressive=progressive_sensitivity,
                processed_idx=processed,
                ramp_frames=ramp_frames,
            )
            details = detect_scale_with_sensitivity(frame, cm_per_tick=cm_per_tick, sensitivity=sensitivity_used)
            score = _score_detection(details)
            overlay = draw_scale_detection(frame, details, title=f"frame {frame_idx + 1} | {sensitivity_used}")

            output_name = out / f"frame_{frame_idx + 1:06d}.png"
            cv2.imwrite(str(output_name), overlay)

            writer.writerow(
                [
                    frame_idx + 1,
                    sensitivity_used,
                    score,
                    int(details.cm_per_pixel is not None),
                    "" if details.cm_per_pixel is None else f"{details.cm_per_pixel:.10f}",
                    "" if details.visible_cm is None else f"{details.visible_cm:.4f}",
                    len(details.major_tick_positions_y or []),
                    len(details.tick_positions_y),
                ]
            )

            written += 1
            processed += 1
            frame_idx += 1

            if max_frames is not None and processed >= max_frames:
                break

            if processed % 25 == 0:
                print(f"{processed} frame(s) exportees... (sensibilite actuelle: {sensitivity_used})")

    cap.release()
    return processed, written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporte les frames de la video Lendo avec overlay auto-detection de l'echelle."
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Chemin de la video. Par defaut: auto-detection de la video apicale (Lendo).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/lendo_scale_overlay",
        help="Dossier de sortie des images et du CSV.",
    )
    parser.add_argument(
        "--cm-per-tick",
        type=float,
        default=1.0,
        help="Distance reelle entre deux gros tirets (en cm).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Nombre max de frames a traiter (optionnel).",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Traite 1 frame sur N (1 = toutes les frames).",
    )
    parser.add_argument(
        "--sensitivity",
        choices=SENSITIVITY_ORDER,
        default="high",
        help="Sensibilite fixe: low, medium, high, ultra, ultra_plus, ultra_max.",
    )
    parser.add_argument(
        "--progressive-sensitivity",
        action="store_true",
        help="Fait monter automatiquement la sensibilite de low vers ultra_max au fil des frames.",
    )
    parser.add_argument(
        "--ramp-frames",
        type=int,
        default=300,
        help="Nombre de frames sur lesquelles la sensibilite monte progressivement.",
    )
    args = parser.parse_args()

    video_path = args.video or _default_lendo_path()
    if not video_path:
        raise ValueError("Aucune video Lendo auto-detectee. Passe --video <chemin>.")

    processed, written = export_frames_with_scale_overlay(
        video_path=video_path,
        output_dir=args.output_dir,
        cm_per_tick=args.cm_per_tick,
        max_frames=args.max_frames,
        every_n=args.every_n,
        sensitivity=args.sensitivity,
        progressive_sensitivity=args.progressive_sensitivity,
        ramp_frames=args.ramp_frames,
    )
    print(f"Termine. Frames traitees={processed}, images ecrites={written}")
    print(f"Sortie: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

