from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import cv2
import numpy as np


@dataclass
class ScaleDetectionResult:
    cm_per_pixel: Optional[float]
    tick_positions_y: list[int]
    roi: tuple[int, int, int, int]
    spacing_px: Optional[float] = None
    visible_cm: Optional[float] = None
    major_tick_positions_y: Optional[list[int]] = None
    ocr_cm_per_pixel: Optional[float] = None


def _filter_close_positions(positions: list[int], min_gap: int = 8) -> list[int]:
    if not positions:
        return []
    ordered = sorted(positions)
    filtered = [ordered[0]]
    for value in ordered[1:]:
        if value - filtered[-1] >= min_gap:
            filtered.append(value)
    return filtered


def _merge_tick_candidates(candidates: list[tuple[int, int]], min_gap: int = 6) -> list[tuple[int, int]]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda v: v[0])
    merged: list[list[tuple[int, int]]] = [[ordered[0]]]
    for y, length in ordered[1:]:
        if abs(y - merged[-1][-1][0]) <= min_gap:
            merged[-1].append((y, length))
        else:
            merged.append([(y, length)])

    out: list[tuple[int, int]] = []
    for group in merged:
        y_mean = int(round(sum(v[0] for v in group) / len(group)))
        length_max = max(v[1] for v in group)
        out.append((y_mean, length_max))
    return out


def _estimate_cm_per_px_from_ticks(ticks: list[tuple[int, int]], cm_per_major_tick: float) -> tuple[Optional[float], Optional[float]]:
    """Return (cm_per_px, spacing_px_for_1cm)."""
    if len(ticks) < 2:
        return None, None

    ys = [t[0] for t in ticks]
    lengths = np.array([t[1] for t in ticks], dtype=np.float32)
    diffs = [ys[i] - ys[i - 1] for i in range(1, len(ys)) if 4 <= (ys[i] - ys[i - 1]) <= 180]

    spacing_1cm_px: Optional[float] = None

    # Detect major ticks (longer strokes) to directly estimate 1 cm spacing.
    q75 = float(np.percentile(lengths, 75)) if len(lengths) > 0 else 0.0
    major_idx = [i for i, l in enumerate(lengths) if l >= max(10.0, q75)]
    major_ys = [ys[i] for i in major_idx]
    major_diffs = [major_ys[i] - major_ys[i - 1] for i in range(1, len(major_ys)) if 12 <= (major_ys[i] - major_ys[i - 1]) <= 220]

    if major_diffs:
        spacing_1cm_px = float(np.median(major_diffs))

    # Fallback: infer the minor 0.2 cm step then multiply by 5.
    if spacing_1cm_px is None and diffs:
        base_minor_px = float(np.percentile(np.array(diffs, dtype=np.float32), 35))
        if base_minor_px > 0:
            # If the detected base gap is already large, it is likely a major 1 cm gap.
            spacing_1cm_px = base_minor_px if base_minor_px >= 30.0 else (base_minor_px * 5.0)

    if spacing_1cm_px is None or spacing_1cm_px <= 0:
        return None, None

    # Typical ultrasound ruler spacing is in this practical range.
    if spacing_1cm_px < 20.0 or spacing_1cm_px > 220.0:
        return None, None

    cm_per_px = cm_per_major_tick / spacing_1cm_px
    return cm_per_px, spacing_1cm_px


def _infer_major_positions_from_periodicity(ticks: list[tuple[int, int]]) -> list[int]:
    if len(ticks) < 4:
        return []

    ys = sorted([t[0] for t in ticks])
    diffs = [ys[i] - ys[i - 1] for i in range(1, len(ys)) if 4 <= (ys[i] - ys[i - 1]) <= 90]
    if not diffs:
        return []

    minor_px = float(np.percentile(np.array(diffs, dtype=np.float32), 35))
    if minor_px <= 0:
        return []
    major_px = minor_px * 5.0
    if major_px < 14 or major_px > 240:
        return []

    tol = max(3.0, minor_px * 0.8)
    best: list[int] = []
    best_anchor = ys[0]

    for anchor in ys:
        aligned: list[int] = []
        for y in ys:
            rel = (y - anchor) / major_px
            if abs(rel - round(rel)) <= (tol / max(major_px, 1.0)):
                aligned.append(y)
        if len(aligned) > len(best):
            best = aligned
            best_anchor = anchor

    if len(best) < 2:
        return []

    # Densify expected major positions and snap to nearest detected tick.
    y_min, y_max = ys[0], ys[-1]
    expected: list[int] = []
    k0 = int(np.floor((y_min - best_anchor) / major_px)) - 1
    k1 = int(np.ceil((y_max - best_anchor) / major_px)) + 1
    snap_tol = max(4.0, minor_px * 1.2)

    for k in range(k0, k1 + 1):
        y_exp = best_anchor + k * major_px
        if y_exp < (y_min - major_px) or y_exp > (y_max + major_px):
            continue
        nearest = min(ys, key=lambda v: abs(v - y_exp))
        if abs(nearest - y_exp) <= snap_tol:
            expected.append(int(nearest))

    dedup = _filter_close_positions(expected, min_gap=max(6, int(minor_px * 2.0)))
    return dedup if len(dedup) >= 2 else _filter_close_positions(best, min_gap=max(6, int(minor_px * 2.0)))


def _split_major_ticks(ticks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ticks:
        return []

    lengths = np.array([t[1] for t in ticks], dtype=np.float32)
    # Less aggressive threshold than q75 to avoid missing true major ticks.
    q60 = float(np.percentile(lengths, 60)) if len(lengths) else 0.0
    threshold = max(8.0, q60)
    length_based = [(y, ln) for y, ln in ticks if ln >= threshold]

    periodic_major_ys = _infer_major_positions_from_periodicity(ticks)
    periodic_set = set(periodic_major_ys)

    # Combine both strategies to reduce false minor classification.
    combined = [(y, ln) for y, ln in ticks if (y in periodic_set or ln >= threshold)]

    if len(combined) < 2 and len(length_based) >= 2:
        combined = length_based

    if len(combined) < 2 and len(ticks) >= 2:
        ticks_sorted = sorted(ticks, key=lambda t: t[1], reverse=True)
        keep = max(2, min(8, len(ticks_sorted) // 2))
        combined = ticks_sorted[:keep]

    return sorted(combined, key=lambda t: t[0])


def _try_ocr_number(img: np.ndarray) -> Optional[float]:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return None

    if img is None or img.size == 0:
        return None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.resize(bw, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(
        bw,
        config="--psm 6 -c tessedit_char_whitelist=0123456789.,",
    )
    if not text:
        return None
    m = re.search(r"\d+(?:[\.,]\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except Exception:
        return None


def _estimate_cm_per_px_with_ocr(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    major_ticks_y: list[int],
) -> Optional[float]:
    if len(major_ticks_y) < 2:
        return None

    x0, y0, x1, y1 = roi
    values: list[tuple[int, float]] = []
    for y in major_ticks_y:
        y_top = max(0, y - 18)
        y_bot = min(frame.shape[0], y + 18)
        x_left = max(0, x0 - 140)
        x_right = max(x_left + 10, x0 - 4)
        crop = frame[y_top:y_bot, x_left:x_right]
        val = _try_ocr_number(crop)
        if val is not None:
            values.append((y, val))

    if len(values) < 2:
        return None

    best: Optional[float] = None
    best_span = -1
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            y_a, v_a = values[i]
            y_b, v_b = values[j]
            dy = abs(y_b - y_a)
            dv = abs(v_b - v_a)
            if dy < 12 or dv <= 0:
                continue
            cm_per_px = dv / dy
            if 0.002 <= cm_per_px <= 0.08 and dy > best_span:
                best_span = dy
                best = cm_per_px

    return best


def _right_white_run_lengths(binary_roi: np.ndarray) -> np.ndarray:
    """For each row, count contiguous white pixels starting from the right border."""
    h, w = binary_roi.shape[:2]
    runs = np.zeros((h,), dtype=np.float32)
    for y in range(h):
        row = binary_roi[y]
        run = 0
        for x in range(w - 1, -1, -1):
            if row[x] > 0:
                run += 1
            elif run > 0:
                break
        runs[y] = float(run)
    return runs


def _right_edge_mean_profile(binary_roi: np.ndarray, edge_width: int = 7) -> np.ndarray:
    """Mean intensity on the right-most N pixels for each row."""
    if binary_roi is None or binary_roi.size == 0:
        return np.zeros((0,), dtype=np.float32)
    w = binary_roi.shape[1]
    n = max(1, min(edge_width, w))
    return binary_roi[:, w - n:w].mean(axis=1).astype(np.float32)


def _detect_ticks_by_rising_front(gray_roi: np.ndarray, y_offset: int) -> list[tuple[int, int]]:
    """Detect tick rows using rising fronts from dark->bright in a 1D row profile."""
    if gray_roi is None or gray_roi.size == 0:
        return []

    _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # User-requested front signal: per-row mean over the 7 right-most pixels.
    edge_profile = _right_edge_mean_profile(binary, edge_width=7)
    runs = _right_white_run_lengths(binary)
    if len(edge_profile) < 6:
        return []

    if float(edge_profile.max()) > float(edge_profile.min()):
        edge_profile = (edge_profile - edge_profile.min()) / (edge_profile.max() - edge_profile.min())
    else:
        edge_profile = np.zeros_like(edge_profile)

    smooth = cv2.GaussianBlur(edge_profile.reshape(-1, 1), (1, 5), 0).reshape(-1)
    p75 = float(np.percentile(smooth, 75)) if len(smooth) else 0.0
    thr = max(0.22, min(0.70, p75 * 0.85))

    candidates: list[tuple[int, int]] = []
    y = 1
    h = len(smooth)
    while y < h - 1:
        if smooth[y - 1] < thr <= smooth[y]:
            y2 = min(h, y + 8)
            local = smooth[y:y2]
            if len(local) == 0:
                y += 1
                continue
            y_peak = y + int(np.argmax(local))
            length = int(round(max(1.0, runs[y_peak]))) if y_peak < len(runs) else 1
            candidates.append((y_offset + y_peak, length))
            y = y_peak + 2
            continue
        y += 1

    # Secondary pass for isolated peaks when a front is weak but still present.
    for i in range(2, h - 2):
        if smooth[i] >= thr and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            length = int(round(max(1.0, runs[i]))) if i < len(runs) else 1
            candidates.append((y_offset + i, length))

    merged = _merge_tick_candidates(candidates, min_gap=5)
    return _trim_leading_artifact_ticks(merged, max_drop=3)


def _trim_leading_artifact_ticks(ticks: list[tuple[int, int]], max_drop: int = 3) -> list[tuple[int, int]]:
    """Drop 2-3 stray first ticks when they are followed by an abnormally large initial gap."""
    if len(ticks) < 7:
        return ticks

    ordered = sorted(ticks, key=lambda t: t[0])
    ys = [t[0] for t in ordered]
    diffs = [ys[i] - ys[i - 1] for i in range(1, len(ys)) if ys[i] > ys[i - 1]]
    if len(diffs) < 5:
        return ordered

    ref = float(np.median(np.array(diffs[min(2, len(diffs) - 1):], dtype=np.float32)))
    if ref <= 0:
        return ordered

    # If one of the first gaps is much larger than normal spacing, drop leading stray ticks.
    for i in range(min(max_drop, len(diffs) - 1)):
        if diffs[i] >= max(16.0, ref * 2.4):
            cut = i + 1
            return ordered[cut:]

    return ordered


def _major_positions_every_five(ticks: list[tuple[int, int]]) -> list[int]:
    """First detected tick is 0.2 cm (minor), so each 5th tick is a 1 cm major tick."""
    ordered = sorted(ticks, key=lambda t: t[0])
    majors: list[int] = []
    for idx, (y, _ln) in enumerate(ordered, start=1):
        if idx % 5 == 0:
            majors.append(y)
    return _filter_close_positions(majors, min_gap=10)


def _estimate_cm_per_px_from_pattern(
    ticks: list[tuple[int, int]],
    cm_per_major_tick: float,
) -> tuple[Optional[float], Optional[float], list[int]]:
    """Estimate cm/px with deterministic major ticks (every 5th front, first tick=0.2cm)."""
    ticks = _trim_leading_artifact_ticks(ticks, max_drop=3)
    if len(ticks) < 5:
        return None, None, []

    ys = sorted([t[0] for t in ticks])
    diffs_all = [ys[i] - ys[i - 1] for i in range(1, len(ys)) if 2 <= (ys[i] - ys[i - 1]) <= 140]
    if len(diffs_all) < 4:
        return None, None, []

    # Minor-step spacing from near-neighbor gaps; robust to occasional skipped ticks.
    base_minor = float(np.percentile(np.array(diffs_all, dtype=np.float32), 30))
    if base_minor <= 0:
        return None, None, []

    major_positions = _major_positions_every_five(ticks)

    spacing_1cm_px: Optional[float] = None
    if len(major_positions) >= 2:
        major_diffs = [
            major_positions[i] - major_positions[i - 1]
            for i in range(1, len(major_positions))
            if 20 <= (major_positions[i] - major_positions[i - 1]) <= 260
        ]
        if major_diffs:
            spacing_1cm_px = float(np.median(np.array(major_diffs, dtype=np.float32)))

    if spacing_1cm_px is None:
        spacing_1cm_px = base_minor * 5.0

    if spacing_1cm_px < 20.0 or spacing_1cm_px > 260.0:
        return None, None, []

    # Coherence check: diffs should be close to integer multiples of the minor step.
    tol = max(1.5, base_minor * 0.35)
    matched = 0
    for d in diffs_all:
        n = int(round(d / base_minor))
        if 1 <= n <= 8 and abs(d - n * base_minor) <= tol:
            matched += 1
    if (matched / float(len(diffs_all))) < 0.55:
        return None, None, []

    cm_per_px = cm_per_major_tick / spacing_1cm_px
    return cm_per_px, spacing_1cm_px, major_positions


def _detect_scale_core(
    frame: np.ndarray,
    cm_per_tick: float = 1.0,
    roi_override: Optional[tuple[int, int, int, int]] = None,
) -> ScaleDetectionResult:
    if frame is None or frame.size == 0:
        return ScaleDetectionResult(cm_per_pixel=None, tick_positions_y=[], roi=(0, 0, 0, 0))

    h, w = frame.shape[:2]
    if roi_override is None:
        # New default mode: fixed ruler ROI glued to the right side with 50 px width.
        x1 = w
        x0 = max(0, w - 50)
        y0 = int(h * 0.03)
        y1 = int(h * 0.95)
    else:
        x0, y0, x1, y1 = roi_override
        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return ScaleDetectionResult(cm_per_pixel=None, tick_positions_y=[], roi=(x0, y0, x1, y1))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Primary path requested: rising-front detection on a fixed right ROI.
    merged_ticks = _detect_ticks_by_rising_front(gray, y0)

    # Fallback to legacy contour/edge method if the rising-front pass is weak.
    if len(merged_ticks) < 5:
        tick_candidates: list[tuple[int, int]] = []

        for threshold_value in (220, 205, 190, 175, 160):
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            local: list[tuple[int, int]] = []
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw >= 5 and bw > (bh * 2) and x + bw >= (gray.shape[1] - 16):
                    local.append((y + y0, bw))

            if len(local) > len(tick_candidates):
                tick_candidates = local

            if len(local) >= 6:
                break

        merged_ticks = _merge_tick_candidates(tick_candidates, min_gap=6)
        merged_ticks = _trim_leading_artifact_ticks(merged_ticks, max_drop=3)

    tick_positions = [t[0] for t in merged_ticks]

    cm_per_pixel, spacing_px, major_positions = _estimate_cm_per_px_from_pattern(merged_ticks, cm_per_major_tick=cm_per_tick)

    if cm_per_pixel is None:
        # Secondary fallback to previous estimator for robustness.
        major_ticks = _split_major_ticks(merged_ticks)
        major_positions = [t[0] for t in major_ticks]
        cm_per_pixel, spacing_px = _estimate_cm_per_px_from_ticks(merged_ticks, cm_per_major_tick=cm_per_tick)
        if len(major_positions) < 2:
            major_positions = _major_positions_every_five(merged_ticks)
        if len(major_positions) >= 2:
            major_diffs = [
                major_positions[i] - major_positions[i - 1]
                for i in range(1, len(major_positions))
                if 12 <= (major_positions[i] - major_positions[i - 1]) <= 220
            ]
            if major_diffs:
                spacing_px = float(np.median(major_diffs))
                cm_per_pixel = cm_per_tick / spacing_px

    if major_positions:
        tick_positions = sorted(set(tick_positions).union(set(major_positions)))

    ocr_cm_per_pixel = _estimate_cm_per_px_with_ocr(frame, (x0, y0, x1, y1), major_positions)
    if ocr_cm_per_pixel is not None:
        cm_per_pixel = ocr_cm_per_pixel
        spacing_px = cm_per_tick / ocr_cm_per_pixel

    visible_cm: Optional[float] = None
    anchor_positions = major_positions if len(major_positions) >= 2 else tick_positions
    if cm_per_pixel is not None and len(anchor_positions) >= 2:
        visible_cm = (anchor_positions[-1] - anchor_positions[0]) * cm_per_pixel

    return ScaleDetectionResult(
        cm_per_pixel=cm_per_pixel,
        tick_positions_y=tick_positions,
        roi=(x0, y0, x1, y1),
        spacing_px=spacing_px,
        visible_cm=visible_cm,
        major_tick_positions_y=major_positions,
        ocr_cm_per_pixel=ocr_cm_per_pixel,
    )


def detect_scale_with_details(
    frame: np.ndarray,
    cm_per_tick: float = 1.0,
    roi_override: Optional[tuple[int, int, int, int]] = None,
) -> ScaleDetectionResult:
    return _detect_scale_core(frame, cm_per_tick=cm_per_tick, roi_override=roi_override)


def draw_scale_detection(frame: np.ndarray, result: ScaleDetectionResult, title: str = "") -> np.ndarray:
    canvas = frame.copy()
    x0, y0, x1, y1 = result.roi
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (255, 200, 0), 1)

    majors = set(result.major_tick_positions_y or [])
    for y in result.tick_positions_y:
        color = (0, 220, 255)
        thickness = 2
        if y in majors:
            color = (0, 255, 0)
            thickness = 3
        cv2.line(canvas, (x1 - 42, y), (x1 - 2, y), color, thickness)

    status = "scale: not detected"
    if result.cm_per_pixel is not None and result.cm_per_pixel > 0:
        px_per_cm = 1.0 / result.cm_per_pixel
        status = f"scale: {result.cm_per_pixel:.6f} cm/px ({px_per_cm:.1f} px/cm)"

    detail = ""
    if result.visible_cm is not None:
        detail = f" | regle visible (gros tirets) ~{result.visible_cm:.2f} cm"

    if result.ocr_cm_per_pixel is not None:
        detail += " | OCR labels utilise"

    label = f"{title} | {status}{detail}" if title else f"{status}{detail}"
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


def detect_scale_on_frame(
    frame: np.ndarray,
    cm_per_tick: float = 1.0,
    roi_override: Optional[tuple[int, int, int, int]] = None,
) -> Optional[float]:
    result = _detect_scale_core(frame, cm_per_tick=cm_per_tick, roi_override=roi_override)
    return result.cm_per_pixel
