from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from heartvolume.core.models import Measurement


def measure_distance(frame: np.ndarray, label: str, window_name: str = "Mesure") -> Optional[Measurement]:
    """Measure a segment by clicking 2 points on the frame."""
    original = frame.copy()
    measurement = Measurement()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    def redraw() -> None:
        canvas = original.copy()
        cv2.putText(canvas, f"Mesure: {label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 255, 20), 2)
        if measurement.point1 is None:
            help_text = "Cliquez le point 1"
        elif measurement.point2 is None:
            help_text = "Cliquez le point 2"
        else:
            help_text = "Entree: valider | R: refaire | ESC: annuler"
        cv2.putText(canvas, help_text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if measurement.point1 is not None:
            cv2.circle(canvas, measurement.point1, 5, (0, 255, 255), -1)
        if measurement.point2 is not None:
            cv2.circle(canvas, measurement.point2, 5, (0, 255, 255), -1)
            cv2.line(canvas, measurement.point1, measurement.point2, (0, 255, 255), 2)
            cv2.putText(canvas, f"{measurement.distance_pixels():.1f} px", (measurement.point2[0] + 10, measurement.point2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(window_name, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _params: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if measurement.point1 is None:
            measurement.point1 = (x, y)
        elif measurement.point2 is None:
            measurement.point2 = (x, y)
        else:
            measurement.point1 = (x, y)
            measurement.point2 = None
        redraw()

    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(window_name)
            return None
        if key in (13, 10) and measurement.is_complete():
            cv2.destroyWindow(window_name)
            return measurement
        if key in (ord("r"), ord("R")):
            measurement = Measurement()
            redraw()

