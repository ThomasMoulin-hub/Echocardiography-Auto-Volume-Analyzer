from __future__ import annotations

import os
from typing import Optional

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP")


def data_dir() -> str:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "data/2nd Session")


def _find_file(keywords: list[str], extensions: tuple[str, ...]) -> Optional[str]:
    folder = data_dir()
    if not os.path.isdir(folder):
        return None

    for filename in os.listdir(folder):
        if not filename.endswith(extensions):
            continue
        low = filename.lower()
        if all(k.lower() in low for k in keywords):
            return os.path.join(folder, filename)
    return None


def find_video(keywords: list[str]) -> Optional[str]:
    return _find_file(keywords, VIDEO_EXT)


def find_image(keywords: list[str]) -> Optional[str]:
    return _find_file(keywords, IMAGE_EXT)


def auto_detect_videos() -> dict[str, Optional[str]]:
    return {
        "apical": find_video(["lendo"]) or find_video(["apical"]) or find_video(["4ch"]),
        "mv": find_video(["dendo", "mv"]) or find_video(["mv"]),
        "pm": find_video(["dendo", "pm"]) or find_video(["pm"]),
        "apex": find_video(["dendo", "ap"]) or find_video(["apex"]),
        "doppler": find_video(["doppler"]),
        "d_ao": find_video(["d_ao"]) or find_video(["dao"]),
    }


def auto_detect_images() -> dict[str, Optional[str]]:
    return {
        "d_ao": find_image(["d_ao"]) or find_image(["dao"]),
        "vmax": find_image(["vmax"]) or find_image(["pulsed"]),
    }

