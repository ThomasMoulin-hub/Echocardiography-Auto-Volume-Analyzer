"""HeartVolume GUI entrypoint.

This file keeps a small compatibility layer for existing imports
(e.g. test_scale.py importing detect_scale_on_frame).
"""


from heartvolume.gui.app import run_gui


def main() -> None:
    run_gui()


if __name__ == "__main__":
    main()

