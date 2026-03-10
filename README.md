# HeartVolume - Echocardiography Volume Analyzer

HeartVolume is a desktop Python app (Tkinter + OpenCV) to estimate cardiac volumes from echocardiography videos using three workflows: Simple method, Modified Simpson method, and Doppler method.

## Project title

**HeartVolume - Echocardiography Volume Analyzer**

## Project description

A practical GUI tool for semi-automatic echo analysis:
- frame selection on videos,
- automatic scale detection on the right ruler,
- interactive length/diameter measurements,
- EDV/ESV/SV/EF computation,
- optional cardiac output estimation from HR,
- Doppler-based stroke volume and CO estimation.

## Main features

- **Simple method**: Lendo/Dendo ED+ES measurements from an apical video.
- **Modified Simpson method**: synchronized 2x2 workflow on 4 videos (Apical, MV, PM, Apex).
- **Doppler method**: D_AO + VTI + HR based output.
- **Scale detection**:
  - right-side fixed ROI logic,
  - rising-front detection on right-edge pixel profile,
  - deterministic major ticks (every 5 minor ticks),
  - manual ROI adjustment and visual overlay.

## Project structure

- `main.py`: GUI entrypoint.
- `heartvolume/gui/app.py`: Tkinter application and workflows.
- `heartvolume/imaging/scale_detection.py`: scale detection pipeline.
- `heartvolume/imaging/video_tools.py`: frame selection and 2x2 interactive tools.
- `heartvolume/core/calculations.py`: volume and Doppler formulas.
- `heartvolume/data/discovery.py`: auto-discovery of data files.

## Quick start

```bash
python main.py
```

If your environment requires dependencies, install at least:

```bash
pip install opencv-python numpy
```

## Data

By default, the app tries to auto-detect files in the `data/` folder (Lendo, Dendo MV/PM/AP, D_AO).

## Notes

- This project is for educational/prototyping use.
- Measurements depend on video quality and ruler visibility.

