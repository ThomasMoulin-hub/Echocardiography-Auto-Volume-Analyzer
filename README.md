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

## Experimental Features: Automatic Ellipse Tracking

This project also includes experimental scripts intended to automatically track an ellipse across ultrasound video frames. This feature is located in the `heartvolume/imaging/automaticTracking/` folder.

> ⚠️ **DISCLAIMER:** This feature is **highly experimental** and **does not work very well**. Tracking the endocardium border on noisy ultrasound footage is a complex computer vision task. The current scripts are a proof-of-concept.

I have explored several techniques to tackle this issue, though none have yielded perfect results yet. Below are some examples/placeholders of the attempts:

1. **Approach 1: [Name/Technique 1]**
   *(Description of what you tried, e.g., Optical flow, Thresholding, etc.)*
   <br>
   *(Placeholder for a GIF/Video demonstration)*
   <br>
   `<img src="assets/method1_tracking_overlay_dendo.gif" width="400">`
   `<img src="assets/method1_tracking_overlay_lendo.gif" width="400">`

2. **Approach 2: [Name/Technique 2]**
   *(Description of what you tried, e.g., Active Contours / Snakes, Machine Learning, etc.)*
   <br>
   *(Placeholder for a GIF/Video demonstration)*
   <br>
   `<img src="path/to/demo_video_2.gif" width="400">`

3. **Approach 3: [Name/Technique 3]**
   *(Description of what you tried, e.g., Thresholding inside->outside based tracking)*
   <br>
   *(Placeholder for a GIF/Video demonstration)*
   <br>
   `<img src="path/to/demo_video_3.gif" width="400">`

*(Note: GitHub and most markdown viewers do not natively support embedding `.mp4` videos directly using standard markdown syntax. The best practice is to convert short preview clips into animated `.gif` files and display them as images, as shown above.)*

## Project structure

- `main.py`: GUI entrypoint.
- `heartvolume/gui/app.py`: Tkinter application and workflows.
- `heartvolume/imaging/scale_detection.py`: scale detection pipeline.
- `heartvolume/imaging/video_tools.py`: frame selection and 2x2 interactive tools.
- `heartvolume/imaging/automaticTracking/`: Experimental scripts for ellipse fitting/tracking.
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

By default, the app tries to auto-detect files in the `data/2nd Session` folder (Lendo, Dendo MV/PM/AP, D_AO).