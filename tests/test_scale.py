import cv2
from heartvolume.imaging.scale_detection import detect_scale_on_frame

images = [
    '../data/2nd Session/demo_Lendo_ED.PNG',
    '../data/2nd Session/demo_Lendo_ES.PNG',
    '../data/2nd Session/demo_Dendo_ED_MV.PNG',
    '../data/2nd Session/demo_Dendo_ED_PM.PNG',
    '../data/2nd Session/demo_Dendo_ED_AP.PNG'
]

for path in images:
    img = cv2.imread(path)
    if img is not None:
        scale = detect_scale_on_frame(img, 1.0)
        if scale:
            print(f'{path}: {scale:.6f} cm/px ({1/scale:.1f} px/cm)')
        else:
            print(f'{path}: Non détecté')
    else:
        print(f'{path}: Image non trouvée')

