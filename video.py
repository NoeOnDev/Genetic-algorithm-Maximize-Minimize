import cv2
import os

def crear_video(carpeta, numero_generaciones):
    carpeta_imagenes = carpeta
    nombre_video = 'video_de_evolucion/VideoAlgoritmoGenetico.avi'

    imagenes = [f"Generacion_{i}.png" for i in range(1, numero_generaciones + 1)]
    frame = cv2.imread(os.path.join(carpeta_imagenes, imagenes[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        nombre_video, cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))

    for imagen in imagenes:
        video.write(cv2.imread(os.path.join(carpeta_imagenes, imagen)))

    cv2.destroyAllWindows()
    video.release()