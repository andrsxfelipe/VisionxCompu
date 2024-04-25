import cv2
import numpy as np
import math
W = 100
H = 100
im=np.zeros((W,H,3),np.uint8)

def linea(im):
    im1=im.copy()
    for i in range(100):
        im1[i,i] = (255,255,255)
    cv2.imwrite('Linea.jpg', im1)

def cuadrados(im):
    im2=im.copy()
    for i in range(19,80):
        for j in range(19,80):
            im2[i,j] = (0,0,255)

    for i in range(29,70):
        for j in range(29,70):
            im2[i,j] = (0,255,0)

    for i in range(39,60):
        for j in range(39,60):
            im2[i,j] = (255,0,0)
    for i in range(44,55):
        for j in range(44,55):
            im2[i,j] = (0,0,0)
    cv2.imwrite('Cuadrados.jpg', im2)
def onda(im,lines,frequency):
    im3=im.copy()
    color=[255,0,0]
    for i in range(lines):
        amplitude = int(H / (50))  # La amplitud de la línea varía con la frecuencia
        y_offset = H // (lines + 1) * (i + 1)
    
    # Generar puntos basados en la función seno
        x = np.arange(0, W)
        y = amplitude * np.sin(2 * np.pi * frequency * x / W) + y_offset
    
    # Convertir los puntos a enteros
        points = np.column_stack((x, y)).astype(np.int32)
    # Dibujar la línea
        cv2.polylines(im3, [points], isClosed=False, color=color, thickness=1)
        #color[0]-=25
        color[1]+=20
        color[2]+=20
    cv2.imwrite('Ondas.jpg', im3)


linea(im)
cuadrados(im)
lineas = 10
frecuencia = 20 
onda(im,lineas,frecuencia)