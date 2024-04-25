import numpy as np
import cv2 as cv

def draw_triangle(im, triangle, color=(0, 255, 0)):
    triangle = np.array([v[:-1] / v[-1] for v in triangle], np.uint16)
    cv.drawContours(im, [triangle.astype(int)], 0, color, -1)
    for v in triangle:
        cv.circle(im, tuple(v), 2, (255, 0, 255), -1)

def scale(vertex, sx, sy):
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]])
    return S @ vertex

def rotate(vertex, a):
    theta = np.radians(a)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    return R @ vertex

def translate(vertex, dx, dy):
    T = np.array([[1, 0, dx],
                  [0, 1, dy],
                  [0, 0, 1]])
    return T @ vertex

# Tamaño de la imagen
w = 500
h = 500
im = np.zeros((h, w, 3), np.uint8)

# Parámetros
angle = 30 #30
s = 3 #3
t = 200 #250

# triángulo original
triangle1 = np.array([[10, 10, 1], [70, 10, 1], [40, 60, 1]])

# transformaciones triángulo 2
triangle2 = translate(rotate(scale(translate(triangle1.T, -40, -30), s, s), angle), t, t)

# Dibujar 2
draw_triangle(im, triangle1)
draw_triangle(im, triangle2.T, color=(0, 130, 255))

cv.imwrite('Triangulo_transformado.jpg', im)