import cv2 as cv
import numpy as np

#Punto 1: Cambiar color de fondo
def color_fondo(img,color):
    img_1 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_green = np.array([10, 110, 155])
    upper_green = np.array([90, 255, 255])
    mask = cv.inRange(img_1, lower_green, upper_green)
    mask = cv.bitwise_not(mask)
    img[mask == 0] = color
    return(img)

#Punto 2 Operadores punto
# 2.1 Brillo y contraste
def brillo_contraste(img,alfa,beta):
    x,y,z=img.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                img[i, j, k] = np.clip(alfa * img[i, j, k] + beta, 0, 255).astype(np.uint8)
    return(img)

'''Otra forma:
def brillo_contraste2(img,alfa,beta):
    img = (img.astype('float32') * alfa)
    img = (img.astype('float32') + beta)
    return(img)
cv.imwrite("gato_nieve_contrasteybrillo2.jpg",brillo_contraste2(cv.imread("gato_nieve.jpg"),contraste,brillo))'''

# 2.2 Image Matting
def Image_matting(img,fondo):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_green = np.array([50, 60, 95])
    upper_green = np.array([90, 255, 255])
    # Crear una máscara para el color verde
    mask = cv.inRange(img_hsv, lower_green, upper_green)
    gato_sin_fondo = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))
    fondo_redimensionado = cv.resize(fondo, (img.shape[1], img.shape[0]))
    nueva_imagen = cv.bitwise_and(fondo_redimensionado, fondo_redimensionado, mask=mask) + gato_sin_fondo
    return(nueva_imagen)

#Punto 3 Implementar Operador Sobel
def Sobel(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx_kernel = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobely_kernel = np.array([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]])
    sobelx = cv.filter2D(img, -1, sobelx_kernel)
    sobely = cv.filter2D(img, -1, sobely_kernel)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    #umbral = 100
    #sobel[sobel < umbral] = 100 
    cv.imwrite('sobelx.jpg',sobelx)
    cv.imwrite('sobely.jpg',sobely)
    return(sobel)

col = [128,0,128] #morado
punto1=color_fondo(cv.imread('cat.jpg'),col)
cv.imwrite("punto1.jpg",punto1)

contraste = 1.5
brillo = 0.
punto2_1 = brillo_contraste(cv.imread("cat2.jpg"),contraste,brillo)
cv.imwrite("punto2_1.jpg",punto2_1)

fondo = cv.imread('fondo.jpg')
punto2_2 = Image_matting(cv.imread("cat.jpg"),fondo)
cv.imwrite('punto2_2.jpg',punto2_2)

cv.imwrite('sobel.jpg',Sobel(cv.imread("Capilla.jpeg")))