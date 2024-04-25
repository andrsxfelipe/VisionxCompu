import cv2
import numpy as np

def panoramica(img1,img2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None) #Ptos claves y descriptores
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2) #Emparejar descriptores
    matches = sorted(matches, key=lambda x: x.distance) #Ordenarlos segun la distancia de sus emparejamientos
    coincidencias = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:60], None, flags=2)
    points1 = np.zeros((len(matches), 2), dtype=np.float32) #Ubicación de los mejores emparejamientos
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img2.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    return(coincidencias,result)


img1 = cv2.imread('img1.jpg') 
img2 = cv2.imread('img2.jpg')
img3 = cv2.imread('img3.jpg')

panoramica1 = panoramica(img2,img3)

cv2.imwrite('Mejores_matches1.jpg',panoramica1[0])
cv2.imwrite('Panoramica1.jpg',panoramica1[1])

panoramica2 = panoramica(img1,img2)
cv2.imwrite('Mejores_matches2.jpg',panoramica2[0])
cv2.imwrite('Panoramica2.jpg',panoramica2[1])

# Metodo encontrado usando STITCH que puede unir varias imágenes (No válido
# para el objetivo de la clase)
'''
image_paths=['img1.jpg','img2.jpg','img3.jpg']
imgs = []
for i in range(len(image_paths)):
    imgs.append(cv2.imread(image_paths[i]))
    imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.4,fy=0.4)

stitchy=cv2.Stitcher.create() 
(dummy,output)=stitchy.stitch(imgs) 

if dummy != cv2.STITCHER_OK: 
    print("Panomarización insatisfactoria") 
else:  
    print('¡Panorama listo!') '''