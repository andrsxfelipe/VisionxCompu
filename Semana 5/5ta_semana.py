import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

def kmeans_sklearns(img,clusters):
    image2 = img.reshape((-1,3))

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(image2)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    segmented_img = centroids[labels].reshape(img.shape)
    return(segmented_img)

def Opencv(img,clusters):
    img2 = img.reshape((-1, 3)).astype(np.float32)

    c = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centroids = cv.kmeans(img2, clusters, None, c, 10, cv.KMEANS_RANDOM_CENTERS)

    centroids = np.uint8(centroids)
    segmented_img = centroids[labels.flatten()]
    segmented_img = segmented_img.reshape(img.shape)
    return(segmented_img)


image = cv.imread("starcrop.jpeg")

sk = kmeans_sklearns(image,2)
cv.imwrite('segmentacion_sk.jpg',sk)

ocv =  Opencv(image,2)
cv.imwrite('segmentacion_cv.jpg',ocv)

#Las diferencias no parecen ser muchas, ahora con una imagen algo más compleja

pandarojo = cv.imread("pandarojo.jpg")
cv.imwrite('pandarojo_segmentado_sk.jpg',kmeans_sklearns(pandarojo,5))
cv.imwrite('pandarojo_segmetado_opencv.jpg',Opencv(pandarojo,5))

#No hay mucha diferencia