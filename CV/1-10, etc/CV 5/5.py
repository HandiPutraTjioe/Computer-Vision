# Corner Detection
##########################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

img = cv2.imread('shape.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Algoritma harris corner detection
img_gray = np.float32(img_gray)
harris_corner = cv2.cornerHarris(img_gray, 3, 5, 0.01)

img_harris = img.copy()
img_harris[harris_corner > 0.01 * harris_corner.max()] = [0, 0, 255]

plt.subplot(121)
plt.imshow(img_harris)
plt.xticks([])
plt.yticks([])

_, thresh = cv2.threshold(harris_corner, harris_corner.max() * 0.01, 255, cv2.THRESH_BINARY)

# # bentuk integer
thresh = np.uint8(thresh)

_, _, _, corner_centroids = cv2.connectedComponentsWithStats(thresh)

corner_centroids = np.float32(corner_centroids)

# # berhentiin loopinganny
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# # algoritma SubPix
enhanced_corner = cv2.cornerSubPix(img_gray, corner_centroids, (2, 2), (-1, -1), criteria)

img_subpix = img.copy()
corner_centroids = np.int16(corner_centroids) # nilainy > dari int8 maks
for centroids in corner_centroids:
    centroids_x, centroids_y = centroids
    img_subpix[centroids_y, centroids_x] = [0, 255, 0]

enhanced_corner = np.int16(enhanced_corner)
for corner in enhanced_corner:
    corner_x, corner_y = corner
    img_subpix[corner_y, corner_x] = [255, 0, 0]

plt.subplot(122)
plt.imshow(img_subpix)

plt.show()


# Fast & ORB
fast = cv2.FastFeatureDetector_create(threshold=50)
orb = cv2.ORB_create()

directory = '05'
for filename in os.listdir(directory):
    img = cv2.imread(directory + "/" + filename)

    keypoint_fast = fast.detect(img, None)
    keypoint_orb = orb.detect(img, None)

    img_fast = img.copy()
    img_orb = img.copy()

    cv2.drawKeypoints(img, keypoint_fast, img_fast, (0, 255, 0))
    cv2.drawKeypoints(img, keypoint_orb, img_orb, (255, 0, 0))
    img_fast = cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB)
    img_orb = cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB)

    plt.subplot(121) # down 1, side 2, idx 1
    plt.imshow(img_fast)

    plt.subplot(122)
    plt.imshow(img_orb)

    plt.show()