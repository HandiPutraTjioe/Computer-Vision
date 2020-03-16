import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 1 show image
# img_panda = cv2.imread("panda.jpg")
# cv2.imshow("panda", img_panda)
# cv2.waitKey(0)

###########################################################################################################

# 2 equalization
img_instant = cv2.imread('lena.jpg', 0)

img_lena = cv2.imread("lena.jpg")

img_gray = cv2.cvtColor(img_lena, cv2.COLOR_BGR2RGB)

img_gray_manual = cv2.cvtColor(img_lena, cv2.COLOR_BGR2GRAY)

# h = img_lena.shape[0]
# w = img_lena.shape[1]

# gray_counter = np.zeros(256, dtype=int)
# for i in range(h):
#     for j in range(w):
#         gray_counter[img_gray[i][j]] += 1

# equ = cv2.equalizeHist(img_gray)
# equ_counter = np.zeros(256, dtype=int)
# for i in range(h):
#     for j in range(w):
#         equ_counter[equ[i][j]] += 1

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
# cl = clahe.apply(img_gray)

# plt.figure(figsize=(8,8))
# plt.subplot(211)
# plt.plot(gray_counter,'r',label="Before")
# plt.legend(loc="upper left")
# plt.ylabel("Quantity")
# plt.xlabel("Intensity")
# plt.axis([0, 255, 0, gray_counter.max()])

# plt.subplot(212)
# plt.plot(equ_counter,'b',label="After")
# plt.legend(loc="upper left")
# plt.ylabel("Quantity")
# plt.xlabel("Intensity")
# plt.axis([0, 255, 0, equ_counter.max()])

# plt.show()

# res = np.hstack((img_gray,equ,cl))
# cv2.imshow("asd",res)
# cv2.waitKey(0)

###########################################################################################################

# 3 Threshold, Filtering
# Thresholding

# plt.subplot(131)
# plt.title("Normal Image")
# plt.imshow(img_gray)
# plt.xticks([])
# plt.yticks([])

# plt.subplot(132)
# plt.title("Instant Gray Scale")
# plt.imshow(img_instant,'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(133)
# plt.title("Manual Gray Scale")
# plt.imshow(img_gray_manual,'gray')
# plt.xticks([])
# plt.yticks([])

# plt.show()

# test1, bnry_thrsh = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_BINARY)
# _, bnry_inv_thrsh = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_BINARY_INV)
# _, trnct_thrsh = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_TRUNC)
# _, to_zero_thrsh = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_TOZERO)
# _, to_zero_inv = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_TOZERO_INV)
# _, otsu_bnry = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_OTSU)

# plt.subplot(231)
# plt.title("Binary thresh")
# plt.imshow(bnry_thrsh, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(232)
# plt.title("Binary Inv thresh")
# plt.imshow(bnry_inv_thrsh, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(233)
# plt.title("Truncate thresh")
# plt.imshow(trnct_thrsh, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(234)
# plt.title("To Zero thresh")
# plt.imshow(to_zero_thrsh, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(235)
# plt.title("To Zero Inv")
# plt.imshow(to_zero_inv, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(236)
# plt.title("Otsu Binary")
# plt.imshow(otsu_bnry, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.show()

# Filtering
# Manual Mean
# def manual_mean_blur(img_lena, size):
#     img_array = np.array(img_lena)
#     img_h, img_w = img_array.shape
#     for i in range(img_h - size - 1):
#         for j in range(img_w - size - 1):
#             arr = np.array(img_array[i:i + size, j:j + size]).flatten()
#             mean = np.mean(arr)
#             img_array[i + size // 2][j + size // 2] = mean
#     return img_array

# def manual_median_blur(img_lena, size):
#     img_array = np.array(img_lena)
#     img_h, img_w = img_array.shape
#     for i in range(img_h - size - 1):
#         for j in range(img_w - size - 1):
#             arr = np.array(img_array[i:i + size, j:j + size]).flatten()
#             median = np.median(arr)
#             img_array[i + size // 2][j + size // 2] = median
#     return img_array

# img_manual_mean = manual_mean_blur(img_gray_manual, 5)
# img_manual_median = manual_median_blur(img_gray_manual, 5)

# # cara auto mean, median , gaussian, bilateral

# img_mean = cv2.blur(img_gray_manual, (5, 5))
# img_median = cv2.medianBlur(img_gray_manual, 5)
# img_gaussian = cv2.GaussianBlur(img_gray_manual, (5, 5), 5)
# img_bilateral = cv2.bilateralFilter(img_gray_manual, 5, 75, 75)

# plt.subplot(231)
# plt.title("Manual Mean Blur")
# plt.imshow(img_manual_mean,'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(232)
# plt.title("Manual Median Blur")
# plt.imshow(img_manual_median,'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(233)
# plt.title("Mean Blur")
# plt.imshow(img_mean,'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(234)
# plt.title("Median Blur")
# plt.imshow(img_median,'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(235)
# plt.title("Gaussian Blur")
# plt.imshow(img_gaussian,'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(236)
# plt.title("Bilateral Blur")
# plt.imshow(img_bilateral,'gray')
# plt.xticks([])
# plt.yticks([])

# plt.show()

###########################################################################################################

# 4. Edge Detection
# Canny, Laplacian, Sobel

img = cv2.imread('fruits.jpg')
img_h = img.shape[0]
img_w = img.shape[1]

img_gray_manual1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny = cv2.Canny(img_gray_manual1, 100, 200)
# cv2.imshow('1', Canny)
# cv2.waitKey(0)

# Laplacian
# laplacian = cv2.Laplacian(img_gray_manual1, cv2.CV_64F)
# laplacian_64f = np.absolute(laplacian)
# laplacian_8u = np.uint8(laplacian_64f)
# # cv2.imshow('2', laplacian_8u)
# # cv2.waitKey(0)

# # Sobel
# sobel_x = cv2.Sobel(img_gray_manual1, cv2.CV_64F, 1, 0, ksize=5)
# sobel_y = cv2.Sobel(img_gray_manual1, cv2.CV_64F, 0, 1, ksize=5)

# img_list = [img_gray_manual1, Canny, laplacian, 
#             laplacian_8u, sobel_x, sobel_y]

# title_list = ['Gray','Canny','Laplacian',
#               'Laplacian 8U','Sobel X','Sobel Y']

# plt.figure(1, figsize=(7, 7))
# for idx, (curr_image, curr_title) in enumerate(zip(img_list, title_list)):
#     plt.subplot(3, 2, (idx + 1))
#     plt.imshow(curr_image,'gray')
#     plt.title(curr_title)
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# sobel_kernel = np.array(
#     [
#         -1, 0, 1,
#         -2, 0, 2,
#         -1, 0, 1
#     ]
# )
# ksize = 3

# img1 = cv2.GaussianBlur(img_gray_manual1, (3,3), 500)
# img1_copy = img1.copy()
# for i in range(img_h - ksize - 1):
#     for j in range(img_w - ksize - 1):
#         img_matrix = img1[i:(i+ksize), j:(j+ksize)]
#         flat_matrix = img_matrix.flatten()
#         res = np.convolve(flat_matrix, sobel_kernel, 'valid')
#         img1_copy[i + ksize // 2][j + ksize // 2] = res[0]
# plt.imshow(img1_copy, 'gray')
# plt.xticks([])
# plt.yticks([])
# plt.show()

###########################################################################################################

# 5. Corner Detection
img_shape = cv2.imread("shape.png")
img_gray_shape = cv2.cvtColor(img_shape, cv2.COLOR_BGR2GRAY)

# Harris
img_gray2 = np.float32(img_gray_shape)
harris_corner = cv2.cornerHarris(img_gray2, 3, 5, 0.01)
img_harris = img_shape.copy()
img_harris[harris_corner > 0.01 * harris_corner.max()] = [0, 0, 255]

plt.subplot(121)
plt.imshow(img_harris)
plt.xticks([])
plt.yticks([])

_, thresh = cv2.threshold(harris_corner, harris_corner.max() * 0.01, 255, cv2.THRESH_BINARY)
thresh = np.uint8(thresh)
_, _, _, corner_centroid = cv2.connectedComponentsWithStats(thresh)
corner_centroid = np.float32(corner_centroid)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# Subpix
enchanced_centroid = cv2.cornerSubPix(img_gray2, corner_centroid, (2, 2), (-1, -1), criteria)

img_subpix = img_shape.copy()
corner_centroid = np.int16(corner_centroid)
for cenroids in corner_centroid:
    x, y = cenroids
    img_subpix[y, x] = [0, 0, 255]

enchanced_centroid = np.int16(enchanced_centroid)
for corner in enchanced_centroid:
    x, y = corner
    img_subpix[y, x] = [255, 0, 0]
plt.subplot(122)
plt.imshow(img_subpix)
plt.show()

# Fast & Orb

fast = cv2.FastFeatureDetector_create(threshold=50)
orb = cv2.ORB_create()

directory = '05'
for filename in os.listdir(directory):
    img4 = cv2.imread(directory + "/" + filename)

    keypoint_fast = fast.detect(img4, None)
    keypoint_orb = orb.detect(img4, None)

    img_fast = img4.copy()
    img_orb = img4.copy()

    cv2.drawKeypoints(img4, keypoint_fast, img_fast, (0, 255, 0))
    cv2.drawKeypoints(img4, keypoint_orb, img_orb, (255, 0, 0))
    img_fast = cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB)
    img_orb = cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(img_fast)

    plt.subplot(122)
    plt.imshow(img_orb)

    # plt.show()

###########################################################################################################

# 6. Image Matching

img_box = cv2.imread("06/box.png")
img_scene = cv2.imread("06/box_in_scene.png")

#surf
surf = cv2.xfeatures2d.SURF_create()
kp_box, desc_box = surf.detectAndCompute(img_box, None)
kp_scene, desc_scene = surf.detectAndCompute(img_scene, None)

#Flann
KDTREE_INDEX = 0
flann = cv2.FlannBasedMatcher(dict(algorithm = KDTREE_INDEX))
desc_box = desc_box.astype('f')
desc_scene = desc_scene.astype('f')

matches = flann.knnMatch(desc_box, desc_scene, k = 2)

valid_match = []
for i in range(len(matches)):
    valid_match.append([0, 0])

total_match_valid = 0
for idx, (p, q) in enumerate(matches):
    if(p.distance < 0.7 * q.distance):
        valid_match[idx] = [1, 0]
        total_match_valid += 1
    else:
        continue

print(total_match_valid)
img_result = cv2.drawMatchesKnn(
    img_box, kp_box,
    img_scene, kp_scene,
    matches, None,
    matchesMask = valid_match
)
cv2.imshow('1',img_result)
cv2.waitKey(0)