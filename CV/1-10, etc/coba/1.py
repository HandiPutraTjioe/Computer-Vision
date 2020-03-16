import cv2
import numpy as np
from matplotlib import pyplot as plt

# Show Image

img = cv2.imread("panda.jpg")
# cv2.imshow("Panda", img)
# cv2.waitKey()

##########################################################################################################

# Histogram Equalization

img1 = cv2.imread('panda.jpg')

img1_manual = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

h = img1.shape[0]
w = img1.shape[1]

gray_counter = np.zeros(256, dtype=int)
for i in range(h):
    for j in range(w):
        gray_counter[img1_manual[i][j]] += 1

equ = cv2.equalizeHist(img1_manual)
equ_counter = np.zeros(256, dtype=int)
for i in range(h):
    for j in range(w):
        equ_counter[equ[i][j]] += 1

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
cl = clahe.apply(img1_manual)

plt.figure(figsize=(8,8))

plt.subplot(211)
plt.plot(gray_counter,'r',label="Before")
plt.legend(loc="upper left")
plt.ylabel("Quantity")
plt.xlabel("Intensity")
plt.axis([0, 255, 0, gray_counter.max()])

plt.subplot(212)
plt.plot(equ_counter,'b',label="After")
plt.legend(loc="upper left")
plt.ylabel("Quantity")
plt.xlabel("Intensity")
plt.axis([0, 255, 0, equ_counter.max()])

plt.show()

res = np.hstack((img1_manual, equ, cl))
cv2.imshow("asd",res)
cv2.waitKey(0)

##########################################################################################################

img_instant = cv2.imread("panda.jpg", 0)

convert_img_manual = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

convert_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.subplot(131)
# plt.title("Convert image BGR to RGB")
# plt.imshow(convert_img1)

# plt.subplot(132)
# plt.title("Instant Image")
# plt.imshow(img_instant, 'gray')

# plt.subplot(133)
# plt.title("Convert image to gray")
# plt.imshow(convert_img, 'gray')

# plt.show()

##########################################################################################################

# Thresholding

# # 1. binary thresholding
# cth, img_bry = cv2.threshold(convert_img_manual, 127, 255, cv2.THRESH_BINARY)

# # 2. binary inverse thresholding
# _, img_inv_bry = cv2.threshold(convert_img_manual, 127, 255, cv2.THRESH_BINARY_INV)

# # 3. truncate thresholding
# _, img_trnc = cv2.threshold(convert_img_manual, 127, 255, cv2.THRESH_TRUNC)

# # 4. to zero thresholding
# _, img_to_zero = cv2.threshold(convert_img_manual, 127, 255, cv2.THRESH_TOZERO)

# # 5. to zero inverse
# _, img_to_zero_imv = cv2.threshold(convert_img_manual, 127, 255, cv2.THRESH_TOZERO_INV)

# # 6. otsu binary
# _, img_otsu = cv2.threshold(convert_img_manual, 127, 255, cv2.THRESH_OTSU)

# plt.subplot(231)
# plt.title("Binary Threshold")
# plt.imshow(img_bry, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(232)
# plt.title("Binary Inverse Threshold")
# plt.imshow(img_inv_bry, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(233)
# plt.title("Truncate Threshold")
# plt.imshow(img_trnc, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(234)
# plt.title("To zero Threshold")
# plt.imshow(img_to_zero, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(235)
# plt.title("To zero inverse")
# plt.imshow(img_to_zero_imv, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(236)
# plt.title("Otsu Binary")
# plt.imshow(img_otsu, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.show()

##########################################################################################################

# filtering

# ada 4 jenis filtering secara auto, ada juga secara manual 

# Mean manual

# def manual_mean_blur(img, size):
#     img_array = np.array(img)
#     img_height, img_width = img_array.shape
#     for i in range(img_height - size - 1):
#         for j in range(img_width - size - 1):
#             arr = np.array(img_array[i:i + size, j:j + size]).flatten()
#             mean = np.mean(arr)
#             img_array[i + size // 2][j + size // 2] = mean
#     return img_array

# # median manual

# def manual_median_blur(img, size):
#     img_array = np.array(img)
#     img_height, img_width = img_array.shape
#     for i in range(img_height - size - 1):
#         for j in range(img_width - size - 1):
#             arr = np.array(img_array[i:i + size, j:j + size]).flatten()
#             median = np.median(arr)
#             img_array[i + size // 2][j + size // 2] = median
#     return img_array

# img_mean_blur = manual_mean_blur(convert_img_manual, 5)
# img_median_blur = manual_median_blur(convert_img_manual, 5)

# # cara auto nya

# img_mean = cv2.blur(convert_img_manual, (5,5))
# img_median = cv2.medianBlur(convert_img_manual, 5)
# img_gausian = cv2.GaussianBlur(convert_img_manual, (5,5), 5)
# img_bilateral = cv2.bilateralFilter(convert_img_manual, 5, 75, 75)

# plt.subplot(231)
# plt.title("Manual Mean Blur")
# plt.imshow(img_mean_blur, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(232)
# plt.title("Manual Median Blur")
# plt.imshow(img_median_blur, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(233)
# plt.title("Mean Blur")
# plt.imshow(img_mean, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(234)
# plt.title("Median Blur")
# plt.imshow(img_median, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(235)
# plt.title("Gaussian Blur")
# plt.imshow(img_gausian, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(236)
# plt.title("Bilateral Blur")
# plt.imshow(img_bilateral, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.show()

##########################################################################################################

# Edge Detection ada 3 jenis
# - Canny
# - Sobel
# - Laplace

# img1 = cv2.imread('fruits.jpg')
# img_h = img1.shape[0]
# img_w = img1.shape[1]

# img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# # Canny
# canny = cv2.Canny(img_gray, 100, 200)
# cv2.imshow('1', canny)
# cv2.waitKey()

# # laplacian
# lapla = cv2.Laplacian(img_gray, cv2.CV_64F)
# lapla64 = np.absolute(lapla)
# lapla8 = np.uint8(lapla64)
# # cv2.imshow('1', lapla8)

# # sobel 
# sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
# sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)

# img_list = [img_gray, canny, lapla,
#             lapla8, sobel_x, sobel_y]

# title_list = ['Gray','Canny','Laplacian','Laplacian 8u',
#               'Sobel X','Sobel Y']

# plt.figure(1, figsize=(7, 7))
# for idx, (curr_image, curr_title) in enumerate(zip(img_list, title_list)):
#     plt.subplot(3, 2, (idx + 1))
#     plt.imshow(curr_image, 'gray')
#     plt.title(curr_title)
#     plt.xticks([])
#     plt.yticks([])

# plt.show()

##########################################################################################################

