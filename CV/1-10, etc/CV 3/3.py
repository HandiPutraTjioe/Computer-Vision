 import cv2
import numpy as np
import matplotlib.pyplot as plt

''' gray scale '''
# cara 1 : imread
img_gray_instant = cv2.imread('lena.jpg', 0)

# cara 2 : cvtColor
img = cv2.imread('lena.jpg')
# convert dari BGR ke RGB
converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# convert dari BGR ke gray scale
img_gray_manual = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# mathplotlib color RGB, CV2 color BGR

# plt.subplot(131)
# plt.title('Normal Image')
# plt.imshow(converted_img)
# plt.xticks([])
# plt.yticks([])

# plt.subplot(132)
# plt.title('Instant Grayscale Image')
# plt.imshow(img_gray_instant, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(133)
# plt.title('Manual Grayscale Image')
# plt.imshow(img_gray_manual, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.show()

''' Thresholding '''
# mengelompokkan siapa yg warna ny terang dan siapa yg warnanny gelap
# ada beberapa jenis :
# binary, binary_inverse, truncate, to zero, to zero inverse

# 127 nilai patokannya, threshold mengembalikan 2 value : retval dan image(v)
# ret masuk ke retval, img_binary ke image

# { src >= threshold } = putih
# { src < threshold } = black

# binary inverse
# { src >= threshold 'black' || src < threshold 'white' } 

# truncate
# { src > threshold = src jadi threshold || src < threshold = jadi src = src}
# 129 > 127 warnanya sama 

# to zero
# { src < threshold maka warnain hitam sisany warna aslinya || src > threshold maka src = src }

# # 1. Binary Thresholding 
# ret, img_binary = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_BINARY) 

# # 2. Binary Inverse Thresholding
# _, img_binary_inv = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_BINARY_INV)

# # 3. Truncate Thresholding
# _, img_trunc = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_TRUNC)

# # 4. To Zero Thresholding
# _, img_tozero = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_TOZERO)

# # 5. To zero Inverse
# _, img_tozero_inv = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_TOZERO_INV)

# # 6. Otzu Binary
# _, img_otsu = cv2.threshold(img_gray_manual, 127, 255, cv2.THRESH_OTSU)

# plt.subplot(231)
# plt.title('Binary Image')
# plt.imshow(img_binary, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(232)
# plt.title('Binary Inverse Image')
# plt.imshow(img_binary_inv, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(233)
# plt.title('Truncate Image')
# plt.imshow(img_trunc, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(234)
# plt.title('To Zero Image')
# plt.imshow(img_tozero, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(235)
# plt.title('To Zero Inverse Image')
# plt.imshow(img_tozero_inv, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.subplot(236)
# plt.title('Otzu Image')
# plt.imshow(img_otsu, 'gray')
# plt.xticks([])
# plt.yticks([])

# plt.show()


''' Filtering '''
# 1. Blur
# ada 4 jenis :
# cara 1 : manual
# cara blur berdasarkan mean dan median
# Mean

def manual_mean_blur(img, size):
    img_array = np.array(img)
    img_height, img_width = img_array.shape
    for i in range(img_height - size - 1):
        for j in range(img_width - size - 1):
            arr = np.array(img_array[i:i + size, j:j + size]).flatten()
            mean = np.mean(arr)
            img_array[i + size // 2][j + size // 2] = mean # [i + size // 2] pembagian dengan pembulatan kebawah
    return img_array

def manual_median_blur(img, size):
    img_array = np.array(img)
    img_height, img_width = img_array.shape
    for i in range(img_height - size - 1):
        for j in range(img_width - size - 1):
            arr = np.array(img_array[i:i + size, j:j + size]).flatten()
            median = np.median(arr)
            img_array[i + size // 2][j + size // 2] = median # [i + size // 2] pembagian dengan pembulatan kebawah
    return img_array

# # cara manual blur image
img_manual_mean = manual_mean_blur(img_gray_manual, 5)
img_manual_median = manual_median_blur(img_gray_manual, 5)

# # auto blur, cara cv2
# # 4 cara blur : mean, median, gausian, bilateral

# # mean
img_mean = cv2.blur(img_gray_manual, (5,5)) # img, ukuran bentuk tuple p dan l

# median
img_median = cv2.medianBlur(img_gray_manual, 5) # image, 1 nilai aja (persegi)

# Gaussian Blur
img_gausian = cv2.GaussianBlur(img_gray_manual, (5, 5), 5)

# Bilateral
img_bilateral = cv2.bilateralFilter(img_gray_manual, 5, 75, 75) # img, ukuran 1 nilai aj, sigma color(perbedaan warnanya), sigma space

plt.subplot(231)
plt.title('Manual Mean')
plt.imshow(img_manual_mean, 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.title('Manual Median')
plt.imshow(img_manual_median, 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(233)
plt.title('Mean Blur')
plt.imshow(img_mean, 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(234)
plt.title('Median Blur')
plt.imshow(img_median, 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(235)
plt.title('Gaussian Blur')
plt.imshow(img_gausian, 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(236)
plt.title('Bilateral Blur')
plt.imshow(img_bilateral, 'gray')
plt.xticks([])
plt.yticks([])

plt.show()