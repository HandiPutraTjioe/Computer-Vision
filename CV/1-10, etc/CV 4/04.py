import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('fruits.jpg')
img_h = img.shape[0]
img_w = img.shape[1]

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny
canny = cv2.Canny(img_gray, 100, 200)
# cv2.imshow('1', canny)
# # cv2.waitKey(0)

# Laplacian
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
laplacian_64f = np.absolute(laplacian)
laplacian_8u = np.uint8(laplacian_64f)

# |   |   | 0-255
# |   |   | unsigned integer
# |   |   | 8 -> 1 byte, why 64F because laplacian matrix 

# Sobel
# matriksny bisa detect edge secara ototmatis (horizontal dan vertical)
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)   # 0 -> horizontal, 1 -> vertical, ksize -> 1 syaratnya : harus ganjil dan gk boleh 1 biar ketemu tengahnya
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)

image_list = [img_gray, canny, laplacian, 
              laplacian_8u, sobel_x, sobel_y]

title_list = ['Gray','Canny','Laplacian',
            'Laplacian 8U','Sobel X','Sobel Y']

plt.figure(1, figsize=(7, 7))
for idx, (curr_image, curr_title) in enumerate(zip(image_list, title_list)):
    plt.subplot(3, 2, (idx + 1))
    plt.imshow(curr_image, 'gray')
    plt.title(curr_title)
    plt.xticks([]) # kalau mau namaian grafik pakai xticks or yticks or empty ([])
    plt.yticks([])
plt.show()


sobel_x_kernel = np.array(
    [
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    ]
)

ksize = 3

img_gray = cv2.GaussianBlur(img_gray, (3, 3), 500)
gray_copy = img_gray.copy()
for i in range(img_h - ksize - 1):
    for j in range(img_w - ksize - 1):
        img_matrix = img_gray[i:(i+ksize), j:(j+ksize)]
        flat_img_matrix = img_matrix.flatten()
        
        res = np.convolve(flat_img_matrix, sobel_x_kernel, 'valid') # hanya terima array 1 dimensi, masalahnya img_matrix masih 2 dimensi, pake flatten, tapi convolve gk sesuai dengan yg kita inginkan
        gray_copy[i+ksize//2][j+ksize//2] = res[0]
plt.imshow(gray_copy, 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

# cth : [1,2,3,4,5,6] -> ada 6 - (size-1) = 4
#      1 2 3
#         2 3 4          4 kali
#             3 4 5
#                 4 5 6