import cv2
import numpy as np
from matplotlib import pyplot as plt

#Histogram Equalization
img = cv2.imread("lena.jpg")
#cv2.imshow("Image",img)
#cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#ngeubah gambarnya jadi gray
h = img.shape[0]# 0 itu untuk ambil heightnya
w = img.shape[1]# 1 itu untuk ambil widthnya

gray_counter = np.zeros(256, dtype=int)#untuk hitung intensitas. 256 itu jumlah index berdasarnya jumlah intensitas 0-255
for i in range(h):
    for j in range(w):
        gray_counter[gray[i][j]] += 1

equ = cv2.equalizeHist(gray) #ini proses equalizationnya
equ_counter = np.zeros(256,dtype=int)
for i in range(h):
    for j in range(w):
        equ_counter[equ[i][j]] += 1

clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(16,16)) # ini materi tambahan aja
cl = clahe.apply(gray)


plt.figure(figsize=(8,8))#canvas 

plt.subplot(211)#ini ngebuat jadi ada ukuran di dalam figure. paling kiri itu baris, yang tengah itu column,yang kanan itu index
plt.plot(gray_counter,'r',label = "before")
plt.legend(loc="upper left")
plt.ylabel("Quantity")
plt.xlabel("Intensity")
plt.axis([0,255, 0, gray_counter.max()])#untuk ngepasin nilai. jadi paling kiri itu 0 untuk x, 256 max x, 0 untuk y, gray blabla max y

plt.subplot(212)#ini ngebuat jadi ada ukuran di dalam figure. paling kiri itu baris, yang tengah itu column,yang kanan itu index
plt.plot(equ_counter,'b',label = "a")
plt.legend(loc="upper left")
plt.ylabel("Quantity")
plt.xlabel("Intensity")
plt.axis([0,255, 0, equ_counter.max()])#untuk ngepasin nilai. jadi paling kiri itu 0 untuk x, 256 max x, 0 untuk y, gray blabla max y

plt.show()

res = np.hstack((gray,equ,cl))#nampilin 2 gambar sekaligus secara horizontal
cv2.imshow("asd",res)
cv2.waitKey(0)





