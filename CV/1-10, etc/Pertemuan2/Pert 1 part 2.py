#Display Image
import cv2

img = cv2.imread("Lena.jpg", 1) #Deklarasi dulu untuk gambarnya, 1 untuk warna, 0 untuk greyscale

cv2.imshow("Gambar", img) #Menampilkan gambar

key = cv2.waitKey(0) #Untuk Tahan Gambar
#0 itu delaynya kalau di isi 1 berarti setelah 1 detik baru bisa input

#untuk settingan input yang di terima
if key == 27: #bila tombol esc diketik
    cv2.destroyAllWindows()
elif key == ord('s'): #bila s di ketik
    cv2.imwrite("Lena.png", img) #untuk save ulangs
    cv2.destroyAllWindows()