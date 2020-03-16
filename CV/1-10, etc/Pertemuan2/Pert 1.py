# pagar untuk komen 
# untuk komen banyak ctrl+/

#output
print("Hello World") 

#buat variabel
number = 10
flt = 10.5
name = "yesun"

print(number," ", flt, " ", name)

#input
binusian = input("Input your Binusian = ") #kalau gini inputnya default string
binusian = int(input("Input your Binusian = ")) #kalau input berupa int
print(binusian)


#looping
for i in range(10):     #artinya i di mulai dari 0 lalu kurang dari 10
                        #kalau mau buat start indexnya bukan dari 0 maka range(2,10) berarti mulai dari 2 hingga 10
                        #bila mau di buat tambah atau kurang lebih dari 1 maka buat (2,10,3) berarti looping selalu tambah 2
    print(i)
    print("loop ", i+1)

a = 10 #untuk while harus di deklarasi dahulu

while a < 20 : #berarti 0 - 19 maka looping sebanyak 20 kali
    print("Hello")
    a = a+1


#Selection
num = 10

if num % 2 == 0:
    print("Even Number")
elif num > 20:
    print("Big Number")
else:
    print("Odd Number")

for i in range(20):
    if i % 2 == 0 and i > 3: #and pengganti dan dan kalau atau atau menggunakan or
        print("*")
    else:
        print("#")


#================================================================================================================#

#numpy => untuk array yg lebih dari 2 dimensi memudahkan
import numpy as np #np untuk meng aliaskan nama

arr = np.array([1,2,3,4,5])

print(arr)

arr2d = np.array(([1,2,3,4],[5,6,7,8])) #arr 2 dimensi

print(arr2d)

print(arr2d[1,0]) #cara akses dalam array angka 5

print(arr2d[0:,2]) #cara ambil dua angka langsung 3 & 7 dengan syarat posisi angka atau index nya sama

#buat array dengan intial semuanya 0
arr = np.zeros((2,2), dtype=int) #(2,2) untuk di dimensi yang artinya 2x2 , kalau mau initial semua jadi satu np.ones
print(arr)
