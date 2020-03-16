''' Membaca data '''

# nama = input("Masukkan nama anda : ")
# karakter = input("Masukkan sebuah karakter : ")

# print("Halo, ", nama, " apa kabar?")
# print("Karakter yang dimasukkan : '", karakter, "'")


''' Baca data bilangan riil '''

# bilril = float(input("Masukkan bilangan riil : "))
# hasil = bilril*2

# print("Bilangan yg dimasukkan adalah %f" % bilril)
# print("%f x 2 = %f" % (bilril, hasil))


''' Objek '''

# x = 12
# y = 13
# print(type(x))
# print(id(x))
# print(id(y))

# s = "Hello world"
# print(type(s))

# li = [1,2,3]
# print(type(li))

# lo = (1,2,3)
# print(type(lo))


''' Tipe String '''

# str1 = 'Ini string yg menggunakan petik tunggal'
# str2 = "Ini string yg menggunakan petik ganda"
# str3 = """ Ini adalah string panjang. """
# print(str1,"\n",str2,"\n",str3)

# str1 = 'Petik tunggal \'escape\', petik ganda "ok"'
# str2 = "Petik tunggal 'ok', petik ganda \"escape\""
# str3 = "Baris pertama\nBaris kedua"
# print(str1)
# print(str2)
# print(str3)


''' Membandingkan string '''

# s1 = 'python'
# s2 = "python"

# print("s1: " + s1)
# print("s2: " + s2)

# if s1 == s2:
#     print("s1 sama dengan s2")
# else:
#     print("s1 tidak sama dengan s2")


''' Mengambil substring '''

# s = "Python"
# print(s[0]) # P
# print(s[1]) # y
# print(s[-1]) # n
# print(s[1:]) # ython
# print(s[3:]) # hon
# print(s[:2]) # Py
# print(s[:1]) # P
# print(s[1:4]) # yth
# print(s[2:5]) # tho
# print(s[-1:1]) # 
# print(s[1:-3]) # yt
# print(s[-4:-2]) # th


''' Tipe numerik '''

z = 4.1 + 2j
print(z.real)
print(z.imag)


''' Tipe boolean '''

# t = True
# f = False
# print(t and f)
# print(t or f)
# print(type(t))


''' Tipe tuple '''

# import datetime as dt
# bulan = ("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
# today = dt.date.isoformat(dt.date.today())

# yyyy = today[:4]
# mm = today[5:7]
# dd = today[8:]

# print(today)
# print("%s %s %s" % (dd, bulan[int(mm)-1], yyyy))


''' Tipe set & frozenset '''
# from __future__ import print_function
# import sys

# def printElements(s, info):
#     print(info)
#     if len(s) == 0:
#         print("Himpunan kosong\n")
#         sys.exit(1)
#     for e in s:
#         print(str(e) + " ", end='')
#     print("\n")

# def main():
#     s = set([11,22,33,44,55])
#     printElements(s, "Elemen awal:")
#     s.add(66)
#     s.add(77)
#     printElements(s, "Setelah pemanggilan add():")

#     s.update([88,99])
#     printElements(s, "Setelah pemanggilan update():")

#     s.remove(44)
#     printElements(s, "Setelah pemanggilan remove():")

#     s.clear()
#     printElements(s, "Setelah pemanggilan clear():")

# if __name__ == "__main__":
#     main()


''' Menentukan akar-akar persamaan kuadrat '''
# import sys
# import math

# print("Akar-akar persamaan kuadrat")

# a = int(input("\nMasukkan nilai a:"))
# b = int(input("\nMasukkan nilai b:"))
# c = int(input("\nMasukkan nilai c:"))

# D = (b*b) - (4*a*c)

# if D < 0:
#     print("Akar-akar imajiner")
#     sys.exit(1)
# elif D == 0:
#     x1 = (-b + math.sqrt(D)) / (2 * a)
#     x2 = x1
# else:
#     x1 = (-b + math.sqrt(D)) / (2 * a)    
#     x2 = (-b - math.sqrt(D)) / (2 * a)

# print("\nx1 = %d" %x1)
# print("x2 = %d" %x2)


''' Menentukan nilai indeks ujian '''

# print("Nilai IPK Mahasiswa\n")
# print("-------------------\n\n")

# uts = float(input("Masukkan Nilai UTS mhs : "))
# uas = float(input("Masukkan Nilai UAS mhs : "))

# na = (0.65 * uas) + (0.35 * uts)

# if na >= 80:
#     Indeks = 'A'
# elif na >= 70 and na < 80:
#     Indeks = 'B'
# elif na >= 55 and na < 70:
#     Indeks = 'C'
# elif na >= 40 and na < 55:
#     Indeks = 'D'
# else:
#     Indeks = 'E'

# print("\nNilai Akhir : %0.2f" % na)
# print("Indeks : %c" % Indeks)


''' Menghitung nilai faktorial '''
import sys
print("Nilai faktorial bilangan")

n = int(input("\nMasukkan Bilangan : "))

if n < 0:
    print("\nBilangan tidak boleh negatif...")
    sys.exit(1)

faktorial = 1

print("%d != " % n, end='')
i = n
while i >= 1:
    if i != 1:
        print("%d x " % i, end='')
    else:
        print("%d = " % i, end='')        
    faktorial *= i
    i -= 1

print(faktorial)        