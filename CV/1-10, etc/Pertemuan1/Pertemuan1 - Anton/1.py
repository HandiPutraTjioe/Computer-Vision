# '''print di python'''
# print("Hello World")

# '''scan di python'''
# name = input("nama : ")
# age=int(input("umur : "))
# print(type(age))
# print(name)

# '''array in python'''
# '''list'''
# movies = ['Harry Potter', 'Ant Man', 'Aqua Man']
# print(type(movies))
# print(movies)

# '''add data to list'''
# movies.append('Venom')
# movies += ['lengedaris', 'Anton']
# print(movies)

# '''update data in list'''
# print(movies[3])
# movies[3]='Alita'
# print(movies[3])

# '''delete data in list'''
# movies.remove('Harry Potter')
# print(movies)

# '''Tuple'''
# genre=('Action', 'Adventure', 'Thriller')
# print(type(genre))
# print(genre)

# '''add data in tuple'''
# genre+=('Horror', 'Drama')
# print(genre)

# print(genre[1])

# '''delete data in tuple'''
# genre=genre[0:2]+genre[3:5]
# print(genre)

# '''update data in tuple'''
# '''delete + insert karena python tidak bisa diupdate (isinya constant)'''

# '''Dictonary'''
# movie = {
#     'name': 'Alita', 
#     'genre': 'Sci-Fi', 
#     'rating': 4.23
# }
# print(movie['name'])

# '''add data in dictonary'''
# movie['duration']=120
# print(movie)

# '''update data in dictonary'''
# # movie['rating']=4.45
# # movie['duration']=240
# '''best practice update data'''
# movie.update({
#     'rating': 4.45,
#     'duration': 240
# })
# print(movie)

# '''delete data in dictonary'''
# '''hapus satu'''
# movie.pop('duration')
# print(movie)

# '''selection'''
# value=10
# if value % 2 == 0:
#     print('Genap')
# else:
#     print('Ganjil')

# '''looping'''
# ''' for and while'''
# '''for(int i=1; i<11; i++)'''
# for i in range(1, 11):
#     print(i)

# '''for(int i=1; i<11; i+=2)'''
# for i in range(1, 11, 2):
#     print(i)

# '''for(int i=1; i<11; i-=2)'''
# for i in range(11, 1, -2):
#     print(i)

# '''foreach'''
# names=['Alita', 'Angel', 'Bob']
# for name in names:
#     print(name)

# '''while'''
# '''input nama, validasi 5-25 char'''
# name=''
# while len(name)<5 or len(name)>25:
#     name=input("Enter your name : ")

# '''function'''
# def pow(a, b):
#     return a ** b

# firstNumber = int(input('Input first number : '))
# secondNumber = int(input('Input second number : '))
# print(pow(firstNumber, secondNumber))

# '''intro to numpy'''
import numpy as np #library

# '''cara bikin array di numpy'''
arr=np.array([1, 2, 3])
print(arr)

arr=np.arange(1, 9)
print(arr)

print(arr.shape) #ukuran array nya

# '''reshape arr menjadi matrix'''
arr=arr.reshape(4, 2)
print(arr)

# '''tumpukan matrix jadi suatu matrix yang baru'''
# '''vstack -> bergabung ke bawah'''
# '''hstack -> bergabung ke samping'''
newArr=np.array([9, 10])
newArr=np.vstack((arr, newArr))
print(newArr)

tempArr=np.array([1, 2, 3, 4, 5])
tempArr=tempArr.reshape(5, 1)
newArr=np.hstack((newArr, tempArr))
print(newArr)

# '''load and show image using opencv'''
# import cv2

# '''read image'''
# image=cv2.imread('panda.jpg', -1)
# '''1 -> RGB'''
# '''0 -> black and white'''
# '''-1 -> RGBA'''

# '''show image'''
# cv2.imshow('Panda', image)
# cv2.waitKey(0)