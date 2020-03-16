# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import sys

# def menu():
#     print("============================")
#     print("      Computer Vision")
#     print("============================")
#     print("1. Show Image Panda")
#     print("2. Read Data")
#     print("3. Read Data Riil")
#     print("4. Bilangan Faktorial")
#     print("5. List")
#     print("6. BGR 2 GRAY")
#     print("7. BGR 2 RGB")
#     print("8. Instant Gray Scale")
#     print("9. Thresholding")
#     print("10. Filtering")
#     print("11. Exit")
#     print("============================")

# def menulist():
#     print("================")
#     print("      List")
#     print("================")
#     print("1. Append List")
#     print("2. Update List")
#     print("3. Delete list")
#     print("4. Search Index")
#     print("5. Show List")
#     print("6. Exit")
#     print("================")

# def thresholding():
#     print("==================================")
#     print("     Thresholding")
#     print("==================================")
#     print("1. Binary Thresholding")
#     print("2. Binary Inverse Thresholding")
#     print("3. Truncate Thresholding")
#     print("4. To Zero Thresholding")
#     print("5. To Zero Inverse")
#     print("6. Otzu Binary")
#     print("7. Exit")
#     print("==================================")


# def clr():
#     for i in range(1,8):
#         print("\n")

# def clr1():
#     for i in range(1,5):
#         print("\n")

# choice = 0
# while choice != 11:
#     clr()
#     menu()
    
#     try:
#         choice = int(input("Choice [1-11] : "))
#     except:
#         print("Must a Numeric")

#     if (choice == 1):
#         img = cv2.imread("panda.jpg")
#         cv2.imshow("Panda", img)
#         cv2.waitKey()

#     elif (choice == 2):
#         name = ""
#         age = 0

#         while len(name) < 5 or len(name) > 15:
#             try:
#                 name = input("Input Your Name [5-15] : ")
#             except:
#                 print("Name must be between 5 - 15 characters")

#         while age < 1 or age > 100:
#             try:
#                 age = int(input("Input Your Age [1-100] : "))
#             except:
#                 print("Age must be between 1 - 100")

#         print("Your Name is " + name + " and Your Age is" + str(age))

#     elif (choice == 3):
#         num1 = 0
        
#         num1 = float(input("Input Float Number : "))

#         hasil = num1 * 2

#         print("Num1 is %f" % num1)
#         print("%f x 2 = %f" % (num1, hasil))


#     elif (choice == 4):
#         print("Nilai Faktorial Bilangan")

#         n = int(input("\nInput a Number : "))
        
#         if n < 0:
#             print("Number can not negative")
#             sys.exit(1)
        
#         faktorial = 1

#         print("%d != " % n, end='')
#         i = n
#         while i >= 1:
#             if i != 1:
#                 print("%d != " % i, end='')
#             else:
#                 print("%d != " % i, end='')
#             faktorial *= i
#             i -= 1
#         print(faktorial)

#     elif (choice == 5):
#         list1 = ['Avengers End Game','The Battles','Police 2','Chucky 9','Ghost 5','Saw 15']

#         choice1 = 0

#         while choice1 != 6:
#             clr1()
#             menulist()

#             try:
#                 choice1 = int(input("Input Choice [1-6] : "))
#             except:
#                 print("Must a Numeric")

#             if choice1 == 1:
#                 name = ""

#                 while len(name) < 5 or len(name) > 25:
#                     try:
#                         name = input("Input Movie Title [5-25] : ")    
#                     except:
#                         print("Name must be between 5 - 25 characters")
                
#                 list1.append(name)
#                 print("Success Added To The List...")

#             elif choice1 == 2:
#                 print(list1,end="\n")

#                 num1 = int(input("Number of list : "))
                
#                 name = ""
#                 while len(name) < 5 or len(name) > 25:
#                     try:
#                         name = input("Input Movie Title [5-25] : ")    
#                     except:
#                         print("Name must be between 5 - 25 characters")
                
#                 list1[num1] = name
#                 print(list1,end="\n")

#             elif choice1 == 3:
#                 print(list1,end="\n")

#                 num1 = int(input("Number of list : "))
#                 print(list1.pop(num1))

#                 print(list1,end="\n")

#             elif choice1 == 4:
#                 name = ""
#                 while len(name) < 5 or len(name) > 25:
#                     try:
#                         name = input("Input Movie Title [5-25] : ")    
#                     except:
#                         print("Name must be between 5 - 25 characters")

#                 print("Name %s " % name + " in index " +  str(list1.index(name)))

#             elif choice1 == 5:
#                 for idx,i in enumerate(list1):
#                     print(str(idx+1) +". "+ i)

#     elif (choice == 6):
#         img = cv2.imread("lena.jpg")

#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         plt.subplot(111)
#         plt.title("Manual Gray Scale - Image")
#         plt.imshow(gray_img, 'gray')
#         plt.xticks([])
#         plt.yticks([])

#         plt.show()

#     elif (choice == 7):
#         img = cv2.imread("lena.jpg")

#         bgr_to_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         plt.subplot(111)
#         plt.title("BGR to RGB Image")
#         plt.imshow(bgr_to_rgb)
#         plt.xticks([])
#         plt.yticks([])

#         plt.show()

#     elif (choice == 8):
#         img_gray_instant = cv2.imread('lena.jpg', 0)

#         plt.subplot(111)
#         plt.title("Instant Gray Scale")
#         plt.imshow(img_gray_instant, 'gray')
#         plt.xticks([])
#         plt.yticks([])

#         plt.show()
    
#     elif (choice == 9):
#         choice2 = 0

#         while choice2 != 7:
#             clr()
#             thresholding()

#             choice2 = int(input("Input thresholding [1-7] : "))

#             if choice2 == 1:

            

#     elif (choice == 10):
#         print("menu 6")

#     else:
#         print("You must choice between 1 - 11")

