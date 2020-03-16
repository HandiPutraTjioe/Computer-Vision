import cv2
import os
import numpy as np
from scipy.cluster.vq import * # gk bisa terima string sebagai class, nerima cuma ID
from sklearn.preprocessing import StandardScaler # gk bisa terima string sebagai class, nerima cuma ID
from sklearn.svm import LinearSVC

# program dibagi menjadi 2, yaitu : train dan test
# train
train_path = '08/train'
train_subfolder_path = os.listdir(train_path) # [aeroplane, bicycle, car]
train_img_list = []
img_class_list = []
# 0 -> aeroplane
# 1 -> bicycle
# 2 -> car
# enumerate -> foreach

for idx, subfolder_path in enumerate(train_subfolder_path):
    print(subfolder_path)
    full_path = train_path + '/' + subfolder_path
    img_path_list = os.listdir(full_path) # [1.jpg, 2.jpg, etc]
    for img_path in img_path_list:
        full_img_path = full_path + '/' +img_path
        train_img_list.append(full_img_path)
        img_class_list.append(idx)

# print(train_img_list)
# print(img_class_list)

surf = cv2.xfeatures2d.SURF_create() # descriptor
descriptorlist = []
for img_path in train_img_list:
    _, des = surf.detectAndCompute(cv2.imread(img_path), None) # return list of keypoint and list of descriptor
    descriptorlist.append(des)
# [elemen 1, elemen 2, ....]
stackdescriptor = descriptorlist[0]
for descriptor in descriptorlist[1:]: # mulai dari index pertama, karena idx 0 udh digunakan, ambil dari belakang sampai 1
    stackdescriptor = np.vstack((stackdescriptor, descriptor))
stackdescriptor = np.float32(stackdescriptor)

centroids, _ = kmeans(stackdescriptor, 100, 1) # mengelmpokkan descriptor yang ada jadi 100 kelompok
train_feature = np.zeros((len(train_img_list), len(centroids)), 'float32')
for i in range(len(train_img_list)):
    # untuk ngasih tau descriptor ke-i ada dikelompok yg mana
    words, _ = vq(descriptorlist[i], centroids) # words -> 1 img punya bnyk descriptor, list isinya itu cuma angka 
    # [4, 8, 1, ...]
    # words[0] = 4 -> descriptor ke-0 ada di klompok 4
    # words[1] = 8 -> descriptor ke-1 ada di klompok 8
    # img_feature[0][2] = 5
    # img ke 0 dengan descriptor ke 2 berjumlah 5
    for w in words:
        train_feature[i][w] += 1

std = StandardScaler().fit(train_feature) # fit -> ukuran 
normalized_train_feature = std.transform(train_feature)

svc = LinearSVC()
svc.fit(normalized_train_feature, np.array(img_class_list))

# cth : img_class_list[4] = 0 -> image dengan index 4 adalah aeroplane
# cth : img_class_list[6] = 2 -> image dengan index 6 adalah car

# test
test_path = '08/test'
test_img_path = os.listdir(test_path)

test_img_list = []
for img_path in test_img_path:
    ful_img_path = test_path + '/' + img_path
    test_img_list.append(ful_img_path)

descriptorlist = []
for img_path in test_img_list:
    _, des = surf.detectAndCompute(cv2.imread(img_path), None) # return list of keypoint and list of descriptor
    descriptorlist.append(des)
# [elemen 1, elemen 2, ....]
stackdescriptor = descriptorlist[0]
for descriptor in descriptorlist[1:]: # mulai dari index pertama, karena idx 0 udh digunakan, ambil dari belakang sampai 1
    stackdescriptor = np.vstack((stackdescriptor, descriptor))
stackdescriptor = np.float32(stackdescriptor)

centroids, _ = kmeans(stackdescriptor, 100, 1) # mengelmpokkan descriptor yang ada jadi 100 kelompok
test_feature = np.zeros((len(test_img_list), len(centroids)), 'float32')
for i in range(len(test_img_list)):
    # untuk ngasih tau descriptor ke-i ada dikelompok yg mana
    words, _ = vq(descriptorlist[i], centroids) # words -> 1 img punya bnyk descriptor, list isinya itu cuma angka 
    # [4, 8, 1, ...]
    # words[0] = 4 -> descriptor ke-0 ada di klompok 4
    # words[1] = 8 -> descriptor ke-1 ada di klompok 8
    # img_feature[0][2] = 5
    # img ke 0 dengan descriptor ke 2 berjumlah 5
    for w in words:
        test_feature[i][w] += 1

std = StandardScaler().fit(test_feature) # fit -> ukuran 
normalized_test_feature = std.transform(test_feature)

for img_path, result in zip(test_img_list, svc.predict(normalized_test_feature)):
    print(img_path, train_subfolder_path[result]) # hasilnya kenapa beda, karena kmeans dia random dulu baru dijalanin algoritmanya