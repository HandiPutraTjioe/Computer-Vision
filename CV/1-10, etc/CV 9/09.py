import cv2
import os
from scipy.spatial.distance import euclidean

# train
img_dir = '09'
train_img_features = []

for img_file in os.listdir(img_dir):
    full_path_img = img_dir + "/" + img_file
    img = cv2.imread(full_path_img)

    hist = cv2.calcHist([img], [2, 1, 0], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) # for channel -> B = 2, G = 1, R = 0

    normalize_hist = cv2.normalize(hist, hist.shape)
    normalize_hist = normalize_hist.flatten()

    train_img_features.append((img_file, normalize_hist))

# testing
test_img = cv2.imread('Mordor-002_True.png')
test_hist = cv2.calcHist([test_img], [2, 1, 0], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
normalize_test_hist = cv2.normalize(test_hist, test_hist.shape)
normalize_test_hist = normalize_test_hist.flatten()

result = []
for name, hist in train_img_features:
    distance = euclidean(hist, normalize_test_hist)
    result.append((name, distance))

for name, similarity in result:
    print(name + " " + str(similarity))