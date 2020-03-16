import cv2

img_box = cv2.imread('06/box.png')
img_scene = cv2.imread('06/box_in_scene.png')

# SURF
# untuk mendetek fitur, fitur -> pembeda suatu image 
# (titik penting (key point))
# descriptor

# cth :
# keypoint[2]
# descriptor[2]

surf = cv2.xfeatures2d.SURF_create() 
kp_box, des_box = surf.detectAndCompute(img_box, None)
kp_scene, des_scene = surf.detectAndCompute(img_scene, None)

# bagaimana membandingkannya ?

# FLANN
# parameter 1 -> algo yang dipakai, diterima dalam dictionary()
# KDTREE -> ??
KDTREE_INDEX = 0
flann = cv2.FlannBasedMatcher(dict(algorithm = KDTREE_INDEX))

des_box = des_box.astype('f')
des_scene = des_scene.astype('f')
# kenapa harus di float
# -> cause kalau ngejalanin mtk integer dibuat auto round, makanya diset jadi float supaya akurasinya tinggi

matches = flann.knnMatch(des_box, des_scene, k = 2)

# des_box            des_scene
#  5 element          10 element
# a = [n,m]
# b = [p,q]
# c
# d
# e

valid_matches = []
for i in range(len(matches)):
    valid_matches.append([0, 0])
    # valid_matches[6] = [1, 0] # gambar best match ke 6
    # kalau (0, 0) -> tdk di gambar
    # [1, 0] -> best macth digambar
    # [0, 1] -> second best match digambar
    # [1, 1] -> digambar semuanya

total_match_valid = 0
for idx, (p, q) in enumerate(matches):
    if(p.distance < 0.7 * q.distance):
        # valid
        valid_matches[idx] = [1, 0]
        total_match_valid += 1
    else:
        # g valid
        continue

print(total_match_valid)
img_result = cv2.drawMatchesKnn(
    img_box, kp_box, 
    img_scene, kp_scene,
    matches, None,
    matchesMask = valid_matches
)

cv2.imshow('1', img_result)
cv2.waitKey(0)