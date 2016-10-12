import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import svm
img1 = cv2.imread('image.png',0)          # queryImage
img2 = cv2.imread('square.jpg',0) # trainImage
img3 = cv2.imread('rectangle.jpg',0) # trainImage
img4 = cv2.imread('triangle1.jpg',0) # trainImage
img5 = cv2.imread('circle.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
sift2 = cv2.xfeatures2d.SIFT_create()
sift3 = cv2.xfeatures2d.SIFT_create()
sift4 = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print(kp1)
print(des1)

print(kp2)
print(des2)
kp3, des3 = sift.detectAndCompute(img1,None)
kp4, des4 = sift.detectAndCompute(img3,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
bf2 = cv2.BFMatcher()
matches2 = bf2.knnMatch(des3, des4, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
for m, n in matches2:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,2)
plt.show()
plt.figure()
plt.imshow(img3),plt.show()
plt.imsave('output.png',img3)