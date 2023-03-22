import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("C:\\Users\\Miryam\\SLAM")

import cv2 as cv

# / cv2.AKAZE_create / cv2.SIFT_create , and many more…
# detectAndCompute
DATA_PATH = r'C:\\Users\\Miryam\\SLAM\\VAN_ex\\dataset\\sequences\\05\\'


def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv.imread(DATA_PATH + 'image_0\\' + img_name, 0)
    img2 = cv.imread(DATA_PATH + 'image_1\\' + img_name, 0)
    return img1, img2


def detect_and_compute(img):
    sift = cv.SIFT_create()
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return sift.detectAndCompute(img, None)


def present_kp(img, kp, img_num):
    img = cv.drawKeypoints(img, kp, img)
    cv.imwrite('siftKeyPoints{}.png'.format(img_num), img)
    plt.imshow(img), plt.show()


def main():
    # 1.1 detect and extract key points
    img1, img2 = read_images(0)
    kp1, des1 = detect_and_compute(img1)
    kp2, des2 = detect_and_compute(img2)

    # present key point
    present_kp(img1, kp1, 1)
    present_kp(img2, kp2, 2)


    # 1.2 print two first descriptors
    print("Descriptors of two first features: ")
    print("img1: ", des1[0], '\n', des1[1])
    print("img2: ", des2[0], '\n', des1[2])

    # 1.3
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)

    img3 = cv.drawMatches(img1, kp1, img2, kp2, np.random.choice(matches, 20), img2, flags=2)
    cv.imwrite('matches.png', img3)
    plt.imshow(img3), plt.show()

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    best_matches, fail_matches = [], []
    ratio = 0.75
    for m1, m2 in matches:
        if m1.distance < ratio * m2.distance:
            best_matches.append([m1])
        else:
            fail_matches.append([m1])

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, np.asarray(best_matches)[np.random.randint(0, len(best_matches), 20), :], img2, flags=2)
    cv.imwrite('best_matches.png', img3)
    plt.imshow(img3), plt.show()

main()
