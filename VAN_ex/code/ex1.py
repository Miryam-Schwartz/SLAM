import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

sys.path.append("C:\\Users\\Miryam\\SLAM")

DATA_PATH = r'C:\\Users\\Miryam\\SLAM\\VAN_ex\\dataset\\sequences\\05\\'
OUTPUT_DIR = 'results\\ex1\\'
SIGNIFICANCE_TST_RATIO = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_images(idx):
    """
    Get index of picture from the data set and read them
    :param idx: index
    :return: left and right cameras photos
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv.imread(DATA_PATH + 'image_0\\' + img_name, 0)
    img2 = cv.imread(DATA_PATH + 'image_1\\' + img_name, 0)
    return img1, img2


def detect_and_compute(img1, img2):
    """
    Call CV algorithms sift to detect and compute features in right and left pictures
    :param img1:
    :param img2:
    :return: key points and descriptors for right & left images
    """
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2


def show_key_points(img1, kp1, img2, kp2):
    """
    For each image (right & left), show the key points on the picture, and save the image with the kp as new image
    :param img1:
    :param kp1:
    :param img2:
    :param kp2:
    :return: right and left images with the key points
    """
    img1 = cv.drawKeypoints(img1, kp1, img1)
    img2 = cv.drawKeypoints(img2, kp2, img2)
    cv.imwrite(OUTPUT_DIR + 'keyPointsImg1.png', img1)
    cv.imwrite(OUTPUT_DIR + 'keyPointsImg2.png', img2)
    plt.imshow(img1), plt.show()
    plt.imshow(img2), plt.show()
    return img1, img2


def find_matches(img1, kp1, des1, img2, kp2, des2):
    """
    Find, for each descriptor in the left image, the closest feature in the right image.
    Present 20 random matches as lines connecting the key-point pixel location on the images pair. T
    :param img1:
    :param kp1:
    :param des1:
    :param img2:
    :param kp2:
    :param des2:
    :return: new image with matches on it
    """
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, np.random.choice(matches, 20), img2, flags=2)
    cv.imwrite(OUTPUT_DIR + 'matches.png', img_matches)
    plt.imshow(img_matches), plt.show()
    return matches


def significance_test(des1, des2, ratio):
    """
    Test the matches: reject all matches that the ratio between first match and second match is not closegi enough
    :param img1:
    :param kp1:
    :param des1:
    :param img2:
    :param kp2:
    :param des2:
    :param ratio: ratio for the test
    :return: best and fail matches after applying the test
    """
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    best_matches, fail_matches = [], []
    for m1, m2 in matches:
        if m1.distance < ratio * m2.distance:
            best_matches.append([m1])
        else:
            fail_matches.append([m1])
    return best_matches, fail_matches


def draw_matches_signi_test(img1, img2, kp1, kp2, matches, output_name):
    img_with_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2,
                                         np.asarray(matches)[np.random.randint(0, len(matches), 20), :], img2,
                                         flags=2)
    cv.imwrite(output_name, img_with_matches)
    plt.imshow(img_with_matches), plt.show()


def ex1_run():

    # 1.1 detect and extract key points
    img1, img2 = read_images(0)
    kp1, des1, kp2, des2 = detect_and_compute(img1, img2)
    show_key_points(img1, kp1, img2, kp2)

    # 1.2 print two first descriptors
    print("Descriptors of two first features: ")
    print("img1: ", des1[0], '\n', des1[1])
    print("img2: ", des2[0], '\n', des1[2])

    # 1.3 match the two descriptors list
    matches = find_matches(img1, kp1, des1, img2, kp2, des2)

    # 1.4 use significance test to reject matches
    best_matches, fail_matches = significance_test(des1, des2, SIGNIFICANCE_TST_RATIO)
    draw_matches_signi_test(img1, img2, kp1, kp2, best_matches, OUTPUT_DIR + 'best_matches.png')

    # how many matches were discarded?
    print("The number of discarded matches after applying significance test is: ", len(fail_matches))

    # draw fail matches (to find correct match that failed the significance test)
    draw_matches_signi_test(img1, img2, kp1, kp2, fail_matches, OUTPUT_DIR + 'fail_matches.png')


if __name__ == '__main__':
    ex1_run()
