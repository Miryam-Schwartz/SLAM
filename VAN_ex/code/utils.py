import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

DATA_PATH = r'C:\\Users\\Miryam\\SLAM\\VAN_ex\\dataset\\sequences\\05\\'


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
    cv.imwrite('matches.png', img_matches)
    plt.imshow(img_matches), plt.show()
    return matches


def read_cameras():
    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)
        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)
        k = m1[:, :3]
        m1 = np.linalg.inv(k) @ m1
        m2 = np.linalg.inv(k) @ m2
    return k, m1, m2
