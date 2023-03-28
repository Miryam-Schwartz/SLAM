import numpy as np
from matplotlib import pyplot as plt
import  cv2 as cv

import utils

def deviation_single_match(match, kp1, kp2):
    return abs(kp1[match.queryIdx].pt[1] - kp2[match.trainIdx].pt[1])


def rectified_stereo_pattern_test(kp1, kp2, matches, threshold):
    img1_inliers, img1_outliers, img2_inliers, img2_outliers = [], [], [], []
    for i, match in enumerate(matches):
        if deviation_single_match(match, kp1, kp2) < threshold:
            img1_inliers.append(kp1[match.queryIdx])
            img2_inliers.append(kp2[match.trainIdx])
        else:
            img1_outliers.append(kp1[match.queryIdx])
            img2_outliers.append(kp2[match.trainIdx])
    return img1_inliers, img1_outliers, img2_inliers, img2_outliers


def triangulation(p, q, p_cam_mat, q_cam_mat):   # p_cam_mat -> left camera, img1
    p_x = p[0]
    p_y = p[1]
    q_x = q[0]
    q_y = q[1]
    a_mat = np.zeros((4, 4))
    a_mat[0] = p_x * p_cam_mat[2] - p_cam_mat[0]
    a_mat[1] = p_y * p_cam_mat[2] - p_cam_mat[1]
    a_mat[2] = q_x * q_cam_mat[2] - q_cam_mat[0]
    a_mat[3] = q_y * q_cam_mat[2] - q_cam_mat[1]
    U, D, V_t = np.linalg.svd(a_mat)
    return V_t[3][:3]



def ex2_run():

    # 2.1
    img1, img2 = utils.read_images(0)
    kp1, des1, kp2, des2 = utils.detect_and_compute(img1, img2)
    matches = utils.find_matches(img1, kp1, des1, img2, kp2, des2)

    deviation = np.zeros(len(matches))
    for i, match in enumerate(matches):
        deviation[i] = abs(kp1[matches[i].queryIdx].pt[1] - kp2[matches[i].trainIdx].pt[1])
    print(deviation)
    plt.hist(deviation, 25)
    plt.xlabel("Deviation from rectified stereo pattern")
    plt.ylabel("Number of Matches")
    plt.show()

    print("Percentage of matches that deviate by more than two pixels: ", np.count_nonzero(deviation > 2) / len(matches) * 100)

    # 2.2
    img1_inliers, img1_outliers, img2_inliers, img2_outliers = rectified_stereo_pattern_test(kp1, kp2, matches, 1)
    img1 = cv.drawKeypoints(img1, img1_inliers, img1, color=(255, 140, 0))
    img1 = cv.drawKeypoints(img1, img1_outliers, img1, color=(0, 255, 255))
    cv.imwrite('rectified_stereo_pattern_img1.png', img1)
    plt.imshow(img1), plt.show()

    img2 = cv.drawKeypoints(img2, img2_inliers, img2, color=(255, 140, 0))
    img2 = cv.drawKeypoints(img2, img2_outliers, img2, color=(0, 255, 255))
    cv.imwrite('rectified_stereo_pattern_img2.png', img2)
    plt.imshow(img2), plt.show()

    p = kp1[matches[0].queryIdx].pt
    q = kp2[matches[0].trainIdx].pt

    k, m1, m2 = utils.read_cameras()
    ret = triangulation(p, q, m1, m2)
    print(ret)


if __name__ == '__main__':
    ex2_run()
