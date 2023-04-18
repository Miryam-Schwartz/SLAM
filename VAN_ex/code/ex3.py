import random

import cv2 as cv
import numpy as np
from cv2 import SOLVEPNP_P3P

import utils


def ex3_run():
    # 3.1
    img0_left, img0_right = utils.read_images(0)
    img1_left, img1_right = utils.read_images(1)
    kp0_left, des0_left, kp0_right, des0_right = utils.detect_and_compute(img0_left, img0_right)
    kp1_left, des1_left, kp1_right, des1_right = utils.detect_and_compute(img1_left, img1_right)
    matches0 = utils.find_matches(des0_left, des0_right)
    matches1 = utils.find_matches(des1_left, des1_right)
    kp0_left, des0_left, kp0_right, des0_right = utils.get_correlated_kps_and_des_from_matches(kp0_left, des0_left, kp0_right, des0_right, matches0)
    kp1_left, des1_left, kp1_right, des1_right = utils.get_correlated_kps_and_des_from_matches(kp1_left, des1_left, kp1_right, des1_right, matches1)
    kp0_left, des0_left, kp0_right, des0_right = utils.rectified_stereo_pattern_test(kp0_left, des0_left, kp0_right, des0_right)
    kp1_left, des1_left, kp1_right, des1_right = utils.rectified_stereo_pattern_test(kp1_left, des1_left, kp1_right,
                                                                                     des1_right)
    points_cloud_0 = utils.triangulation_list_of_points(kp0_left, kp0_right)
    points_cloud_1 = utils.triangulation_list_of_points(kp1_left, kp1_right)

    # 3.2
    matches_0_1_left = utils.find_matches(np.array(des0_left), np.array(des1_left))

    # 3.3
    kp0_left_l, des0_left_l, kp1_left_l, des1_left_l = utils.get_correlated_kps_and_des_from_matches(kp0_left, des0_left, kp1_left, des1_left, matches_0_1_left)
    rand_idxs = random.sample(range(len(kp1_left_l) - 1), 4)
    print(rand_idxs)
    points_3d = np.zeros((4, 3))
    points_2d = np.zeros((4, 2))
    for i in range(4):
        points_3d[i] = points_cloud_0[kp0_left_l[rand_idxs[i]]]
        points_2d[i] = kp1_left_l[rand_idxs[i]].pt
    k, _, _ = utils.read_cameras()
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv.solvePnP(points_3d, points_2d, k, dist_coeffs, flags=cv.SOLVEPNP_P3P)
    r_mat, _ = cv.Rodrigues(rotation_vector)
    extrinsic_camera_mat = np.hstack((r_mat, translation_vector))
    print(extrinsic_camera_mat)

ex3_run()