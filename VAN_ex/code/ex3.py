import random

import cv2 as cv
import numpy as np
import plotly.graph_objs as go
from cv2 import SOLVEPNP_P3P
from matplotlib import pyplot as plt

import utils


def project_3d_pt_to_pixel(k, extrinsic_camera_mat, pt_3d):
    pt_3d_h = np.append(pt_3d, [1])
    projected = k @ extrinsic_camera_mat @ pt_3d_h
    return projected[0:2] / projected[2]


def ex3_run():
    # 3.1
    img0_left, img0_right = utils.read_images(0)
    img1_left, img1_right = utils.read_images(1)
    kp0_left, des0_left, kp0_right, des0_right = utils.detect_and_compute(img0_left, img0_right)
    kp1_left, des1_left, kp1_right, des1_right = utils.detect_and_compute(img1_left, img1_right)
    matches0 = utils.find_matches(des0_left, des0_right)
    matches1 = utils.find_matches(des1_left, des1_right)
    kp0_left, des0_left, kp0_right, des0_right, dict_matches0 = utils.get_correlated_kps_and_des_from_matches(kp0_left,
                                                                                                              des0_left,
                                                                                                              kp0_right,
                                                                                                              des0_right,
                                                                                                              matches0)
    kp1_left, des1_left, kp1_right, des1_right, dict_matches1 = utils.get_correlated_kps_and_des_from_matches(kp1_left,
                                                                                                              des1_left,
                                                                                                              kp1_right,
                                                                                                              des1_right,
                                                                                                              matches1)
    kp0_left, des0_left, kp0_right, des0_right = utils.rectified_stereo_pattern_test(kp0_left, des0_left, kp0_right,
                                                                                     des0_right)
    kp1_left, des1_left, kp1_right, des1_right = utils.rectified_stereo_pattern_test(kp1_left, des1_left, kp1_right,
                                                                                     des1_right)
    points_cloud_0 = utils.triangulation_list_of_points(kp0_left, kp0_right)
    points_cloud_1 = utils.triangulation_list_of_points(kp1_left, kp1_right)

    # 3.2
    matches_0_1_left = utils.find_matches(np.array(des0_left), np.array(des1_left))

    # 3.3
    random.seed(4)
    kp0_left_l, des0_left_l, kp1_left_l, des1_left_l, dict_matches0_1_left = utils.get_correlated_kps_and_des_from_matches(
        kp0_left, des0_left, kp1_left, des1_left, matches_0_1_left)
    rand_idxs = random.sample(range(len(kp1_left_l) - 1), 4)
    # print(rand_idxs)
    points_3d = np.zeros((4, 3))
    points_2d = np.zeros((4, 2))
    for i in range(4):
        points_3d[i] = points_cloud_0[kp0_left_l[rand_idxs[i]]]
        points_2d[i] = kp1_left_l[rand_idxs[i]].pt
    k, extrinsic_camera_mat_left0, extrinsic_camera_mat_right0 = utils.read_cameras()
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv.solvePnP(points_3d, points_2d, k, dist_coeffs,
                                                               flags=cv.SOLVEPNP_P3P)
    r_mat, _ = cv.Rodrigues(rotation_vector)
    extrinsic_camera_mat_left1 = np.hstack((r_mat, translation_vector))
    extrinsic_camera_mat_right1 = np.array(extrinsic_camera_mat_left1, copy=True)
    extrinsic_camera_mat_right1[0][3] += extrinsic_camera_mat_right0[0][3]
    print("extrinsic_camera_mat_left0:\n", extrinsic_camera_mat_left0)
    print("extrinsic_camera_mat_right0:\n", extrinsic_camera_mat_right0)
    print("extrinsic_camera_mat_left1:\n", extrinsic_camera_mat_left1)
    print("extrinsic_camera_mat_right1:\n", extrinsic_camera_mat_right1)

    # another way to find extrinsic_camera_mat_right1
    # todo

    left0_pos = [0, 0, 0]
    right0_pos = -extrinsic_camera_mat_right0[:, 3]
    left1_pos = ((-np.linalg.inv(r_mat) @ translation_vector).reshape(1, 3))[0]
    right1_pos = -np.linalg.inv(extrinsic_camera_mat_right1[:, 0:3]) @ (extrinsic_camera_mat_right1[:, 3])
    print("left0 camera position: ", left0_pos)
    print("right0 camera position: ", right0_pos)
    print("left1 camera position: ", left1_pos)
    print("right1 camera position: ", right1_pos)

    # plot relative position of the four cameras

    trace = go.Scatter3d(x=[left0_pos[0]], y=[left0_pos[1]], z=[left0_pos[2]], mode='markers', marker=dict(size=7),
                         name='left0_pos')
    fig = go.Figure(data=[trace])
    fig.add_trace(go.Scatter3d(x=[right0_pos[0]], y=[right0_pos[1]], z=[right0_pos[2]], mode='markers',
                               marker=dict(size=7, color='red'), name='right0_pos'))
    fig.add_trace(go.Scatter3d(x=[left1_pos[0]], y=[left1_pos[1]], z=[left1_pos[2]], mode='markers',
                               marker=dict(size=7, color='orange'), name='left1_pos'))
    fig.add_trace(go.Scatter3d(x=[right1_pos[0]], y=[right1_pos[1]], z=[right1_pos[2]], mode='markers',
                               marker=dict(size=7, color='purple'), name='right1_pos'))
    fig.write_html('cameras_pos.html')

    # 3.4
    # we used the points_cloud_0 because there, the 3d points are aligned to the global coordinate
    supporter_left0, supporter_left1 = [], []
    unsupporter_left0, unsupporter_left1 = [], []
    for i in range(len(kp1_left_l)):
        pt_3d = points_cloud_0[kp0_left_l[i]]
        p_left0 = project_3d_pt_to_pixel(k, extrinsic_camera_mat_left0, pt_3d)
        real_p_left0 = kp0_left_l[i].pt
        p_right0 = project_3d_pt_to_pixel(k, extrinsic_camera_mat_right0, pt_3d)
        real_p_right0 = dict_matches0[kp0_left_l[i]].pt
        p_left1 = project_3d_pt_to_pixel(k, extrinsic_camera_mat_left1, pt_3d)
        real_p_left1 = kp1_left_l[i].pt
        p_right1 = project_3d_pt_to_pixel(k, extrinsic_camera_mat_right1, pt_3d)
        real_p_right1 = dict_matches1[kp1_left_l[i]].pt
        if (
                np.linalg.norm(real_p_left0 - p_left0) <= 2 and np.linalg.norm(real_p_right0 - p_right0) <= 2 and
                np.linalg.norm(real_p_left1 - p_left1) <= 2 and np.linalg.norm(real_p_right1 - p_right1) <= 2
                ):
            supporter_left0.append(kp0_left_l[i])
            supporter_left1.append(kp1_left_l[i])
        else:
            unsupporter_left0.append(kp0_left_l[i])
            unsupporter_left1.append(kp1_left_l[i])

    img0_left = cv.drawKeypoints(img0_left, supporter_left0, img0_left, color=(46, 139, 87))
    img0_left = cv.drawKeypoints(img0_left, unsupporter_left0, img0_left, color=(0, 0, 128))
    cv.imwrite(f'img0_left_supporters.png', img0_left)
    plt.imshow(img0_left), plt.show()

    img1_left = cv.drawKeypoints(img1_left, supporter_left1, img1_left, color=(46, 139, 87))
    img1_left = cv.drawKeypoints(img1_left, unsupporter_left1, img1_left, color=(0, 0, 128))
    cv.imwrite(f'img1_left_supporters.png', img1_left)
    plt.imshow(img1_left), plt.show()



ex3_run()
