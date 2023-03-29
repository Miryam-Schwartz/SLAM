import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import plotly.express as px
import plotly.graph_objs as go

import utils

THRESHOLD = 1

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


def triangulation(kp1, kp2, triangulation_func):
    k, m1, m2 = utils.read_cameras()
    dots_3d = []
    for i in range(len(kp1)):
        p = kp1[i].pt
        q = kp2[i].pt
        ret_pt_4d = triangulation_func(k@m1, k@m2, p, q)
        ret_pt_3d = ret_pt_4d[:3] / ret_pt_4d[3]
        dots_3d.append(ret_pt_3d)
    shape_cv = np.array(dots_3d).shape
    dots_3d = np.array(dots_3d).reshape(shape_cv[:2])
    return dots_3d

def show_dots_3d_cloud(dots_3d, output_name):
    dots_3d = np.array(dots_3d)
    x = dots_3d[:, 0].reshape(-1)
    y = dots_3d[:, 1].reshape(-1)
    z = dots_3d[:, 2].reshape(-1)
    trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3), name='objects')
    fig = go.Figure(data=[trace])
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='red'), name='camera'))
    fig.show()
    fig.write_html(output_name)


def triangulation_single_match(p_cam_mat, q_cam_mat, p, q):   # p_cam_mat -> left camera, img1
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
    return V_t[3]

def median_distance_between_3d_points(lst_pt1, lst_pt2):
    dist = np.linalg.norm(np.array(lst_pt2)-np.array(lst_pt1), axis=1)
    return np.median(dist)

def get_correlated_kps_from_matched(kp1, kp2, matches):
    kp1_ret, kp2_ret = [], []
    for i, match in enumerate(matches):
        kp1_ret.append(kp1[match.queryIdx])
        kp2_ret.append(kp2[match.trainIdx])
    return kp1_ret, kp2_ret

def match_and_triangulate_image(idx, triangulation_func, rectified_test=False):
    img1, img2 = utils.read_images(0)
    kp1, des1, kp2, des2 = utils.detect_and_compute(img1, img2)
    matches = utils.find_matches(img1, kp1, des1, img2, kp2, des2)
    if rectified_test:
        kp1_matched, _, kp2_matched, _ = rectified_stereo_pattern_test(kp1, kp2, matches, THRESHOLD)
    else:
        kp1_matched, kp2_matched = get_correlated_kps_from_matched(kp1, kp2, matches)
    dots_3d = triangulation(kp1_matched, kp2_matched, triangulation_func)
    show_dots_3d_cloud(dots_3d, f"3D_points_cloud_{idx}.html")

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
    img1_inliers, img1_outliers, img2_inliers, img2_outliers = rectified_stereo_pattern_test(kp1, kp2, matches, THRESHOLD)
    img1 = cv.drawKeypoints(img1, img1_inliers, img1, color=(255, 140, 0))
    img1 = cv.drawKeypoints(img1, img1_outliers, img1, color=(0, 255, 255))
    cv.imwrite('rectified_stereo_pattern_img1.png', img1)
    plt.imshow(img1), plt.show()

    img2 = cv.drawKeypoints(img2, img2_inliers, img2, color=(255, 140, 0))
    img2 = cv.drawKeypoints(img2, img2_outliers, img2, color=(0, 255, 255))
    cv.imwrite('rectified_stereo_pattern_img2.png', img2)
    plt.imshow(img2), plt.show()

    #2.3
    #kp1_matched, kp2_matched = get_correlated_kps_from_matched(kp1, kp2, matches)
    #dots_3d_us = triangulation(kp1_matched, kp2_matched, triangulation_single_match)
    #dots_3d_cv = triangulation(kp1_matched, kp2_matched, cv.triangulatePoints)
    #print(len(dots_3d_us), len(dots_3d_cv))
    #show_dots_3d_cloud(dots_3d_us, "3D_points_cloud_us.html")
    #show_dots_3d_cloud(dots_3d_cv, "3D_points_cloud_cv.html")
    #median = median_distance_between_3d_points(dots_3d_cv, dots_3d_us)
    #print(median)
    match_and_triangulate_image(0, triangulation_single_match)
    #2.4



if __name__ == '__main__':
    ex2_run()
