import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import plotly.graph_objs as go
import utils

OUTPUT_DIR = 'results\\ex2\\'
THRESHOLD = 1
ORANGE_COLOR = (255, 140, 0)
CYAN_COLOR = (0, 255, 255)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_correlated_kps_from_matches(kp1, kp2, matches):
    kp1_ret, kp2_ret = [], []
    for i, match in enumerate(matches):
        kp1_ret.append(kp1[match.queryIdx])
        kp2_ret.append(kp2[match.trainIdx])
    return kp1_ret, kp2_ret


def deviation_from_pattern(kp1, kp2):
    deviation = np.zeros(len(kp1))
    for i in range(len(kp1)):
        deviation[i] = abs(kp1[i].pt[1] - kp2[i].pt[1])
    return deviation


def create_deviation_hist(deviation):
    plt.hist(deviation, 25)
    plt.xlabel("Deviation from rectified stereo pattern")
    plt.ylabel("Number of Matches")
    plt.savefig(OUTPUT_DIR + 'hist.png')
    plt.show()


def deviation_single_match(single_kp1, single_kp2):
    return abs(single_kp1.pt[1] - single_kp2.pt[1])


def rectified_stereo_pattern_test(kp1, kp2, threshold):
    img1_inliers, img1_outliers, img2_inliers, img2_outliers = [], [], [], []
    for i in range(len(kp1)):
        if deviation_single_match(kp1[i], kp2[i]) < threshold:
            img1_inliers.append(kp1[i])
            img2_inliers.append(kp2[i])
        else:
            img1_outliers.append(kp1[i])
            img2_outliers.append(kp2[i])
    return img1_inliers, img1_outliers, img2_inliers, img2_outliers


def show_accepted_and_rejected_points_on_img(img, inliers_pts, outliers_pts, img_num):
    img = cv.drawKeypoints(img, inliers_pts, img, color=ORANGE_COLOR)
    img = cv.drawKeypoints(img, outliers_pts, img, color=CYAN_COLOR)
    cv.imwrite(f'{OUTPUT_DIR}rectified_stereo_pattern_img{img_num}.png', img)
    plt.imshow(img), plt.show()


def triangulation_single_match(p_cam_mat, q_cam_mat, p, q):  # p_cam_mat -> left camera, img1
    p_x, p_y = p[0], p[1]
    q_x, q_y = q[0], q[1]
    a_mat = np.array([p_x * p_cam_mat[2] - p_cam_mat[0],
                      p_y * p_cam_mat[2] - p_cam_mat[1],
                      q_x * q_cam_mat[2] - q_cam_mat[0],
                      q_y * q_cam_mat[2] - q_cam_mat[1]])
    U, D, V_t = np.linalg.svd(a_mat)
    return V_t[3]


def triangulation(kp1, kp2, triangulation_func):
    k, m1, m2 = utils.read_cameras()
    dots_3d = []
    for i in range(len(kp1)):
        p, q = kp1[i].pt, kp2[i].pt
        ret_pt_4d = triangulation_func(k @ m1, k @ m2, p, q)
        ret_pt_3d = ret_pt_4d[:3] / ret_pt_4d[3]
        dots_3d.append(ret_pt_3d)
    dots_3d = np.array(dots_3d)
    shape_dots_3d = dots_3d.shape
    dots_3d = dots_3d.reshape(shape_dots_3d[:2])
    return dots_3d


def show_dots_3d_cloud(dots_3d, output_name, show=False):
    dots_3d = np.array(dots_3d)
    x = dots_3d[:, 0].reshape(-1)
    y = dots_3d[:, 1].reshape(-1)
    z = dots_3d[:, 2].reshape(-1)
    trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3), name='objects')
    fig = go.Figure(data=[trace])
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='red'), name='camera'))
    if show:
        fig.show()
    fig.write_html(output_name)


def median_distance_between_3d_points(lst_pt1, lst_pt2):
    dist = np.linalg.norm(np.array(lst_pt2) - np.array(lst_pt1), axis=1)
    return np.median(dist)


def match_and_triangulate_image(im_idx, triangulation_func, rectified_test=False):
    img1, img2 = utils.read_images(0)
    kp1, des1, kp2, des2 = utils.detect_and_compute(img1, img2)
    matches = utils.find_matches(img1, kp1, des1, img2, kp2, des2)
    kp1, kp2 = get_correlated_kps_from_matches(kp1, kp2, matches)
    if rectified_test:
        kp1, _, kp2, _ = rectified_stereo_pattern_test(kp1, kp2, THRESHOLD)
    dots_3d = triangulation(kp1, kp2, triangulation_func)
    return dots_3d


def ex2_run():
    # 2.1 Create a histogram of the deviations from the pattern
    # of rectified stereo images for all the matches.
    img1, img2 = utils.read_images(0)
    kp1, des1, kp2, des2 = utils.detect_and_compute(img1, img2)
    matches = utils.find_matches(img1, kp1, des1, img2, kp2, des2)
    kp1, kp2 = get_correlated_kps_from_matches(kp1, kp2, matches)

    deviation = deviation_from_pattern(kp1, kp2)

    create_deviation_hist(deviation)

    print("Percentage of matches that deviate by more than two pixels: ",
          '%.2f' % (np.count_nonzero(deviation > 2) / len(matches) * 100))

    # 2.2 Present all the resulting matches as dots on the image pair, in different colors for accepted matches
    # (inliers) and rejected matches (outliers).
    img1_inliers, img1_outliers, img2_inliers, img2_outliers = rectified_stereo_pattern_test(kp1, kp2, THRESHOLD)
    show_accepted_and_rejected_points_on_img(img1, img1_inliers, img1_outliers, 1)
    show_accepted_and_rejected_points_on_img(img2, img2_inliers, img2_outliers, 2)

    print("Number of matches were discarded: ", len(img1_outliers))

    # 2.3 linear least squares triangulation
    our_dots_3d = triangulation(kp1, kp2, triangulation_single_match)

    # present a 3d plot of the calculated 3d points
    show_dots_3d_cloud(our_dots_3d, f"{OUTPUT_DIR}our_3D_points_cloud_img0.html")

    # repeat the triangulation using ‘cv2.triangulatePoints’
    cv_dots_3d = triangulation(kp1, kp2, cv.triangulatePoints)
    show_dots_3d_cloud(our_dots_3d, f"{OUTPUT_DIR}cv_3D_points_cloud_img0.html")

    # print the median distance between the corresponding 3d points in our implementation,
    # compare to opencv implementation
    median = median_distance_between_3d_points(cv_dots_3d, our_dots_3d)
    print("Median distance between corresponding 3d points in our implementation,"
          "compare to opencv implementation: ", median)

    # 2.4 Run this process (matching and triangulation) over a few pairs of images
    for i in [500, 1000, 1500, 2000, 2500]:
        dots_3d = match_and_triangulate_image(i, triangulation_single_match)
        show_dots_3d_cloud(dots_3d, f"{OUTPUT_DIR}our_3D_points_cloud_img{i}.html")


ex2_run()
