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


def find_matches(des1, des2):
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


def get_correlated_kps_from_matches(kp1, kp2, matches):
    """
    Return matched key-points.
    :param kp1: list of key-points of left image
    :param kp2: list of key-points of right image
    :param matches: list of matches between right and left image
    :return: 2 lists of key-points, for right and left images,
    when, the i-th keypoint in kp1, corresponding to the i-th keypoint in kp2.
    Elements type of kp1 and kp2 is KeyPoint.
    """
    kp1_ret, kp2_ret = [], []
    for i, match in enumerate(matches):
        kp1_ret.append(kp1[match.queryIdx])
        kp2_ret.append(kp2[match.trainIdx])
    return kp1_ret, kp2_ret


def difference_y_axis_between_two_kps(single_kp1, single_kp2):
    """
    Calculate the difference in y-axis between two key-points
    :param single_kp1: single KeyPoint instance
    :param single_kp2: also
    :return: difference in y-axis between two key-points
    """
    return abs(single_kp1.pt[1] - single_kp2.pt[1])


def rectified_stereo_pattern_test(kp1, des1, kp2, des2, threshold=1):
    img1_inliers_kps, img1_inliers_des, img2_inliers_kps, img2_inliers_des = [], [], [], []
    for i in range(len(kp1)):
        if difference_y_axis_between_two_kps(kp1[i], kp2[i]) < threshold:
            img1_inliers_kps.append(kp1[i])
            img1_inliers_des.append(des1[i])
            img2_inliers_kps.append(kp2[i])
            img2_inliers_des.append(des2[i])
    return img1_inliers_kps, img1_inliers_des, img2_inliers_kps, img2_inliers_des


def triangulation_single_match(p_cam_mat, q_cam_mat, p, q):  # p_cam_mat -> left camera, img1
    """
    Triangulate a point - convert from picture coordinates of two matched points, to world coordinates.
    :param p_cam_mat: left camera matrix
    :param q_cam_mat: right camera matrix
    :param p: point on image 1, represented by x and y coordinates of pixels
    :param q: point on image 2, represented by x and y coordinates of pixels
    :return: 3d world coordinates of the point
    """
    p_x, p_y = p[0], p[1]
    q_x, q_y = q[0], q[1]
    a_mat = np.array([p_x * p_cam_mat[2] - p_cam_mat[0],
                      p_y * p_cam_mat[2] - p_cam_mat[1],
                      q_x * q_cam_mat[2] - q_cam_mat[0],
                      q_y * q_cam_mat[2] - q_cam_mat[1]])
    _, _, v_t = np.linalg.svd(a_mat)
    return v_t[3]


def triangulation_list_of_points(kp1, kp2, triangulation_func=triangulation_single_match):
    """
    Triangulation for lists of points from right and left images.
    :param kp1: list of key-points, matched to kp2
    :param kp2:
    :param triangulation_func: function that makes the triangulation.
    Can be OpenCv.triangulatePoints or our implementation - triangulation_single_match
    :return: list of 3d points
    """
    k, m1, m2 = read_cameras()
    dots_3d = dict()
    for i in range(len(kp1)):
        p, q = kp1[i].pt, kp2[i].pt
        ret_pt_4d = triangulation_func(k @ m1, k @ m2, p, q)
        ret_pt_3d = ret_pt_4d[:3] / ret_pt_4d[3]
        dots_3d[kp1[i]] = ret_pt_3d
    return dots_3d


def match_and_triangulate_image(im_idx, triangulation_func=triangulation_single_match, rectified_test=False,
                                threshold=1):
    """
    Read, find key-points, match and triangulate points.
    :param threshold:
    :param im_idx: index pair images
    :param triangulation_func: function that makes the triangulation.
    :param rectified_test: if True, reject points according to rectified_test
    :return: list of 3d points, after the triangulation
    """
    img1, img2 = read_images(im_idx)
    kp1, des1, kp2, des2 = detect_and_compute(img1, img2)
    matches = find_matches(img1, kp1, des1, img2, kp2, des2)
    kp1, kp2 = get_correlated_kps_from_matches(kp1, kp2, matches)
    if rectified_test:
        kp1, _, kp2, _ = rectified_stereo_pattern_test(kp1, kp2, threshold)
    dots_3d = triangulation_list_of_points(kp1, kp2, triangulation_func)
    return kp1, kp2, dots_3d


def get_correlated_kps_and_des_from_matches(kp1, des1, kp2, des2, matches):
    kp1_ret, kp2_ret = [], []
    des1_ret, des2_ret = [], []
    for i, match in enumerate(matches):
        kp1_ret.append(kp1[match.queryIdx])
        kp2_ret.append(kp2[match.trainIdx])
        des1_ret.append(des1[match.queryIdx])
        des2_ret.append(des2[match.trainIdx])
    return kp1_ret, des1_ret, kp2_ret, des2_ret
