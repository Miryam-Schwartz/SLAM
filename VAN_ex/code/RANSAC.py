import random
import cv2 as cv

import utils
import numpy as np

K, m_left, m_right = utils.read_cameras()
INDENTATION_RIGHT_CAM_MAT = m_right[0][3]


def read_match_and_straighten(frame_id):
    img_left, img_right = utils.read_images(frame_id)
    kp_left, des_left, kp_right, des_right = utils.detect_and_compute(img_left, img_right)
    matches = utils.find_matches(des_left, des_right)
    kp_left, des_left, kp_right, des_right, _ = \
        utils.get_correlated_kps_and_des_from_matches(kp_left, des_left, kp_right, des_right, matches)
    kp_left, des_left, kp_right, des_right = \
        utils.rectified_stereo_pattern_test(kp_left, des_left, kp_right, des_right, 2)
    kp_left = [kp.pt for kp in kp_left]
    kp_right = [kp.pt for kp in kp_right]
    return np.array(kp_left), np.array(des_left), np.array(kp_right), np.array(des_right)


def RANSAC(first_frame_id, second_frame_id):
    kp_left_first, des_left_first, kp_right_first, des_right_first = read_match_and_straighten(first_frame_id)
    kp_left_second, des_left_second, kp_right_second, des_right_second = read_match_and_straighten(second_frame_id)
    matches = utils.find_matches(des_left_first, des_left_second)
    matches = [(match.queryIdx, match.trainIdx) for match in matches]

    p, eps = 0.99, 0.99  # eps = prob to be outlier
    i = 0
    size = len(matches)
    max_supporters_num = 0
    idxs_max_supports_matches = None
    while eps > 0 and i < utils.calc_max_iterations(p, eps, 4) and i < 1000:
        # print("i = ", i)
        points_2d, points_3d = _sample_4_points(kp_left_first, kp_right_first, kp_left_second, matches)
        extrinsic_camera_mat_second_frame_left, extrinsic_camera_mat_second_frame_right = \
            utils.estimate_second_frame_mats_pnp(points_2d, points_3d, INDENTATION_RIGHT_CAM_MAT, K)
        if extrinsic_camera_mat_second_frame_left is None:
            continue
        idxs_of_supporters_matches, _ = _find_supporters(matches, kp_left_first, kp_right_first, kp_left_second,
                                                         kp_right_second,
                                                         extrinsic_camera_mat_second_frame_left,
                                                         extrinsic_camera_mat_second_frame_right)
        supporters_num = len(idxs_of_supporters_matches)
        # print("suporters num ", idxs_of_supporters_matches)
        if supporters_num > max_supporters_num:
            max_supporters_num = supporters_num
            idxs_max_supports_matches = idxs_of_supporters_matches
            # update eps
            eps = (size - max_supporters_num) / size
        i += 1
    # print(f"max supportes num: {max_supporters_num}, inliers percentage: {max_supporters_num / size}")
    if max_supporters_num <= 6:
        return None, None, None, None, None, None, None
    points_3d_supporters = np.empty((max_supporters_num, 3))
    points_2d_supporters = np.empty((max_supporters_num, 2))
    for i in range(max_supporters_num):
        cur_match = matches[idxs_max_supports_matches[i]]
        points_3d_supporters[i] = _get_3d_pt(kp_left_first[cur_match[0]], kp_right_first[cur_match[0]])
        pixel_left = kp_left_second[cur_match[1]]
        points_2d_supporters[i] = np.array(pixel_left)
    extrinsic_camera_mat_second_frame_left, extrinsic_camera_mat_second_frame_right = \
        utils.estimate_second_frame_mats_pnp(points_2d_supporters, points_3d_supporters,
                                             INDENTATION_RIGHT_CAM_MAT, K,
                                             flag=cv.SOLVEPNP_ITERATIVE)

    idxs_max_supports_matches, idxs_unsupporters_matches = \
        _find_supporters(matches, kp_left_first, kp_right_first, kp_left_second, kp_right_second,
                         extrinsic_camera_mat_second_frame_left,
                         extrinsic_camera_mat_second_frame_right)
    if len(idxs_max_supports_matches) == 0:
        return None, None, None, None, None, None, None
    matches = np.array(matches)
    idxs_max_supports_matches = np.array(idxs_max_supports_matches)
    return extrinsic_camera_mat_second_frame_left,\
        matches[idxs_max_supports_matches],\
        matches[idxs_unsupporters_matches], \
        kp_left_first, kp_right_first, kp_left_second, kp_right_second


def _get_3d_pt(left_pixel, right_pixel):
    pt_4d = utils.triangulation_single_match(K @ m_left, K @ m_right, left_pixel, right_pixel)
    return pt_4d[:3] / pt_4d[3]


def _sample_4_points(kp_left_first, kp_right_first, kp_left_second, matches):
    rand_idxs = random.sample(range(len(matches)), 4)
    points_3d = np.empty((4, 3))
    points_2d = np.empty((4, 2))
    for i in range(4):
        cur_match = matches[rand_idxs[i]]  # kp_idx of first frame, kp_idx of second frame
        points_3d[i] = _get_3d_pt(kp_left_first[cur_match[0]], kp_right_first[cur_match[0]])
        points_2d[i] = np.array(kp_left_second[cur_match[1]])
    return points_2d, points_3d


def _find_supporters(matches, kp_left_first, kp_right_first, kp_left_second, kp_right_second,
                     extrinsic_camera_mat_second_frame_left,
                     extrinsic_camera_mat_second_frame_right):
    idxs_supports_matches = []
    idxs_unsupports_matches = []
    for i in range(len(matches)):
        cur_match = matches[i]
        pt_3d = _get_3d_pt(kp_left_first[cur_match[0]], kp_right_first[cur_match[0]])
        pixel_second_left = utils.project_3d_pt_to_pixel(K, extrinsic_camera_mat_second_frame_left, pt_3d)
        pixel_left, pixel_right = kp_left_second[cur_match[1]], kp_right_second[cur_match[1]]
        real_pixel_second_left = np.array(pixel_left)
        pixel_second_right = utils.project_3d_pt_to_pixel(K, extrinsic_camera_mat_second_frame_right, pt_3d)
        real_pixel_second_right = np.array(pixel_right)
        if np.linalg.norm(real_pixel_second_left - pixel_second_left) <= 2 \
                and np.linalg.norm(real_pixel_second_right - pixel_second_right) <= 2:
            idxs_supports_matches.append(i)
        else:
            idxs_unsupports_matches.append(i)
    return idxs_supports_matches, idxs_unsupports_matches
