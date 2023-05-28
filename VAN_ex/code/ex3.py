import random
import cv2 as cv
import numpy as np
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import utils
import os

OUTPUT_DIR = 'results\\ex3\\'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDENTATION_RIGHT_CAM_MAT = None
K = None


def project_3d_pt_to_pixel(extrinsic_camera_mat, pt_3d):
    pt_3d_h = np.append(pt_3d, [1])
    projected = K @ extrinsic_camera_mat @ pt_3d_h
    return projected[0:2] / projected[2]


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def estimate_frame1_mats_pnp(points_2d, points_3d, flag=cv.SOLVEPNP_P3P):
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv.solvePnP(points_3d, points_2d, K, dist_coeffs, flags=flag)
    if success == 0:
        return None, None
    extrinsic_camera_mat_left1 = rodriguez_to_mat(rotation_vector, translation_vector)
    extrinsic_camera_mat_right1 = np.array(extrinsic_camera_mat_left1, copy=True)
    extrinsic_camera_mat_right1[0][3] += INDENTATION_RIGHT_CAM_MAT
    return extrinsic_camera_mat_left1, extrinsic_camera_mat_right1


def sample_4_points(kp0_left_l, kp1_left_l, points_cloud_0):
    rand_idxs = random.sample(range(len(kp1_left_l) - 1), 4)
    points_3d = np.empty((4, 3))
    points_2d = np.empty((4, 2))
    for i in range(4):
        points_3d[i] = points_cloud_0[kp0_left_l[rand_idxs[i]]]
        points_2d[i] = kp1_left_l[rand_idxs[i]].pt
    return points_2d, points_3d


def find_supporters(kp0_left_l, kp1_left_l, points_cloud_0, extrinsic_camera_mat_left1, extrinsic_camera_mat_right1,
                    dict_matches1, is_find_unsupported=False):
    supporter_left0, supporter_left1 = [], []
    unsupporter_left0, unsupporter_left1 = [], []
    for i in range(len(kp1_left_l)):
        pt_3d = points_cloud_0[kp0_left_l[i]]
        p_left1 = project_3d_pt_to_pixel(extrinsic_camera_mat_left1, pt_3d)
        real_p_left1 = kp1_left_l[i].pt
        p_right1 = project_3d_pt_to_pixel(extrinsic_camera_mat_right1, pt_3d)
        real_p_right1 = dict_matches1[kp1_left_l[i]].pt
        if np.linalg.norm(real_p_left1 - p_left1) <= 2 and np.linalg.norm(real_p_right1 - p_right1) <= 2:
            supporter_left0.append(kp0_left_l[i])
            supporter_left1.append(kp1_left_l[i])
        elif is_find_unsupported:
            unsupporter_left0.append(kp0_left_l[i])
            unsupporter_left1.append(kp1_left_l[i])
    return supporter_left0, supporter_left1, unsupporter_left0, unsupporter_left1


def calc_max_iterations(p, eps, size):
    return np.log(1 - p) / np.log(1 - np.power(1 - eps, size))


def RANSAC(kp0_left_l, kp1_left_l, points_cloud_0, dict_matches1, dict_matches0_1_left):
    p, eps = 0.99, 0.99  # eps = prob to be outlier
    i = 0
    size = len(kp0_left_l)
    max_supporters_num = 0
    max_supporters_left0 = None
    while eps > 0 and i < calc_max_iterations(p, eps, 4):
        points_2d, points_3d = sample_4_points(kp0_left_l, kp1_left_l, points_cloud_0)
        extrinsic_camera_mat_left1, extrinsic_camera_mat_right1 = estimate_frame1_mats_pnp(points_2d, points_3d)
        if extrinsic_camera_mat_left1 is None:
            continue
        supporter_left0, supporter_left1, _, _ = find_supporters(kp0_left_l, kp1_left_l, points_cloud_0,
                                                                 extrinsic_camera_mat_left1,
                                                                 extrinsic_camera_mat_right1,
                                                                 dict_matches1)
        supporters_num = len(supporter_left0)
        if supporters_num > max_supporters_num:
            max_supporters_num = supporters_num
            max_supporters_left0 = supporter_left0
            # update eps
            eps = (size - max_supporters_num) / size
        i += 1

    points_3d_supporters = np.empty((max_supporters_num, 3))
    points_2d_supporters = np.empty((max_supporters_num, 2))
    for i in range(max_supporters_num):
        points_3d_supporters[i] = points_cloud_0[max_supporters_left0[i]]
        points_2d_supporters[i] = dict_matches0_1_left[max_supporters_left0[i]].pt
    extrinsic_camera_mat_left1, extrinsic_camera_mat_right1 = \
        estimate_frame1_mats_pnp(points_2d_supporters, points_3d_supporters, flag=cv.SOLVEPNP_ITERATIVE)
    supporter_left0, supporter_left1, unsupporter_left0, unsupporter_left1 = \
        find_supporters(kp0_left_l, kp1_left_l, points_cloud_0,
                        extrinsic_camera_mat_left1,
                        extrinsic_camera_mat_right1,
                        dict_matches1)
    return extrinsic_camera_mat_left1, supporter_left0, supporter_left1, unsupporter_left0, unsupporter_left1


def find_cam_location_and_concat_mats(prev_concat_r, prev_concat_t, cur_r, cur_t):
    concat_r = cur_r @ prev_concat_r
    concat_t = cur_r @ prev_concat_t + cur_t
    left_cur_pos = ((-concat_r.T @ concat_t).reshape(1, 3))[0]
    return concat_r, concat_t, left_cur_pos


def read_frame_match_triangulate(idx_frame1, to_rectified=True):
    img1_left, img1_right = utils.read_images(idx_frame1)
    kp1_left, des1_left, kp1_right, des1_right = utils.detect_and_compute(img1_left, img1_right)
    matches1 = utils.find_matches(des1_left, des1_right)
    kp1_left, des1_left, kp1_right, des1_right, dict_matches1 = utils.get_correlated_kps_and_des_from_matches(kp1_left,
                                                                                                              des1_left,
                                                                                                              kp1_right,
                                                                                                              des1_right,
                                                                                                              matches1)
    if to_rectified:
        kp1_left, des1_left, kp1_right, des1_right = utils.rectified_stereo_pattern_test(kp1_left, des1_left, kp1_right,
                                                                                         des1_right)
    points_cloud_1 = utils.triangulation_list_of_points(kp1_left, kp1_right)
    return des1_left, des1_right, dict_matches1, kp1_left, kp1_right, points_cloud_1


def localization_two_frames(idx_frame0, kp0_left, des0_left, points_cloud_0, prev_concat_r, prev_concat_t):
    des1_left, des1_right, dict_matches1, kp1_left, kp1_right, points_cloud_1 = read_frame_match_triangulate(
        idx_frame0 + 1)
    matches_0_1_left = utils.find_matches(np.array(des0_left), np.array(des1_left))
    kp0_left_l, des0_left_l, kp1_left_l, des1_left_l, dict_matches0_1_left = utils.get_correlated_kps_and_des_from_matches(
        kp0_left, des0_left, kp1_left, des1_left, matches_0_1_left)
    extrinsic_camera_mat_left1, supporter_left0, supporter_left1, unsupporter_left0, unsupporter_left1 = \
        RANSAC(kp0_left_l, kp1_left_l, points_cloud_0, dict_matches1,
               dict_matches0_1_left)
    concat_r, concat_t, left_cur_pos = find_cam_location_and_concat_mats \
        (prev_concat_r, prev_concat_t, extrinsic_camera_mat_left1[:, :3], extrinsic_camera_mat_left1[:, 3])
    return kp1_left, des1_left, points_cloud_1, concat_r, concat_t, left_cur_pos


def read_matrices(path):
    matrices = []
    with open(path) as f:
        for line in f:
            list_line = [float(i) for i in line.split()]
            m1 = np.array(list_line).reshape(3, 4)
            matrices.append(m1)
    return matrices


def ex3_run():
    # 3.1
    img0_left, img0_right, kp0_left, des0_left, kp0_right, des0_right, points_cloud_0, dict_matches0 = utils.match_and_triangulate_image(0)
    img1_left, img1_right, kp1_left, des1_left, kp1_right, des1_right, points_cloud_1, dict_matches1 = utils.match_and_triangulate_image(1)

    # 3.2
    matches_0_1_left = utils.find_matches(np.array(des0_left), np.array(des1_left))

    # 3.3
    random.seed(4)
    kp0_left_l, des0_left_l, kp1_left_l, des1_left_l, dict_matches0_1_left =\
        utils.get_correlated_kps_and_des_from_matches(kp0_left, des0_left, kp1_left, des1_left, matches_0_1_left)
    points_2d, points_3d = sample_4_points(kp0_left_l, kp1_left_l, points_cloud_0)
    global K
    K, extrinsic_camera_mat_left0, extrinsic_camera_mat_right0 = utils.read_cameras()
    global INDENTATION_RIGHT_CAM_MAT
    INDENTATION_RIGHT_CAM_MAT = extrinsic_camera_mat_right0[0][3]
    extrinsic_camera_mat_left1, extrinsic_camera_mat_right1 = estimate_frame1_mats_pnp(points_2d, points_3d)

    plot_cameras_positions(extrinsic_camera_mat_left1, extrinsic_camera_mat_right0, extrinsic_camera_mat_right1)

    # 3.4
    # we used the points_cloud_0 because there, the 3d points are aligned to the global coordinate
    supporter_left0, supporter_left1, unsupporter_left0, unsupporter_left1 =\
        find_supporters(kp0_left_l, kp1_left_l, points_cloud_0, extrinsic_camera_mat_left1, extrinsic_camera_mat_right1, dict_matches1, is_find_unsupported=True)

    plot_supporters_and_unsupporters_on_img(img0_left, supporter_left0, unsupporter_left0, 'img0_left_supporters.png')
    plot_supporters_and_unsupporters_on_img(img1_left, supporter_left1, unsupporter_left1, 'img1_left_supporters.png')

    # 3.5
    extrinsic_camera_mat_left1, \
        supporter_left0, supporter_left1, unsupporter_left0, unsupporter_left1 = \
        RANSAC(kp0_left_l, kp1_left_l, points_cloud_0, dict_matches1, dict_matches0_1_left)

    R, t = extrinsic_camera_mat_left1[:, :3], extrinsic_camera_mat_left1[:, 3]
    estimated_3d_point_frame1 = np.zeros((len(points_cloud_0), 3))
    pts_cloud_frame0 = list(points_cloud_0.values())
    pt_len = len(pts_cloud_frame0)
    for i in range(pt_len):
        estimated_3d_point_frame1[i] = R @ pts_cloud_frame0[i] + t

    utils.show_dots_3d_cloud([estimated_3d_point_frame1, list(points_cloud_1.values())],
                             ['points from frame 1 (after triangulation)',
                              'points from frame 0 (after transformation T'],
                             ['blue', 'red'],
                             f'{OUTPUT_DIR}3d_point_cloud_frame1 triangulation vs transformation.html')

    plot_supporters_and_unsupporters_on_img(img0_left, supporter_left0, unsupporter_left0,
                                            'img0_left_supporters_after_ransac.png')
    plot_supporters_and_unsupporters_on_img(img1_left, supporter_left1, unsupporter_left1,
                                            'img1_left_supporters_after_ransac.png')

    # 3.6
    left_cam_poses = []
    concat_r, concat_t = extrinsic_camera_mat_left0[:, :3], extrinsic_camera_mat_left0[:, 3]
    for i in range(2559):
        print(f'---- frame iteration{i}----')
        kp1_left, des1_left, points_cloud_1, concat_r, concat_t, left_cur_pos = \
            localization_two_frames(i, kp0_left, des0_left, points_cloud_0, concat_r, concat_t)
        left_cam_poses.append(left_cur_pos)
        kp0_left, des0_left, points_cloud_0 = kp1_left, des1_left, points_cloud_1

    left_cam_poses = np.array(left_cam_poses)
    x = left_cam_poses[:, 0]
    z = left_cam_poses[:, 2]

    show_localization(left_cam_poses)


def show_localization(locations):
    real_x, real_z = read_ground_truth()
    fig, ax = plt.subplots()
    ax.scatter(x=real_x, y=real_z, c='tab:orange', label='Ground truth localization', s=0.3, alpha=0.5)
    ax.scatter(x=locations[:, 0], y=locations[:, 2], label='Our_estimated_localization', s=0.5, alpha=0.7)
    ax.legend()
    plt.title('Estimated vs Real localization')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(f'{OUTPUT_DIR}localization.png')


def read_ground_truth():
    ground_truth_matrices = read_matrices("C:\\Users\\Miryam\\SLAM\\VAN_ex\\dataset\\poses\\05.txt")
    real_left_cam_poses = []
    for mat in ground_truth_matrices:
        pos = ((-(mat[:, :3]).T @ mat[:, 3]).reshape(1, 3))[0]
        real_left_cam_poses.append(pos)
    real_left_cam_poses = np.array(real_left_cam_poses)
    real_x = real_left_cam_poses[:, 0]
    real_z = real_left_cam_poses[:, 2]
    return real_x, real_z


def plot_supporters_and_unsupporters_on_img(img0_left, supporter_left0, unsupporter_left0, name):
    img0_left = cv.drawKeypoints(img0_left, supporter_left0, img0_left, color=(46, 139, 87))
    img0_left = cv.drawKeypoints(img0_left, unsupporter_left0, img0_left, color=(0, 0, 128))
    cv.imwrite(OUTPUT_DIR + name, img0_left)
    plt.imshow(img0_left), plt.show()


def plot_cameras_positions(extrinsic_camera_mat_left1, extrinsic_camera_mat_right0, extrinsic_camera_mat_right1):
    left0_pos = [0, 0, 0]
    right0_pos = -extrinsic_camera_mat_right0[:, 3]
    left1_pos = ((-(extrinsic_camera_mat_left1[:, 0:3]).T @ extrinsic_camera_mat_left1[:, 3]).reshape(1, 3))[0]
    right1_pos = -(extrinsic_camera_mat_right1[:, 0:3]).T @ (extrinsic_camera_mat_right1[:, 3])
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
    fig.write_html(f'{OUTPUT_DIR}cameras_pos.html')


ex3_run()
