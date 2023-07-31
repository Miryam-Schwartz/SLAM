import cv2 as cv
import gtsam
import numpy as np
# from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from matplotlib import pyplot as plt

# current year data
DATA_PATH = r'/mnt/c/Users/Miryam/SLAM/VAN_ex/dataset/sequences/05/'
DB_PATH = r'/mnt/c/Users/Miryam/SLAM/VAN_ex/code/DB/files/'
GROUND_TRUTH_PATH = '/mnt/c/Users/Miryam/SLAM/VAN_ex/dataset/poses/05.txt'
FRAMES_NUM = 2560

# last year data
# DATA_PATH = r'/mnt/c/Users/Miryam/SLAM/VAN_ex/dataset_last_year/sequences/00/'
# DB_PATH = r'/mnt/c/Users/Miryam/SLAM/VAN_ex/code/DB_last_year/'
# GROUND_TRUTH_PATH = '/mnt/c/Users/Miryam/SLAM/VAN_ex/dataset_last_year/poses/00.txt'
# FRAMES_NUM = 3450


def read_images(idx):
    """
    Get index of picture from the data set and read them
    :param idx: index
    :return: left and right cameras photos
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv.imread(DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return img1, img2


def detect_and_compute(img1, img2, detector_type = 'SIFT'):
    """
    Call CV algorithms sift to detect and compute features in right and left pictures
    :param img1:
    :param img2:
    :return: key points and descriptors for right & left images
    """
    detector = None
    if detector_type == 'SIFT':
        detector = cv.SIFT_create()
    if detector_type == 'ORB': #work bed (ransac failed)
        detector = cv.ORB_create()
    if detector_type == 'AKAZE': #work
        detector = cv.AKAZE_create()
    if detector_type == 'BRIEF':
        star = cv.xfeatures2d.StarDetector_create()
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
        kp1 = star.detect(img1,None)
        kp2 = star.detect(img2,None)
        kp1, des1 = brief.compute(img1, kp1)
        kp2, des2 = brief.compute(img2, kp2)
        return kp1, des1, kp2, des2
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
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
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
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
        dots_3d[kp2[i]] = ret_pt_3d
    return dots_3d


def match_and_triangulate_image(im_idx, rectified_test=False,
                                triangulation_func=triangulation_single_match, threshold=1):
    img1, img2 = read_images(im_idx)
    kp1, des1, kp2, des2 = detect_and_compute(img1, img2)
    matches = find_matches(des1, des2)
    kp1, des1, kp2, des2, dict_left_to_right = get_correlated_kps_and_des_from_matches(kp1, des1, kp2, des2, matches)
    if rectified_test:
        kp1, des1, kp2, des2 = rectified_stereo_pattern_test(kp1, des1, kp2, des2, threshold)
    dots_3d = triangulation_list_of_points(kp1, kp2, triangulation_func)
    return img1, img2, kp1, des1, kp2, des2, dots_3d, dict_left_to_right


def get_correlated_kps_and_des_from_matches(kp1, des1, kp2, des2, matches):
    kp1_ret, kp2_ret = [], []
    des1_ret, des2_ret = [], []
    dict_left_to_right = dict()
    for i, match in enumerate(matches):
        kp1_ret.append(kp1[match.queryIdx])
        kp2_ret.append(kp2[match.trainIdx])
        des1_ret.append(des1[match.queryIdx])
        des2_ret.append(des2[match.trainIdx])
        dict_left_to_right[kp1[match.queryIdx]] = kp2[match.trainIdx]
    return kp1_ret, des1_ret, kp2_ret, des2_ret, dict_left_to_right


def show_dots_3d_cloud(arrays_of_dots_3d, names_of_arrays, colors, output_name, show=False):
    fig = go.Figure(
        go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=5, color='yellow'), name='camera'))
    for i in range(len(arrays_of_dots_3d)):
        dots_3d = np.array(arrays_of_dots_3d[i])
        x = dots_3d[:, 0].reshape(-1)
        y = dots_3d[:, 1].reshape(-1)
        z = dots_3d[:, 2].reshape(-1)
        trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color=colors[i]),
                             name=names_of_arrays[i])
        fig.add_trace(trace)
        fig.update_layout(scene=dict(xaxis_range=[-40, 30], yaxis_range=[-35, 10], zaxis_range=[0, 400]))
    if show:
        fig.show()
    fig.write_html(output_name)


def calc_max_iterations(p, eps, size):
    return np.log(1 - p) / np.log(1 - np.power(1 - eps, size))


def estimate_second_frame_mats_pnp(points_2d, points_3d, identation_right_cam_mat, K, flag=cv.SOLVEPNP_P3P):
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv.solvePnP(points_3d, points_2d, K, dist_coeffs, flags=flag)
    if success == 0:
        return None, None
    extrinsic_camera_mat_left1 = rodriguez_to_mat(rotation_vector, translation_vector)
    extrinsic_camera_mat_right1 = np.array(extrinsic_camera_mat_left1, copy=True)
    extrinsic_camera_mat_right1[0][3] += identation_right_cam_mat
    return extrinsic_camera_mat_left1, extrinsic_camera_mat_right1


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def project_3d_pt_to_pixel(k, extrinsic_camera_mat, pt_3d):
    pt_3d_h = np.append(pt_3d, [1])
    projected = k @ extrinsic_camera_mat @ pt_3d_h
    return projected[0:2] / projected[2]


def create_hist(data_array, x_label, y_label, title, output_path):
    fig = plt.figure()
    plt.hist(data_array)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(output_path)


def read_ground_truth_matrices(path=GROUND_TRUTH_PATH):
    matrices = []
    with open(path) as f:
        for line in f:
            list_line = [float(i) for i in line.split()]
            m1 = np.array(list_line).reshape(3, 4)
            matrices.append(m1)
    return matrices


def calculate_ground_truth_locations_from_matrices(ground_truth_matrices):
    locations = []
    for mat in ground_truth_matrices:
        pos = ((-(mat[:, :3]).T @ mat[:, 3]).reshape(1, 3))[0]
        locations.append(pos)
    locations = np.array(locations)
    return locations


def show_localization(estimated_locations, ground_truth_locations, output_path, title='Estimated vs Real localization'):
    fig, ax = plt.subplots()
    if ground_truth_locations is not None:
        ax.scatter(x=ground_truth_locations[:, 0], y=ground_truth_locations[:, 2],
                   c='tab:orange', label='Ground truth localization', s=0.3, alpha=0.5)
    ax.scatter(x=estimated_locations[:, 0], y=estimated_locations[:, 2],
               label='Our_estimated_localization', s=0.5, alpha=0.7)
    ax.legend()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(output_path)


def invert_extrinsic_matrix(r_mat, t_vec):
    # origin extrinsic_mat take world coordinate -> return camera/pose coordinate
    # new extrinsic_mat (used by gtsam) take camera/pose coordinate -> return world coordinate
    new_t_vec = -np.transpose(r_mat) @ t_vec
    new_r_mat = r_mat.T
    return new_r_mat, new_t_vec


def get_stereo_point2(db, frame_id, kp_idx):
    left_pixel, right_pixel = db.get_frame_obj(frame_id).get_feature_pixels(kp_idx)
    x_l, x_r, y = left_pixel[0], right_pixel[0], (left_pixel[1] + right_pixel[1]) / 2
    return gtsam.StereoPoint2(x_l, x_r, y)


def keyframes_localization_error(alg, global_locations, ground_truth_locations, initial_estimate, output_path):
    keyframes = alg.get_keyframes()
    keyframe_localization_error = np.sum((ground_truth_locations[keyframes] - global_locations) ** 2, axis=-1) ** 0.5
    initial_estimate_error = np.sum((ground_truth_locations[keyframes] - initial_estimate) ** 2,
                                    axis=-1) ** 0.5
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=keyframes, y=keyframe_localization_error, mode='lines+markers', name='after optimization error'))
    fig.add_trace(
        go.Scatter(x=keyframes, y=initial_estimate_error, mode='lines+markers', name='initial estimate error'))
    fig.update_layout(title="Keyframe localization error over time",
                      xaxis_title='Keyframe id', yaxis_title='localization error')
    fig.write_image(output_path)

def inliers_percentage_graph(db, output_path):
    """
    Create and save inliers percentage graph -
    for each frame, the percentage of inliers out of all the features that were matched.
    Note: only inliers were been saved in the db, then, inliers percentage were calculated
    and saved before removing outliers
    :param db: database
    """
    frames_num = db.get_frames_number()
    inliers_percentage = np.empty(frames_num)
    for i in range(frames_num):
        inliers_percentage[i] = db.get_frame_obj(i).get_inliers_percentage()
    fig = px.line(x=np.arange(frames_num), y=inliers_percentage, title="Inliers percentage per frame",
                  labels={'x': 'Frame', 'y': 'Inliers Percentage'})
    fig.write_image(f"{output_path}inliers_percentage_graph.png")

def connectivity_graph(db, output_path):
    """
    Create and save a connectivity graph from the dataa base
    (For each frame, the number of tracks outgoing to the next frame)
    :param db: database
    """
    frames_num = db.get_frames_number()
    outgoing_frames = np.empty(frames_num)
    for i in range(frames_num):
        outgoing_frames[i] = db.get_frame_obj(i).get_number_outgoing_tracks()
    fig = px.line(x=np.arange(frames_num), y=outgoing_frames, title="Connectivity",
                  labels={'x': 'Frame', 'y': 'Outgoing tracks'})
    fig.write_image(f"{output_path}connectivity_graph.png")


def tracks_length_histogram(db, output_path):
    """
    Create and save tracks length histogram (how many tracks are in a specific length)
    :param db:
    """
    tracks_number = db.get_tracks_number()
    tracks_length = np.empty(tracks_number)
    for i in range(tracks_number):
        tracks_length[i] = db.get_track_obj(i).get_track_len()
    unique, count = np.unique(tracks_length, return_counts=True)
    fig = px.line(x=unique, y=count, title='Tracks Length Histogram',
                  labels={'x': 'Track length', 'y': 'Track #'})
    fig.write_image(f'{output_path}tracks_length_histogram.png')