import os
import gtsam
import numpy as np
from matplotlib import pyplot as plt

import utils
import plotly.graph_objs as go
from DB.DataBase import DataBase
# from Bundle import BundleWindow
from Bundle.BundleAdjustment import BundleAdjustment
# import Bundle
# from Bundle.BundleWindow import BundleWindow
# import Bundle.BundleWindow
# from Bundle.Bundle import Bundle
# from Bundle import BundleWindow
from VAN_ex.code.Bundle.BundleWindow import BundleWindow

OUTPUT_DIR = 'results/ex5/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_intrinsic_mat():
    k, _, m_right = utils.read_cameras()
    indentation_right_cam = m_right[0][3]
    K = gtsam.Cal3_S2Stereo(k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], -indentation_right_cam)
    return K

def _add_pixels(grid, measurement_pixel, color, label):
    grid[0].scatter(measurement_pixel[0], measurement_pixel[2], s=15, c=color, label=label)
    grid[1].scatter(measurement_pixel[1], measurement_pixel[2], s=15, c=color, label=label)


def show_pixels_before_and_after_optimize(frame_id, track_id, measurement_pixel, before_pixel, after_pixel):
    fig, grid = plt.subplots(1, 2)
    fig.set_figwidth(25)
    fig.set_figheight(4)
    fig.suptitle(f"Track {track_id}, in frame {frame_id}", size='xx-large')
    # fig.update_layout(font=dict(size=18))
    grid[0].set_title("Left")
    grid[1].set_title("Right")
    img_left, img_right = utils.read_images(frame_id)
    grid[0].axes.xaxis.set_visible(False)
    grid[1].axes.xaxis.set_visible(False)
    grid[0].axes.yaxis.set_visible(False)
    grid[1].axes.yaxis.set_visible(False)
    grid[0].imshow(img_left, cmap="gray", aspect='auto')
    grid[1].imshow(img_right, cmap="gray", aspect='auto')
    _add_pixels(grid, measurement_pixel, "r", "measurement")
    _add_pixels(grid, before_pixel, "b", "before_optimize")
    _add_pixels(grid, after_pixel, "g", "after_optimize")
    fig.savefig(f'{OUTPUT_DIR}compare_pixel_projections.png')
    # todo - check how to output it better


def reprojection_error_gtsam(db):
    track = db.get_random_track_in_len(10)
    frames_cameras, real_pts_2d = create_frames_cameras_and_pixels_from_track(db, track)

    track_len = track.get_track_len()
    last_frame_camera = frames_cameras[-1]
    # last_frame_id, last_kp_idx = track.get_last_frame_id_and_kp_idx()
    # last_frame_pt_2d = utils.get_stereo_point2(db, last_frame_id, last_kp_idx)
    last_frame_pt_2d = real_pts_2d[-1]
    # triangulation - 3d_pt in coordinates of first frame in track
    pt_3d = last_frame_camera.backproject(last_frame_pt_2d)
    # todo - seperate to error in left and right?
    reprojection_error = np.empty(track_len)
    factor_error = np.empty(track_len)
    cov = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    values = gtsam.Values()
    values.insert(gtsam.symbol('q', track.get_id()), pt_3d)
    for i, cam_frame in enumerate(frames_cameras):
        values.insert(gtsam.symbol('c', i), cam_frame.pose())
    for i, cam_frame in enumerate(frames_cameras):
        pt_2d_proj = cam_frame.project(pt_3d)
        pt_2d_real = real_pts_2d[i]
        diff = pt_2d_real - pt_2d_proj
        diff = np.array([diff.uL(), diff.uR(), diff.v()])
        reprojection_error[i] = np.linalg.norm(diff)

        factor = gtsam.GenericStereoFactor3D \
            (pt_2d_real, cov, gtsam.symbol('c', i), gtsam.symbol('q', track.get_id()), BundleWindow.K)
        factor_error[i] = factor.error(values)

    frames_arr = np.array(list(track.get_frames_dict().keys()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frames_arr, y=reprojection_error, mode='lines+markers', name='reprojection error'))
    fig.add_trace(
        go.Scatter(x=frames_arr, y=factor_error, mode='lines+markers', name='factor error'))
    fig.update_layout(title=f"Reprojection error size (L2 norm) over track {track.get_id()} images",
                      xaxis_title='Frame id', yaxis_title='Reprojection error')
    fig.write_image(f"{OUTPUT_DIR}reprojection_error.png")

    factor_err_as_func_reproj_err(track.get_id(), factor_error, reprojection_error)


def create_frames_cameras_and_pixels_from_track(db, track):
    frames_cameras = []
    real_pts_2d = []
    frames_dict = track.get_frames_dict()
    K = create_intrinsic_mat()
    concat_r = np.identity(3)
    concat_t = np.zeros(3)
    is_first_frame = True
    for frame_id, kp_idx in frames_dict.items():
        if is_first_frame:
            is_first_frame = False
        else:
            mat = db.get_frame_obj(frame_id).get_left_camera_pose_mat()
            concat_r = mat[:, :3] @ concat_r  # in coordinates of first frame of track
            concat_t = mat[:, :3] @ concat_t + mat[:, 3]
        inv_r, inv_t = utils.invert_extrinsic_matrix(concat_r, concat_t)
        cur_pos3 = gtsam.Pose3(np.hstack((inv_r, inv_t.reshape(3, 1))))
        cur_stereo_camera = gtsam.StereoCamera(cur_pos3, K)
        frames_cameras.append(cur_stereo_camera)
        real_pts_2d.append(utils.get_stereo_point2(db, frame_id, kp_idx))
    return frames_cameras, real_pts_2d


def factor_err_as_func_reproj_err(track_id, factor_err, reproj_err):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reproj_err, y=factor_err, mode='lines+markers'))
    fig.update_layout(title=f"Factor err as a function of the reprojection error over track {track_id} images",
                      xaxis_title='Reprojection err', yaxis_title='Factor err')
    fig.write_image(f"{OUTPUT_DIR}factor_err_of_reproj_err.png")


def run_ex5():
    db = DataBase()
    db.read_database(utils.DB_PATH)

    # 5.1
    reprojection_error_gtsam(db)

    # 5.2
    bundle_window = BundleWindow(db, 0, 10)
    factor_key = bundle_window.get_random_factor()
    frame_id, track_id = factor_key
    factor_error_before = bundle_window.factor_error(frame_id, track_id)
    print("Total factor error before optimization: ", bundle_window.total_factor_error())
    bundle_window.optimize()
    print("Total factor error after optimization: ", bundle_window.total_factor_error())
    factor_error_after = bundle_window.factor_error(frame_id, track_id)
    print(f"Pick random factor: frame id: {frame_id}, track id: {track_id}")
    print(f"factor error before optimize: {factor_error_before}")
    print(f"factor error after optimize: {factor_error_after}")
    measurement, before, after = bundle_window.get_pixels_before_and_after_optimize(frame_id, track_id)
    show_pixels_before_and_after_optimize(frame_id, track_id, measurement, before, after)

    bundle_window.plot_3d_positions_graph(f'{OUTPUT_DIR}resulting_3d_positions.png')
    bundle_window.plot_2d_positions_graph(f'{OUTPUT_DIR}resulting_2d_positions.png')

    # 5.3
    ground_truth_matrices = utils.read_matrices("/cs/usr/nava.goetschel/SLAM/VAN_ex/dataset/poses/05.txt")
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_matrices)
    alg = BundleAdjustment(2560, 10, db)
    alg.optimize_all_windows()

    last_window = alg.get_last_window()
    print("Final position of first frame of last window: ", last_window.get_frame_location(last_window.get_first_keyframe_id()))
    print("Prior factor error of last window: ", last_window.get_prior_factor_error())

    global_poses = alg.get_global_keyframes_poses()
    global_locations = BundleAdjustment.from_poses_to_locations(global_poses)
    global_3d_points = alg.get_global_3d_points(global_poses)

    fig, ax = plt.subplots()
    ax.scatter(x=global_locations[:, 0], y=global_locations[:, 2],
               c='tab:blue', label='Keyframes positions', s=3, alpha=1)
    ax.scatter(x=global_3d_points[:, 0], y=global_3d_points[:, 2],
               c='tab:orange', label='Points', s=0.1, alpha=0.1)
    ax.scatter(x=ground_truth_locations[:, 0], y=ground_truth_locations[:, 2],
               c='tab:green', label='Ground Truth Locations', s=0.5, alpha=0.7)
    # ax.set_ylim(0, 200)
    ax.set_ylim(-200, 200)
    ax.set_xlim(-300, 300)
    ax.legend()
    plt.title('Cameras & Points - View from above')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(f'{OUTPUT_DIR}Key frames after optimization.png')


    keyframes = alg.get_keyframes()
    keyframe_localization_error = np.sum((ground_truth_locations[keyframes] - global_locations)**2, axis=-1)**0.5

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=keyframes, y=keyframe_localization_error, mode='lines+markers'))
    fig.update_layout(title="Keyframe localization error over time",
                      xaxis_title='Keyframe id', yaxis_title='localization error')
    fig.write_image(f"{OUTPUT_DIR}localization error over time.png")

    return 0




run_ex5()
