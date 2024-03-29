import os
import gtsam
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

import utils
from DB.DataBase import DataBase
from Bundle.BundleAdjustment import BundleAdjustment
from VAN_ex.code.Bundle.BundleWindow import BundleWindow

OUTPUT_DIR = 'results/ex5/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_LEN = 20

def create_intrinsic_mat():
    """
    read intrinsic matrix from a file and create gtsam object of it
    :return: gtsam object that represents the intrinsic camera matrix of
    """
    k, _, m_right = utils.read_cameras()
    indentation_right_cam = m_right[0][3]
    K = gtsam.Cal3_S2Stereo(k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], -indentation_right_cam)
    return K


def reprojection_and_factor_error(db):
    """
    Calculate and plot graphs of reprojection error, factor error and reprojection error as a function of factor error,
    All the error are evaluated from a random track
    :param db:
    """
    track = db.get_random_track_in_len(10)
    frames_cameras, real_pts_2d = create_frames_cameras_and_pixels_from_track(db, track)

    track_len = track.get_track_len()
    last_frame_camera = frames_cameras[-1]
    last_frame_pt_2d = real_pts_2d[-1]
    pt_3d = last_frame_camera.backproject(last_frame_pt_2d)
    reprojection_error_left = np.empty(track_len)
    reprojection_error_right = np.empty(track_len)
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
        reprojection_error_left[i] = np.linalg.norm(diff[[0, 2]])
        reprojection_error_right[i] = np.linalg.norm(diff[[1, 2]])

        factor = gtsam.GenericStereoFactor3D \
            (pt_2d_real, cov, gtsam.symbol('c', i), gtsam.symbol('q', track.get_id()), BundleWindow.K)
        factor_error[i] = factor.error(values)

    frames_arr = np.array(list(track.get_frames_dict().keys()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frames_arr, y=reprojection_error_right, mode='lines+markers', name='right'))
    fig.add_trace(
        go.Scatter(x=frames_arr, y=reprojection_error_left, mode='lines+markers', name='left'))
    fig.add_trace(
        go.Scatter(x=frames_arr, y=factor_error, mode='lines+markers', name='factor error'))
    fig.update_layout(title=f"Reprojection error size (L2 norm) over track {track.get_id()} images",
                      xaxis_title='Frame id', yaxis_title='Reprojection error')
    fig.write_image(f"{OUTPUT_DIR}reprojection_error.png")

    factor_err_as_func_reproj_err(track.get_id(), factor_error, reprojection_error)


def factor_err_as_func_reproj_err(track_id, factor_err, reproj_err):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reproj_err, y=factor_err, mode='lines+markers'))
    fig.update_layout(title=f"Factor err as a function of the reprojection error over track {track_id} images",
                      xaxis_title='Reprojection err', yaxis_title='Factor err')
    fig.write_image(f"{OUTPUT_DIR}factor_err_of_reproj_err.png")


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


def q5_2(db):
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
    show_pixels_before_and_after_optimize(frame_id, measurement, before, after)
    bundle_window.plot_3d_positions_graph(f'{OUTPUT_DIR}resulting_3d_positions.png')
    bundle_window.plot_2d_positions_graph(f'{OUTPUT_DIR}resulting_2d_positions.png')


def q5_3(db):
    ground_truth_matrices = utils.read_ground_truth_matrices()
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_matrices)
    alg = BundleAdjustment(utils.FRAMES_NUM, WINDOW_LEN, db)
    alg.optimize_all_windows()
    last_window = alg.get_last_window()
    print("Final position of first frame of last window: ",
          last_window.get_frame_location(last_window.get_first_keyframe_id()))
    print("Prior factor error of last window: ", last_window.get_prior_factor_error())
    global_poses = alg.get_global_keyframes_poses()
    global_locations = BundleAdjustment.from_poses_to_locations(global_poses)
    global_3d_points = alg.get_global_3d_points(global_poses)
    initial_estimate = db.get_camera_locations()
    show_localization_after_optimization(global_3d_points, global_locations, ground_truth_locations, initial_estimate)
    keyframes_localization_error(alg, global_locations, ground_truth_locations, initial_estimate)


def _show_pixels_over_image(img, measurement_pixel, before_pixel, after_pixel, side):
    fig = px.imshow(img, color_continuous_scale='gray')
    fig.update_traces(dict(showscale=False, coloraxis=None, colorscale='gray'))
    fig.add_trace(
        go.Scatter(x=[measurement_pixel[0]], y=[measurement_pixel[1]],
                   marker=dict(color='red', size=5), name='measurement'))
    fig.add_trace(
        go.Scatter(x=[before_pixel[0]], y=[before_pixel[1]],
                   marker=dict(color='blue', size=5), name='before optimization'))
    fig.add_trace(
        go.Scatter(x=[after_pixel[0]], y=[after_pixel[1]],
                   marker=dict(color='green', size=5), name='after optimization'))
    fig.update_layout(title=dict(text=f"{side}"))
    fig.write_html(f'{OUTPUT_DIR}compare_pixel_projections_{side}.html')


def show_pixels_before_and_after_optimize(frame_id, measurement_pixel, before_pixel, after_pixel):
    img_left, img_right = utils.read_images(frame_id)
    _show_pixels_over_image(img_left, measurement_pixel[[0, 2]], before_pixel[[0, 2]], after_pixel[[0, 2]], 'Left')
    _show_pixels_over_image(img_right, measurement_pixel[[1, 2]], before_pixel[[1, 2]], after_pixel[[1, 2]], 'Right')


def keyframes_localization_error(alg, global_locations, ground_truth_locations, initial_estimate):
    keyframes = alg.get_keyframes()
    keyframe_localization_error = np.sum((ground_truth_locations[keyframes] - global_locations) ** 2, axis=-1) ** 0.5
    initial_estimate_error = np.sum((ground_truth_locations[keyframes] - initial_estimate[keyframes]) ** 2,
                                    axis=-1) ** 0.5
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=keyframes, y=keyframe_localization_error, mode='lines+markers', name='after optimization error'))
    fig.add_trace(
        go.Scatter(x=keyframes, y=initial_estimate_error, mode='lines+markers', name='initial estimate error'))
    fig.update_layout(title="Keyframe localization error over time",
                      xaxis_title='Keyframe id', yaxis_title='localization error')
    fig.write_image(f"{OUTPUT_DIR}localization error over time.png")


def show_localization_after_optimization(global_3d_points, global_locations, ground_truth_locations, initial_estimate):
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    ax.scatter(x=global_locations[:, 0], y=global_locations[:, 2],
               c='tab:blue', label='Keyframes positions', s=3, alpha=1, linestyle='-', marker='o')

    ax.scatter(x=global_3d_points[:, 0], y=global_3d_points[:, 2],
               c='tab:orange', label='Points', s=0.1, alpha=0.1)
    ax.scatter(x=ground_truth_locations[:, 0], y=ground_truth_locations[:, 2],
               c='tab:green', label='Ground Truth Locations', s=0.5, alpha=0.7)
    ax.scatter(x=initial_estimate[:, 0], y=initial_estimate[:, 2],
               c='tab:purple', label='Initial estimate', s=0.5, alpha=0.7)

    ax.set_ylim(-200, 200)
    ax.set_xlim(-300, 300)
    ax.legend()

    plt.title('Cameras & Points - View from above')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(f'{OUTPUT_DIR}Key frames after optimization.png')


def run_ex5():
    db = DataBase()
    db.read_database(utils.DB_PATH)

    # 5.1
    reprojection_and_factor_error(db)

    # 5.2 bundle adjustment window
    q5_2(db)

    # 5.3 convert to world coordinates, plot the localization and the error
    q5_3(db)

    return 0


if __name__ == '__main__':
    run_ex5()
