import os
import gtsam
import numpy as np
from matplotlib import pyplot as plt

import utils
import plotly.graph_objs as go
from DB.DataBase import DataBase
from BundleAdjustment.BundleWindow import BundleWindow

OUTPUT_DIR = 'results/ex5/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _add_pixels(grid, measurement_pixel, color, label):
    grid[0].scatter(measurement_pixel[0], measurement_pixel[2], s=1, c=color)
    grid[1].scatter(measurement_pixel[1], measurement_pixel[2], s=1, c=color, label=label)


def show_pixels_before_and_after_optimize(frame_id, track_id, measurement_pixel, before_pixel, after_pixel):
    fig, grid = plt.subplots(1, 2)
    fig.set_figwidth(20)
    fig.set_figheight(10)
    fig.suptitle(f"Track {track_id}, in frame {frame_id}")
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
        reprojection_error[i] = np.linalg.norm(diff)  # todo - check what is the factor error

        factor = gtsam.GenericStereoFactor3D \
            (pt_2d_real, cov, gtsam.symbol('c', i), gtsam.symbol('q', track.get_id()), K)
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
    K = BundleWindow.create_intrinsic_mat()
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
    return 0


run_ex5()
