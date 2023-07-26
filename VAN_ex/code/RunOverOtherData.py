import os

import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import cv2 as cv

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleAdjustment import BundleAdjustment
from VAN_ex.code.DB.DataBase import DataBase
from VAN_ex.code.LoopClosure import LoopClosure
from VAN_ex.code.PoseGraph import PoseGraph
from VAN_ex.code.ProjectReport import from_gtsam_poses_to_world_coordinates_mats, slice_cov_mat_to_localization_angle

OUTPUT_DIR = 'results/run_over_other_data/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def show_locations_trajectory(ground_truth_locations, pnp_locations, bundle_adjustment_locations,
                              loop_closure_locations):
    fig, ax = plt.subplots(figsize=(19.2, 14.4))
    ax.scatter(x=ground_truth_locations[:, 0], y=ground_truth_locations[:, 2],
               label='Ground Truth', s=10, alpha=0.4, c='tab:blue')
    ax.scatter(x=pnp_locations[:, 0], y=pnp_locations[:, 2],
               label='PnP', s=10, alpha=0.7, c='tab:orange')
    ax.scatter(x=bundle_adjustment_locations[:, 0], y=bundle_adjustment_locations[:, 2],
               label='Bundle Adjustment', s=30, alpha=0.7, c='tab:green')
    ax.scatter(x=loop_closure_locations[:, 0], y=loop_closure_locations[:, 2],
               label='Loop Closure', s=30, alpha=0.7, c='tab:red')
    ax.legend(fontsize='20')
    ax.set_title('Loclization Trajectory - bird eye view')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(f'{OUTPUT_DIR}localization.png')


def present_tracking_statistics(database):
    print("Total number of tracks: ", database.get_tracks_number())
    print("Number of frames: ", database.get_frames_number())
    mean, max_t, min_t = database.get_mean_max_and_min_track_len()
    print(f"Mean track length: {mean}\nMaximum track length: {max_t}\nMinimum track length: {min_t}")
    print("Mean number of tracks on average frame: ", database.get_mean_number_of_tracks_on_frame())


def plot_location_error_along_axes(ground_truth, estimation, title, keyframes_list):
    # ground truth and estimation is list of locations (vector with 3 coordinates)
    diff = np.abs(ground_truth - estimation)
    x_error = diff[:, 0]
    y_error = diff[:, 1]
    z_error = diff[:, 2]
    norm_error = np.sum(diff ** 2, axis=-1) ** 0.5
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=keyframes_list, y=x_error, mode='lines', name='x'))
    fig.add_trace(go.Scatter(x=keyframes_list, y=y_error, mode='lines', name='y'))
    fig.add_trace(go.Scatter(x=keyframes_list, y=z_error, mode='lines', name='z'))
    fig.add_trace(go.Scatter(x=keyframes_list, y=norm_error, mode='lines', name='norm'))

    fig.update_layout(title=f'Axes Estimation Error - {title}', xaxis_title='Keyframe id', yaxis_title='error')
    fig.write_image(f'{OUTPUT_DIR}axes_estimation_error_{title}.png')

def plot_angle_error(ground_truth_matrices, estimation_mats, title, keyframes_list):
    length = ground_truth_matrices.shape[0]
    angle_error = np.empty(length)
    for i in range(length):
        gt_vec, _ = cv.Rodrigues(ground_truth_matrices[i][:, 3])
        est_vec, _ = cv.Rodrigues(estimation_mats[i][:, 3])
        angle_error[i] = np.linalg.norm(gt_vec - est_vec)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=keyframes_list, y=angle_error, mode='lines', name='angle'))

    fig.update_layout(title=f'Angle Estimation Error - {title}', xaxis_title='Keyframe id', yaxis_title='error')
    fig.write_image(f'{OUTPUT_DIR}angle_estimation_error_{title}.png')


def plot_uncertainty(cov_list_before, cov_list_after, keyframes, title):
    det_cov_before = [np.sqrt(np.linalg.det(before_cov)) for before_cov in cov_list_before]
    det_cov_after = [np.sqrt(np.linalg.det(after_cov)) for after_cov in cov_list_after]
    keyframes = np.array(keyframes)
    det_cov_before = np.array(det_cov_before)
    det_cov_after = np.array(det_cov_after)

    fig = plt.figure()
    plt.title(f"{title} uncertainty before and after Loop Closure")
    plt.plot(keyframes, det_cov_before, label="Before")
    plt.plot(keyframes, det_cov_after, label="After")
    plt.yscale('log')
    plt.ylabel("uncertainty -sqrt covariance determinate (log scale)")
    plt.xlabel("Keyframe")
    plt.legend()

    ax = plt.gca()
    ax.set_yticklabels([])

    fig.savefig(f'{OUTPUT_DIR}{title}_uncertainty_before_after_loop_closure.png')
    plt.close(fig)


def run_over_other_data():
    ground_truth_mats = np.array(utils.read_ground_truth_matrices(utils.GROUND_TRUTH_PATH))
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_mats)

    db = DataBase()

    db.fill_database(utils.FRAMES_NUM)
    db.save_database(utils.DB_PATH)

    # db.read_database(utils.DB_PATH)

    pnp_mats = db.get_left_camera_mats_in_world_coordinates()
    pnp_locations = utils.calculate_ground_truth_locations_from_matrices(pnp_mats)

    bundle_adjustment = BundleAdjustment(utils.FRAMES_NUM, 20, db)
    bundle_adjustment.optimize_all_windows()
    bundle_adjustment_poses = bundle_adjustment.get_global_keyframes_poses()  # dict
    bundle_adjustment_mats = from_gtsam_poses_to_world_coordinates_mats(bundle_adjustment_poses)
    bundle_adjustment_locations = \
        BundleAdjustment.from_poses_to_locations(bundle_adjustment_poses)

    pose_graph = PoseGraph(bundle_adjustment)
    loop_closure = LoopClosure(db, pose_graph, threshold_close=500,
                               threshold_inliers_rel=0.4)
    full_cov_before_LC = np.array(pose_graph.get_covraince_all_poses())
    loops_dict = loop_closure.find_loops('', plot=False)
    full_cov_after_LC = np.array(pose_graph.get_covraince_all_poses())
    loop_closure_poses = pose_graph.get_global_keyframes_poses()  # dict
    loop_closure_mats = from_gtsam_poses_to_world_coordinates_mats(loop_closure_poses)
    loop_closure_locations = BundleAdjustment.from_poses_to_locations(loop_closure_poses)

    show_locations_trajectory(ground_truth_locations, pnp_locations, bundle_adjustment_locations,
                              loop_closure_locations)

    present_tracking_statistics(db)

    keyframes_list = bundle_adjustment.get_keyframes()
    ground_truth_mats_of_keyframes = ground_truth_mats[keyframes_list]
    ground_truth_locations_of_keyframes = ground_truth_locations[keyframes_list]

    plot_location_error_along_axes(ground_truth_locations_of_keyframes, loop_closure_locations, 'LC', keyframes_list)
    plot_angle_error(ground_truth_mats_of_keyframes, loop_closure_mats, 'LC')

    # uncertainty size vs keyframes
    covs_angle_before, covs_location_before = slice_cov_mat_to_localization_angle(full_cov_before_LC)
    covs_angle_after, covs_location_after = slice_cov_mat_to_localization_angle(full_cov_after_LC)
    plot_uncertainty(covs_angle_before, covs_angle_after, keyframes_list, "Angle")
    plot_uncertainty(covs_location_before, covs_location_after, keyframes_list, "Location")

if __name__ == '__main__':
    run_over_other_data()