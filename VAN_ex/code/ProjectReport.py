import os
import random
import statistics

import cv2 as cv
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
import plotly.graph_objs as go

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleAdjustment import BundleAdjustment
from VAN_ex.code.DB.DataBase import DataBase
from VAN_ex.code.DB.Frame import Frame
from VAN_ex.code.LoopClosure import LoopClosure
from VAN_ex.code.PoseGraph import PoseGraph

OUTPUT_DIR = 'results/project_report/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def number_of_matches_per_frame_graph(database):
    frames_num = db.get_frames_number()
    matches_number = np.empty(frames_num)
    for i in range(frames_num):
        matches_number[i] = db.get_frame_obj(i).get_kp_len()
    fig = px.line(x=np.arange(frames_num), y=matches_number, title="Number of matches per frame",
                  labels={'x': 'Frame', 'y': 'Number of matches'})
    fig.write_image(f"{OUTPUT_DIR}number_of_matches_graph.png")


def present_tracking_statistics(database):
    print("Total number of tracks: ", database.get_tracks_number())
    print("Number of frames: ", database.get_frames_number())
    mean, max_t, min_t = database.get_mean_max_and_min_track_len()
    print(f"Mean track length: {mean}\nMaximum track length: {max_t}\nMinimum track length: {min_t}")
    print("Mean number of tracks on average frame: ", database.get_mean_number_of_tracks_on_frame())
    number_of_matches_per_frame_graph(database)
    # utils.inliers_percentage_graph(db, output_path=OUTPUT_DIR)


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


def plot_bundle_error_before_after_opt(keyframes_list, mean_factor_error_before, mean_factor_error_after, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=keyframes_list, y=mean_factor_error_before, mode='lines', name='Before'))
    fig.add_trace(
        go.Scatter(x=keyframes_list, y=mean_factor_error_after, mode='lines', name='After'))
    fig.update_layout(title=f'{title} before and after BA optimization', xaxis_title='Keyframe id', yaxis_title=title)
    fig.write_image(f'{OUTPUT_DIR}{title}.png')


def PnP_median_projection_error_per_distance():
    random_track_ids = random.sample(range(db.get_tracks_number()), 500)
    # left_camera_mats_in_world_coordinates = db.get_left_camera_mats_in_world_coordinates()
    # key = distance from last frame of track, value = list of projection errors (from different tracks)
    right_projection_err = dict()
    left_projection_err = dict()
    for track_id in random_track_ids:
        track = db.get_track_obj(track_id)
        frames_dict = track.get_frames_dict()
        last_frame_id, last_kp_idx = track.get_last_frame_id_and_kp_idx()
        pt_3d = db.get_frame_obj(last_frame_id).get_3d_point(last_kp_idx)  # in coordinates of last frame
        last_frame_mat = pnp_mats[last_frame_id]
        r, t = last_frame_mat[:, :3], last_frame_mat[:, 3]
        world_coor_3d_pt = r.T @ (pt_3d - t)
        for frame_id, kp_idx in frames_dict.items():
            if frame_id == last_frame_id:
                break
            frame_obj = db.get_frame_obj(frame_id)
            extrinsic_left = pnp_mats[frame_id]
            extrinsic_right = np.array(extrinsic_left, copy=True)
            extrinsic_right[0][3] += Frame.INDENTATION_RIGHT_CAM_MAT
            proj_left_pixel = utils.project_3d_pt_to_pixel(Frame.k, extrinsic_left, world_coor_3d_pt)
            proj_right_pixel = utils.project_3d_pt_to_pixel(Frame.k, extrinsic_right, world_coor_3d_pt)
            real_pixel_left, real_pixel_right = frame_obj.get_feature_pixels(kp_idx)
            distance = last_frame_id - frame_id
            left_err = np.linalg.norm(proj_left_pixel - np.array(real_pixel_left))
            right_err = np.linalg.norm(proj_right_pixel - np.array(real_pixel_right))
            if distance in right_projection_err:
                left_projection_err[distance].append(left_err)
                right_projection_err[distance].append(right_err)
            else:
                left_projection_err[distance] = [left_err]
                right_projection_err[distance] = [right_err]

    distances = np.array(sorted(right_projection_err.keys()))
    median_proj_err_left = np.array([statistics.median(left_projection_err[d]) for d in distances])
    median_proj_err_right = np.array([statistics.median(right_projection_err[d]) for d in distances])
    links_num_per_distance = np.array([len(right_projection_err[d]) for d in distances])

    fig = plt.figure()
    plt.title('PnP - median projection error vs track length')
    plt.plot(distances, median_proj_err_left, label='left')
    plt.plot(distances, median_proj_err_right, label='right')
    plt.ylabel('projection error')
    plt.xlabel('distance from last frame')
    plt.legend()

    fig.savefig(f'{OUTPUT_DIR}PnP_median_projection_error_vs_track_length.png')
    plt.close(fig)

    return distances, links_num_per_distance

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=distances, y=median_proj_err_left, mode='lines', name='left'))
    # fig.add_trace(go.Scatter(x=distances, y=median_proj_err_right, mode='lines', name='right'))
    # fig.update_layout(title='PnP - median projection error vs track length',
    #                   xaxis_title='distance from last frame', yaxis_title='projection error')
    # fig.write_image(f'{OUTPUT_DIR}PnP_median_projection_error_vs_track_length.png')


def from_gtsam_poses_to_world_coordinates_mats(poses_dict):
    mats_list = []
    for pose_id in sorted(poses_dict.keys()):
        pose = poses_dict[pose_id]
        rotation = pose.rotation()
        r_vec = np.array([rotation.pitch(), rotation.roll(), rotation.yaw()])
        t_vec = np.array([pose.x(), pose.y(), pose.z()])
        r_mat, _ = cv.Rodrigues(r_vec)
        r_extrinsic, t_extrinsic = utils.invert_extrinsic_matrix(r_mat, t_vec)
        mats_list.append(np.hstack((r_extrinsic, t_extrinsic.reshape(3, 1))))
    return np.array(mats_list)


def BA_median_projection_error_per_distance():
    distances, median_proj_err_left, median_proj_err_right, links_num_per_distance =\
        bundle_adjustment.get_median_projection_error_per_distance()
    fig = plt.figure()
    plt.title('BA - median projection error vs distance from last frame')
    plt.plot(distances, median_proj_err_left, label='left')
    plt.plot(distances, median_proj_err_right, label='right')
    plt.ylabel('projection error')
    plt.xlabel('distance from last frame')
    plt.legend()

    fig.savefig(f'{OUTPUT_DIR}BA_median_projection_error_vs_distance_from_last_frame.png')
    plt.close(fig)

    return distances, links_num_per_distance

    # distances, median_proj_err_left, median_proj_err_right = bundle_adjustment.get_median_projection_error_per_distance()
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=distances, y=median_proj_err_left, mode='lines', name='left'))
    # fig.add_trace(go.Scatter(x=distances, y=median_proj_err_right, mode='lines', name='right'))
    # fig.update_layout(title='BA - median projection error vs distance from last frame',
    #                   xaxis_title='distance from last frame', yaxis_title='projection error')
    # fig.write_image(f'{OUTPUT_DIR}BA_median_projection_error_vs_distance_from_last_frame.png')


def plot_location_error_along_axes(ground_truth, estimation, title):
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


def plot_angle_error(ground_truth_matrices, estimation_mats, title):
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


def loop_closure_statistics():
    print("Number of Matches and Inliers Percentage in loop closure:")
    for loop, inliers_and_outliers in loops_dict.items():
        inliers_i, inliers_n, outliers_i, outliers_n = inliers_and_outliers
        inliers_percentage = (len(inliers_i) / (len(inliers_i) + len(outliers_i))) * 100
        print(f"loop {loop}: matches number - {len(inliers_i)}, inliers percentage - {inliers_percentage}")


def plot_uncertainty(cov_list_before, cov_list_after, keyframes, title):
    det_cov_before = [np.sqrt(np.linalg.det(before_cov)) for before_cov in cov_list_before]
    det_cov_after = [np.sqrt(np.linalg.det(after_cov)) for after_cov in cov_list_after]
    keyframes = np.array(keyframes)
    det_cov_before = np.array(det_cov_before)
    det_cov_after = np.array(det_cov_after)

    fig, ax = plt.subplots()
    ax.plot(keyframes, det_cov_before, label='Before')
    ax.plot(keyframes, det_cov_after, label='After')
    ax.legend()
    ax.set_title(f"{title} uncertainty before and after Loop Closure")
    plt.xlabel("Keyframe")
    plt.ylabel("uncertainty -sqrt covariance determinate (log scale)")
    plt.yscale('log')
    ax.get_yaxis().set_visible(False)
    plt.savefig(f'{OUTPUT_DIR}{title}_uncertainty_before_after_loop_closure.png')

    # fig = plt.figure()
    # plt.title(f"{title} uncertainty before and after Loop Closure")
    # plt.plot(keyframes, det_cov_before, label="Before")
    # plt.plot(keyframes, det_cov_after, label="After")
    # plt.yscale('log')
    # plt.ylabel("uncertainty -sqrt covariance determinate (log scale)")
    # plt.xlabel("Keyframe")
    # plt.legend()
    #
    # ax = plt.gca()
    # ax.set_yticklabels([])
    #
    # fig.savefig(f'{OUTPUT_DIR}{title}_uncertainty_before_after_loop_closure.png')
    # plt.close(fig)


def slice_cov_mat_to_localization_angle(cov_list):
    cov_list_angle = []
    cov_list_loc = []
    for cov in cov_list:
        cov_list_angle.append(cov[:3, :3])
        cov_list_loc.append(cov[3:, 3:])
    return cov_list_angle, cov_list_loc


def total_length_trajectory(start_seq, end_seq, gt_locations):
    total = 0
    for i in range(start_seq, end_seq):
        total += np.linalg.norm(gt_locations[i + 1] - gt_locations[i])
    return total


def calc_relative_motion(mat_a_to_b, mat_a_to_c):  # return mat_b_to_c
    r_b_to_a, t_b_to_a = utils.invert_extrinsic_matrix(mat_a_to_b[:, :3], mat_a_to_b[:, 3])
    new_r = mat_a_to_c[:, :3] @ r_b_to_a
    new_t = mat_a_to_c[:, :3] @ t_b_to_a + mat_a_to_c[:, 3]
    return new_r, new_t


def relative_estimation_error_in_sequences(estimation_mats, gt_mats, gt_locations, sequnce_len):
    # estimation_mats, gt_mats are in world coordinates
    mats_num = estimation_mats.shape[0]
    sequences = range(0, mats_num, sequnce_len)
    location_errors = np.empty(len(sequences))
    angle_errors = np.empty(len(sequences))
    for i, start_seq in enumerate(sequences):
        end_seq = min(start_seq + sequnce_len - 1, mats_num - 1)
        # mechane - sum differences locations
        total_length = total_length_trajectory(start_seq, end_seq, gt_locations)

        rel_estimation_R, rel_estimation_t = calc_relative_motion(estimation_mats[start_seq], estimation_mats[end_seq])
        rel_gt_R, rel_gt_t = calc_relative_motion(gt_mats[start_seq], gt_mats[end_seq])

        # location
        estimation_loc = ((-rel_estimation_R.T @ rel_estimation_t).reshape(1, 3))[0]
        gt_loc = ((-rel_gt_R.T @ rel_gt_t).reshape(1, 3))[0]
        err_loc = np.linalg.norm(estimation_loc - gt_loc)
        location_errors[i] = err_loc / total_length

        # angle
        rel_estimation_R_vec, _ = cv.Rodrigues(rel_estimation_R)
        rel_gt_R_vec, _ = cv.Rodrigues(rel_gt_R)
        err_angle = np.linalg.norm(rel_estimation_R_vec - rel_gt_R_vec)
        angle_errors[i] = err_angle / total_length

    return location_errors, angle_errors, sequences


def plot_relative_estimation_error_in_sequences(title_model, title_error_type, list_seq_len, list_errors, list_sequences_list):
    fig = plt.figure()
    plt.title(f"Relative {title_model} estimation {title_error_type} error with sequences of")
    avg = 0
    for i, seq_len in enumerate(list_seq_len):
        plt.plot(list_sequences_list[i], list_errors[i], label=str(seq_len))
        avg += np.average(list_errors[i])
    avg = avg / len(list_seq_len)
    if title_error_type == 'location':
        plt.ylabel("error (m/m)")
    else:
        plt.ylabel("error (deg/m)")
    if title_model == 'PnP':
        plt.xlabel("frameId")
    else:
        plt.xlabel("KeyFrameNum") #todo: find a way to convert it the actually frame id
    plt.legend()
    fig.savefig(f'{OUTPUT_DIR}{title_model}_{title_error_type}_relative_estimation_error_in_seq.png')
    plt.close(fig)
    print(f"Relative {title_model} estimation error - Avarege {title_error_type} error of all the sequences: {avg}")


def plot_links_num_per_distance(pnp_distances, pnp_links_num_per_distance, ba_distances, ba_links_num_per_distance):
    fig = plt.figure()
    plt.title('Links Number Per Distance From Last Frame - PnP vs BA')
    plt.plot(pnp_distances, pnp_links_num_per_distance, label='PnP')
    plt.plot(ba_distances, ba_links_num_per_distance, label='BA')
    plt.ylabel('Links Num')
    plt.xlabel('distance from last frame')
    plt.legend()
    plt.yscale('log')

    fig.savefig(f'{OUTPUT_DIR}links_num_per_distance.png')
    plt.close(fig)



if __name__ == '__main__':
    ground_truth_mats = np.array(utils.read_ground_truth_matrices(utils.GROUND_TRUTH_PATH))
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_mats)

    db = DataBase()
    db.read_database(utils.DB_PATH)
    pnp_mats = db.get_left_camera_mats_in_world_coordinates()
    pnp_locations = utils.calculate_ground_truth_locations_from_matrices(pnp_mats)
    print("finished read data")

    bundle_adjustment = BundleAdjustment(utils.FRAMES_NUM, 20, db)
    keyframes_list = bundle_adjustment.get_keyframes()
    mean_factor_error_before = bundle_adjustment.get_mean_factor_error_for_all_windows()
    median_projection_error_before = bundle_adjustment.get_median_projection_error_for_all_windows()

    bundle_adjustment.optimize_all_windows()

    mean_factor_error_after = bundle_adjustment.get_mean_factor_error_for_all_windows()
    median_projection_error_after = bundle_adjustment.get_median_projection_error_for_all_windows()
    bundle_adjustment_poses = bundle_adjustment.get_global_keyframes_poses()  # dict
    bundle_adjustment_mats = from_gtsam_poses_to_world_coordinates_mats(bundle_adjustment_poses)
    bundle_adjustment_locations = \
        BundleAdjustment.from_poses_to_locations(bundle_adjustment_poses)
    print("finished bundle adjustment")

    pose_graph = PoseGraph(bundle_adjustment)
    print("starting loop closure")
    loop_closure = LoopClosure(db, pose_graph, threshold_close=500,
                               threshold_inliers_rel=0.4)
    full_cov_before_LC = np.array(pose_graph.get_covraince_all_poses())
    loops_dict = loop_closure.find_loops(OUTPUT_DIR, plot=False)
    full_cov_after_LC = np.array(pose_graph.get_covraince_all_poses())
    loop_closure_poses = pose_graph.get_global_keyframes_poses()  # dict
    loop_closure_mats = from_gtsam_poses_to_world_coordinates_mats(loop_closure_poses)
    loop_closure_locations = BundleAdjustment.from_poses_to_locations(loop_closure_poses)

    # part 3 - Performance Analysis
    present_tracking_statistics(db)

    utils.connectivity_graph(db, OUTPUT_DIR)

    utils.tracks_length_histogram(db, OUTPUT_DIR)

    show_locations_trajectory(ground_truth_locations, pnp_locations, bundle_adjustment_locations,
                              loop_closure_locations)

    # mean factor error for each window
    plot_bundle_error_before_after_opt \
        (keyframes_list, mean_factor_error_before, mean_factor_error_after,
         'mean factor error')

    # median projection error for each window
    plot_bundle_error_before_after_opt \
        (keyframes_list, median_projection_error_before, median_projection_error_after,
         'median projection error')

    pnp_distances, pnp_links_num_per_distance = PnP_median_projection_error_per_distance()  # distance from last frame of track were we computed triangulation

    ba_distances, ba_links_num_per_distance = BA_median_projection_error_per_distance()  # distance from last frame of min(track, window) were we computed triangulation

    plot_links_num_per_distance(pnp_distances, pnp_links_num_per_distance, ba_distances, ba_links_num_per_distance)

    ground_truth_mats_of_keyframes = ground_truth_mats[keyframes_list]
    ground_truth_locations_of_keyframes = ground_truth_locations[keyframes_list]

    plot_location_error_along_axes(ground_truth_locations, pnp_locations, 'PnP')
    plot_angle_error(ground_truth_mats, pnp_mats, 'PnP')

    plot_location_error_along_axes(ground_truth_locations_of_keyframes, bundle_adjustment_locations,
                                   'BA')  # pose graph without LC
    plot_angle_error(ground_truth_mats_of_keyframes, bundle_adjustment_mats, 'BA')

    plot_location_error_along_axes(ground_truth_locations_of_keyframes, loop_closure_locations, 'LC')
    plot_angle_error(ground_truth_mats_of_keyframes, loop_closure_mats, 'LC')

    loop_closure_statistics()

    # uncertainty size vs keyframes
    covs_angle_before, covs_location_before = slice_cov_mat_to_localization_angle(full_cov_before_LC)
    covs_angle_after, covs_location_after = slice_cov_mat_to_localization_angle(full_cov_after_LC)
    plot_uncertainty(covs_angle_before, covs_angle_after, keyframes_list, "Angle")
    plot_uncertainty(covs_location_before, covs_location_after, keyframes_list, "Location")

    # relative pnp estimation error of location and angle

    pnp_errors_location, pnp_errors_angle, pnp_sequences_list = [], [], []
    for seq_len in [100, 300, 800]:
        location_errors, angle_errors, sequences = \
            relative_estimation_error_in_sequences(pnp_mats, ground_truth_mats, ground_truth_locations, seq_len)
        pnp_errors_location.append(location_errors)
        pnp_errors_angle.append(angle_errors)
        pnp_sequences_list.append(sequences)
    plot_relative_estimation_error_in_sequences('PnP', 'location', [100, 300, 800], pnp_errors_location, pnp_sequences_list)
    plot_relative_estimation_error_in_sequences('PnP', 'angle', [100, 300, 800],  pnp_errors_angle, pnp_sequences_list)

    ba_errors_location, ba_errors_angle, ba_sequences_list = [], [], []
    for seq_len in [100, 300, 800]:
        location_errors, angle_errors, sequences = \
            relative_estimation_error_in_sequences(bundle_adjustment_mats, ground_truth_mats_of_keyframes,
                                                   ground_truth_locations_of_keyframes,
                                                   int(seq_len / bundle_adjustment.get_window_len()) + 1)
        ba_errors_location.append(location_errors)
        ba_errors_angle.append(angle_errors)
        ba_sequences_list.append(sequences)
    plot_relative_estimation_error_in_sequences('BA', 'location', [100, 300, 800], ba_errors_location, ba_sequences_list)
    plot_relative_estimation_error_in_sequences('BA', 'angle', [100, 300, 800], ba_errors_angle, ba_sequences_list)