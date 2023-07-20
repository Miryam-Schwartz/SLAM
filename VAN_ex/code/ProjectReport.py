import os
import random
import statistics

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
    utils.inliers_percentage_graph(db, output_path=OUTPUT_DIR)


def show_locations_trajectory(ground_truth_locations, pnp_locations, bundle_adjustment_locations,
                              loop_closure_locations):
    fig, ax = plt.subplots(figsize=(19.2, 14.4))
    ax.plot(x=ground_truth_locations[:, 0], y=ground_truth_locations[:, 2],
            label='Ground Truth', s=2, alpha=0.4, c='tab:blue')
    ax.plot(x=pnp_locations[:, 0], y=pnp_locations[:, 2],
            label='PnP', s=10, alpha=0.7, c='tab:yellow')
    ax.plot(x=bundle_adjustment_locations[:, 0], y=bundle_adjustment_locations[:, 2],
            label='Bundle Adjustment', s=10, alpha=0.7, c='tab:green')
    ax.plot(x=loop_closure_locations[:, 0], y=loop_closure_locations[:, 2],
            label='Loop Closure', s=30, alpha=0.7, c='tab:red')
    ax.legend(fontsize='20')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(f'{OUTPUT_DIR}localization.png')


def plot_bundle_error_before_after_opt(keyframes_list, mean_factor_error_before, mean_factor_error_after, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=keyframes_list, y=mean_factor_error_before, mode='lines', name='before optimization error'))
    fig.add_trace(
        go.Scatter(x=keyframes_list, y=mean_factor_error_after, mode='lines', name='after optimization error'))
    fig.update_layout(title=title, xaxis_title='Keyframe id', yaxis_title=title)
    fig.write_image(f'{OUTPUT_DIR}{title}.png')


def PnP_median_projection_error_per_distance_from_last_frame():
    random_track_ids = random.sample(range(db.get_tracks_number()), 500)
    left_camera_mats_in_world_coordinates = db.get_left_camera_mats_in_world_coordinates()
    # key = distance from last frame of track, value = list of projection errors (from different tracks)
    right_projection_err = dict()
    left_projection_err = dict()
    for track_id in random_track_ids:
        track = db.get_track_obj(track_id)
        frames_dict = track.get_frames_dict()
        last_frame_id, last_kp_idx = track.get_last_frame_id_and_kp_idx()
        pt_3d = db.get_frame_obj(last_frame_id).get_3d_point(last_kp_idx)  # in coordinates of last frame
        last_frame_mat = left_camera_mats_in_world_coordinates[last_frame_id]
        r, t = last_frame_mat[:, :3], last_frame_mat[:, 3]
        world_coor_3d_pt = r.T @ (pt_3d - t)
        for frame_id, kp_idx in frames_dict.items():
            if frame_id == last_frame_id:
                break
            frame_obj = db.get_frame_obj(frame_id)
            extrinsic_left = left_camera_mats_in_world_coordinates[frame_id]
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

    distances = np.array(list(right_projection_err.keys()))
    median_proj_err_left = np.array([statistics.median(left_projection_err[d]) for d in distances])
    median_proj_err_right = np.array([statistics.median(right_projection_err[d]) for d in distances])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=distances, y=median_proj_err_left, mode='lines', name='left'))
    fig.add_trace(go.Scatter(x=distances, y=median_proj_err_right, mode='lines', name='right'))
    fig.update_layout(title='PnP - median projection error vs track length',
                      xaxis_title='distance from last frame', yaxis_title='projection error')
    fig.write_image(f'{OUTPUT_DIR}PnP_median_projection_error_vs_track_length.png')


def BA_median_projection_error_per_distance_from_first_frame():
    pass


if __name__ == '__main__':
    ground_truth_mats = utils.read_ground_truth_matrices(utils.GROUND_TRUTH_PATH)
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_mats)

    db = DataBase()
    db.read_database(utils.DB_PATH)
    pnp_locations = db.get_camera_locations()
    print("finished read data")

    bundle_adjustment = BundleAdjustment(2560, 20, db)
    keyframes_list = bundle_adjustment.get_keyframes()
    mean_factor_error_before = bundle_adjustment.get_mean_factor_error_for_all_windows()
    median_projection_error_before = bundle_adjustment.get_median_projection_error_for_all_windows()

    bundle_adjustment.optimize_all_windows()

    mean_factor_error_after = bundle_adjustment.get_mean_factor_error_for_all_windows()
    median_projection_error_after = bundle_adjustment.get_median_projection_error_for_all_windows()
    bundle_adjustment_locations = \
        BundleAdjustment.from_poses_to_locations(bundle_adjustment.get_global_keyframes_poses())
    print("finished bundle adjustment")

    pose_graph = PoseGraph(bundle_adjustment)
    print("starting loop closure")
    loop_closure = LoopClosure(db, pose_graph, threshold_close=500,
                               threshold_inliers_rel=0.4)
    loops_dict = loop_closure.find_loops(OUTPUT_DIR)
    loop_closure_locations = BundleAdjustment.from_poses_to_locations(pose_graph.get_global_keyframes_poses())

    # part 3 - Performance Analysis
    present_tracking_statistics(db)

    utils.connectivity_graph(db, OUTPUT_DIR)

    utils.tracks_length_histogram(db, OUTPUT_DIR)

    show_locations_trajectory(ground_truth_locations, pnp_locations, bundle_adjustment_locations,
                              loop_closure_locations)

    # mean factor error for each window
    plot_bundle_error_before_after_opt\
        (keyframes_list, mean_factor_error_before, mean_factor_error_after, 'mean factor error')

    # median projection error for each window
    # todo
    # plot_bundle_error_before_after_opt \
    #     (keyframes_list, median_projection_error_before, median_projection_error_after, 'median projection error')

    PnP_median_projection_error_per_distance_from_last_frame()

    BA_median_projection_error_per_distance_from_first_frame()


