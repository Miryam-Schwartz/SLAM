import os

import numpy as np
from matplotlib import pyplot as plt

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleAdjustment import BundleAdjustment
from VAN_ex.code.DB.DataBase import DataBase
from VAN_ex.code.LoopClosure import LoopClosure
from VAN_ex.code.PoseGraph import PoseGraph

OUTPUT_DIR = 'results/ex7/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def from_idx_to_pixels(db, i_keyframe, n_keyframe, matches):
    i_pixels, n_pixels = np.empty(len(matches), 2), np.empty(len(matches), 2)
    i_frame_obj = db.get_frame_obj(i_keyframe)
    n_frame_obj = db.get_frame_obj(n_keyframe)
    for i, match in enumerate(matches):
        i_pixels[i] = np.array(i_frame_obj.get_feature_pixels()[0])
        n_pixels[i] = np.array(n_frame_obj.get_feature_pixels()[0])
    return i_pixels, n_pixels


def plot_supporters(i_keyframe, n_keyframe, inliers_i, inliers_n, outliers_i, outliers_n):
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1

    fig.suptitle(f'inliers and outliers between keyframe {i_keyframe} to {n_keyframe}')

    # n_keyframe camera
    fig.add_subplot(rows, cols, 2)
    n_img_l, _ = utils.read_images(n_keyframe)
    plt.imshow(n_img_l, cmap='gray')
    plt.title(f"keyframe {n_keyframe}")

    plt.scatter(outliers_n[:, 0], outliers_n[:, 2], s=3, color="red")
    plt.scatter(inliers_n[:, 0], inliers_n[:, 2], s=3, color="blue")

    # i_keyframe camera
    fig.add_subplot(rows, cols, 1)
    i_img_l, _ = utils.read_images(i_keyframe)
    plt.imshow(i_img_l, cmap='gray')
    plt.title(f"keyframe {i_keyframe}")

    plt.scatter(outliers_i[:, 0], outliers_i[:, 2], s=3, color="red")
    plt.scatter(inliers_i[:, 0], inliers_i[:, 2], s=3, color="blue")

    fig.savefig(f'{OUTPUT_DIR}inliers_and_outliers_{n_keyframe}_to_{i_keyframe}.png')
    plt.close(fig)


def plot_match():
    i_keyframe, n_keyframe = list(loops_dict.keys())[0]
    inliers_i, inliers_n, outliers_i, outliers_n = loops_dict[(i_keyframe, n_keyframe)]
    plot_supporters(i_keyframe, n_keyframe, inliers_i, inliers_n, outliers_i, outliers_n)


def plot_location_uncertainty(cov_list_before, cov_list_after, keyframes):
    det_cov_before = [np.sqrt(np.linalg.det(before_cov)) for before_cov in cov_list_before]
    det_cov_after = [np.sqrt(np.linalg.det(after_cov)) for after_cov in cov_list_after]
    keyframes = np.array(keyframes)
    det_cov_before = np.array(det_cov_before)
    det_cov_after = np.array(det_cov_after)

    fig = plt.figure()
    plt.title("Location uncertainty before and after Loop Closure")
    plt.plot(keyframes, det_cov_before, label="Before")
    plt.plot(keyframes, det_cov_after, label="After")
    plt.yscale('log')
    plt.ylabel("uncertainty -sqrt covariance determinate")
    plt.xlabel("Keyframe")
    plt.legend()

    fig.savefig(f'{OUTPUT_DIR}uncertainty_before_after_loop_closure.png')
    plt.close(fig)


if __name__ == '__main__':
    db = DataBase()
    db.read_database(utils.DB_PATH)
    print("finished read data")
    bundle_adjustment = BundleAdjustment(utils.FRAMES_NUM, 20, db)
    bundle_adjustment.optimize_all_windows()
    print("finished bundle adjustment")
    pose_graph = PoseGraph(bundle_adjustment)
    global_poses = pose_graph.get_global_keyframes_poses()
    global_locations = BundleAdjustment.from_poses_to_locations(global_poses)

    print("starting loop closure")
    loop_closure = LoopClosure(db, pose_graph, threshold_close=500,
                               threshold_inliers_rel=0.4)
    cov_list_before = pose_graph.get_covraince_all_poses()
    loops_dict = loop_closure.find_loops(OUTPUT_DIR)
    cov_list_after = pose_graph.get_covraince_all_poses()

    # 7.5
    print(f"Number of successful loop closures that were detected: {len(loops_dict)}")
    print("loops that were added: ", loops_dict.keys())

    plot_match()

    global_poses_optimized = pose_graph.get_global_keyframes_poses()
    global_locations_optimized = BundleAdjustment.from_poses_to_locations(global_poses_optimized)
    ground_truth_matrices = utils.read_ground_truth_matrices()
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_matrices)

    utils.keyframes_localization_error(pose_graph, global_locations_optimized, ground_truth_locations, global_locations,
                                       f'{OUTPUT_DIR}location_error_before_and_after.png')

    fig, ax = plt.subplots(figsize=(19.2, 14.4))
    ax.scatter(x=ground_truth_locations[:, 0], y=ground_truth_locations[:, 2],
               label='Ground Truth', s=2, alpha=0.4, c='tab:blue')
    ax.scatter(x=global_locations[:, 0], y=global_locations[:, 2],
               label='Bundle Adjustment', s=10, alpha=0.7, c='tab:green')
    ax.scatter(x=global_locations_optimized[:, 0], y=global_locations_optimized[:, 2],
               label='Loop Closure', s=30, alpha=0.7, c='tab:red')
    ax.legend(fontsize='20')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(f'{OUTPUT_DIR}localization.png')

    plot_location_uncertainty(cov_list_before, cov_list_after, bundle_adjustment.get_keyframes())
