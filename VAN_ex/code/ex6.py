import os

import numpy as np
from gtsam.utils import plot
from matplotlib import pyplot as plt

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleAdjustment import BundleAdjustment
from VAN_ex.code.DB.DataBase import DataBase
from VAN_ex.code.PoseGraph import PoseGraph
from VAN_ex.code.ex4 import crop_img

OUTPUT_DIR = 'results/ex6/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def show_feature_track(track, db):
    """
    Choose random track in len of num_frames at least, create and save an image, that shows the feature
    the first 10 frames of the track.
    :param db: database
    :param num_frames: track length at least to choose
    """
    fig, grid = plt.subplots(track.get_track_len(), 2)
    fig.set_figwidth(4)
    fig.set_figheight(12)
    fig.suptitle(f"Track {track.get_id()}, with length of {track.get_track_len()} frames")
    plt.rcParams["font.size"] = "10"
    plt.subplots_adjust(wspace=-0.55, hspace=0.1)
    grid[0, 0].set_title("Left")
    grid[0, 1].set_title("Right")
    i = 0
    for frame_id, kp_idx in track.get_frames_dict().items():
        pixel_left, pixel_right = db.get_frame_obj(frame_id).get_feature_pixels(kp_idx)
        img_left, img_right = utils.read_images(frame_id)
        img_left, x_left, y_left = crop_img(img_left, pixel_left[0], pixel_left[1])
        img_right, x_right, y_right = crop_img(img_right, pixel_right[0], pixel_right[1])
        grid[i, 0].axes.xaxis.set_visible(False)
        grid[i, 0].axes.yaxis.set_label_text(f"frame: {frame_id}")
        grid[i, 0].set_yticklabels([])
        grid[i, 0].imshow(img_left, cmap="gray")
        grid[i, 0].scatter(x_left, y_left, s=10, c="r")
        grid[i, 1].axes.xaxis.set_visible(False)
        grid[i, 1].imshow(img_right, cmap="gray")
        grid[i, 1].scatter(x_right, y_right, s=10, c="r")
        grid[i, 1].set_yticklabels([])
        i = i + 1
    fig.savefig(f'{OUTPUT_DIR}track.png')

def ex6_run():
    db = DataBase()
    db.read_database(utils.DB_PATH)

    # track = db.get_track_obj(350160)
    # print(track.get_frames_dict())
    # show_feature_track(track, db)


    bundle_adjustment = BundleAdjustment(2560, 20, db)
    bundle_adjustment.optimize_all_windows()


    # first_window = bundle_adjustment.get_first_window()
    # marginals = first_window.get_marginals()
    # # print(marginals)
    # result = first_window.get_current_values()
    # plot.plot_trajectory(0, result, marginals=marginals, scale=1, title="Covariance poses for first bundle",
    #                      save_file=f"{OUTPUT_DIR}Poses rel_covs.png"
    #                      # , d2_view=False
    #                      ) #todo: fix graph
    #
    # first_camera = first_window.get_frame_pose3(first_window.get_first_keyframe_id())
    # second_camera = first_window.get_frame_pose3(first_window.get_last_keyframe_id())
    # relative_pose = first_camera.between(second_camera)
    #
    # print("Relative pose between first two keyframes:")
    # print(first_window.get_frame_pose3(first_window.get_last_keyframe_id()))
    # print("relative pose - with between method")
    # print(relative_pose)
    # print("Covariance:")
    # print(first_window.get_covariance())

    # 6.2
    # track = db.get_track_obj(350160)
    # show_feature_track(track, db)

    pose_graph = PoseGraph(bundle_adjustment)
    global_poses = pose_graph.get_global_keyframes_poses()
    locations = BundleAdjustment.from_poses_to_locations(global_poses)
    utils.show_localization(locations, None, f"{OUTPUT_DIR}initial_locations.png",
                            "Pose Graph Localization - Before Optimization")
    factor_error_before = pose_graph.total_factor_error()
    pose_graph.optimize()
    factor_error_after = pose_graph.total_factor_error()
    global_poses_after_optimize = pose_graph.get_global_keyframes_poses()
    global_locations_after_optimize = BundleAdjustment.from_poses_to_locations(global_poses_after_optimize)
    utils.show_localization(global_locations_after_optimize, None, f"{OUTPUT_DIR}locations_after_optimize.png",
                            "Pose Graph Localization - After Optimization")
    print("Factor error BEFORE optimization: ", factor_error_before)
    print("Factor error AFTER optimization: ", factor_error_after)

    marginals = pose_graph.get_marginals()
    # print(marginals)
    result = pose_graph.get_current_values()
    plot.plot_trajectory(0, result, marginals=marginals, scale=1, title="Locations with marginal covariance",
                         save_file=f"{OUTPUT_DIR}locatization with covariance.png"
                         # , d2_view=False
                         ) #todo: fix graph



if __name__ == '__main__':
    ex6_run()