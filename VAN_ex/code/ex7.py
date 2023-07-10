import os

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
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


# def plot_match_results(keyframe_id, inliers, outliers):
#     img_l, _ = utils.read_images(keyframe_id)
#     fig = px.imshow(img_l, color_continuous_scale='gray')
#     fig.update_traces(dict(showscale=False, coloraxis=None, colorscale='gray'))
#     fig.add_trace(
#         go.Scatter(x=inliers[:, 0], y=inliers[:, 1],
#                    marker=dict(color='red', size=5), name='inliers'))
#     fig.add_trace(
#         go.Scatter(x=outliers[:, 0], y=outliers[:, 1],
#                    marker=dict(color='blue', size=5), name='outliers'))
#     fig.update_layout(title=dict(text=f"Keyframe id :{keyframe_id}"))
#     fig.write_html(f'{OUTPUT_DIR}inliers_and_outliers_{keyframe_id}.html')


def plot_supporters(i_keyframe, n_keyframe, inliers_i, inliers_n, outliers_i, outliers_n):
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 2, 1

    perc_title = ""
    # if left1_matches_coor is not None:
    #     inliers_perc = format(inliers_perc, ".2f")
    #     perc_title = f"supporters({inliers_perc}% inliers, {INLIER_COLOR})"

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


def localization_error():
    global_poses_optimized = pose_graph.get_global_keyframes_poses()
    global_locations_optimized = BundleAdjustment.from_poses_to_locations(global_poses_optimized)
    ground_truth_matrices = utils.read_ground_truth_matrices()
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_matrices)
    utils.keyframes_localization_error(pose_graph, global_locations_optimized, ground_truth_locations, global_locations,
                                       f'{OUTPUT_DIR}location_error_before_and_after.png')


if __name__ == '__main__':
    db = DataBase()
    db.read_database(utils.DB_PATH)
    print("finished read data")
    bundle_adjustment = BundleAdjustment(2560, 20, db)
    bundle_adjustment.optimize_all_windows()
    print("finished bundle adjustment")
    pose_graph = PoseGraph(bundle_adjustment)
    global_poses = pose_graph.get_global_keyframes_poses()
    global_locations = BundleAdjustment.from_poses_to_locations(global_poses)

    print("starting loop closure")
    loop_closure = LoopClosure(db, pose_graph, threshold_close=500, threshold_inliers_rel=0.4)  # todo:we are not sure about threshold + add default values
    loops_dict = loop_closure.find_loops(OUTPUT_DIR)

    # 7.5
    print(f"Number of successful loop closures that were detected: {len(loops_dict)}")
    print("loops that were added: ", loops_dict.keys())

    plot_match()

    localization_error()



