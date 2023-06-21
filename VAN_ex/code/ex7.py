import os

import numpy as np
import plotly.express as px
import plotly.graph_objs as go

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


def plot_match_results(keyframe_id, inliers, outliers):
    img_l, _ = utils.read_images(keyframe_id)
    fig = px.imshow(img_l, color_continuous_scale='gray')
    fig.update_traces(dict(showscale=False, coloraxis=None, colorscale='gray'))
    fig.add_trace(
        go.Scatter(x=inliers[:, 0], y=inliers[:, 1],
                   marker=dict(color='red', size=5), name='inliers'))
    fig.add_trace(
        go.Scatter(x=outliers[:, 0], y=outliers[:, 1],
                   marker=dict(color='blue', size=5), name='outliers'))
    fig.update_layout(title=dict(text=f"Keyframe id :{keyframe_id}"))
    fig.write_html(f'{OUTPUT_DIR}inliers_and_outliers_{keyframe_id}.html')


def plot_match():
    i_keyframe, n_keyframe = list(loops_dict.keys())[0]
    inliers_matches, outliers_matches = loops_dict[(i_keyframe, n_keyframe)]
    i_inliers, n_inliers = from_idx_to_pixels(db, i_keyframe, n_keyframe, inliers_matches)
    i_outliers, n_outliers = from_idx_to_pixels(db, i_keyframe, n_keyframe, outliers_matches)
    plot_match_results(i_keyframe, i_inliers, i_outliers)
    plot_match_results(n_keyframe, n_inliers, n_outliers)


def localization_error():
    global_poses_optimized = pose_graph.get_global_keyframes_poses()
    global_locations_optimized = BundleAdjustment.from_poses_to_locations(global_poses_optimized)
    ground_truth_matrices = utils.read_ground_truth_matrices()
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_matrices)
    utils.keyframes_localization_error(pose_graph, global_locations_optimized, ground_truth_locations, global_locations,
                                       f'{OUTPUT_DIR}location_error_before_and_after')


if __name__ == '__main__':
    db = DataBase()
    db.read_database(utils.DB_PATH)
    bundle_adjustment = BundleAdjustment(2560, 20, db)
    bundle_adjustment.optimize_all_windows()
    pose_graph = PoseGraph(bundle_adjustment)
    global_poses = pose_graph.get_global_keyframes_poses()
    global_locations = BundleAdjustment.from_poses_to_locations(global_poses)

    loop_closure = LoopClosure(db, pose_graph, 1, 50)  # todo:we are not sure about threshold + add default values
    loops_dict = loop_closure.find_loops()
    pose_graph.optimize() # todo: we should call it after each n iteration?

    # 7.5
    print(f"Number of successful loop closures that were detected: {len(loops_dict)}")

    plot_match()

    localization_error()



