import numpy as np

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleAdjustment import BundleAdjustment
from VAN_ex.code.DB.DataBase import DataBase
from VAN_ex.code.LoopClosure import LoopClosure
from VAN_ex.code.PoseGraph import PoseGraph


def from_idx_to_pixels(db, i_keyframe, n_keyframe, matches):
    i_pixels, n_pixels = np.empty(len(matches), 2), np.empty(len(matches), 2)
    i_frame_obj = db.get_frame_obj(i_keyframe)
    n_frame_obj = db.get_frame_obj(n_keyframe)
    for i, match in enumerate(matches):
        i_pixels[i] = np.array(i_frame_obj.get_feature_pixels()[0])
        n_pixels[i] = np.array(n_frame_obj.get_feature_pixels()[0])
    return i_pixels, n_pixels


def plot_match_results(i_keyframe, n_keyframe, i_inliers, n_inliers, i_outliers, n_outliers):
    pass



if __name__ == '__main__':
    db = DataBase()
    db.read_database(utils.DB_PATH)
    bundle_adjustment = BundleAdjustment(2560, 20, db)
    bundle_adjustment.optimize_all_windows()
    pose_graph = PoseGraph(bundle_adjustment)

    loop_closure = LoopClosure(db, pose_graph, 1, 50)  # todo:we are not sure about threshold + add default values
    loops_dict = loop_closure.find_loops()
    pose_graph.optimize()

    # 7.5
    print(f"Number of successful loop closures that were detected: {len(loops_dict)}")

    i_keyframe, n_keyframe = list(loops_dict.keys())[0]
    inliers_matches, outliers_matches = loops_dict[(i_keyframe, n_keyframe)]
    i_inliers, n_inliers = from_idx_to_pixels(db, i_keyframe, n_keyframe, inliers_matches)
    i_outliers, n_outliers = from_idx_to_pixels(db, i_keyframe, n_keyframe, outliers_matches)
    plot_match_results(i_keyframe, n_keyframe, i_inliers, n_inliers, i_outliers, n_outliers)



