import os

from gtsam.utils import plot

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleAdjustment import BundleAdjustment
from VAN_ex.code.DB.DataBase import DataBase
from VAN_ex.code.PoseGraph import PoseGraph

OUTPUT_DIR = 'results/ex6/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ex6_run():
    db = DataBase()
    db.read_database(utils.DB_PATH)

    bundle_adjustment = BundleAdjustment(utils.FRAMES_NUM, 20, db)
    bundle_adjustment.optimize_all_windows()

    q6_1(bundle_adjustment)

    q6_2(bundle_adjustment)


def q6_2(bundle_adjustment):
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
    result = pose_graph.get_current_values()
    plot.plot_trajectory(0, result, marginals=marginals, scale=1, title="Locations with marginal covariance",
                         save_file=f"{OUTPUT_DIR}locatization with covariance.png"
                         , d2_view=True
                         )


def q6_1(bundle_adjustment):
    first_window = bundle_adjustment.get_first_window()
    marginals = first_window.get_marginals()
    result = first_window.get_current_values()
    plot.plot_trajectory(0, result, marginals=marginals, scale=1, title="Covariance poses for first bundle",
                         save_file=f"{OUTPUT_DIR}Poses rel_covs.png"
                         , d2_view=False
                         )
    first_camera = first_window.get_frame_pose3(first_window.get_first_keyframe_id())
    second_camera = first_window.get_frame_pose3(first_window.get_last_keyframe_id())
    relative_pose = first_camera.between(second_camera)
    print("Relative pose between first two keyframes:")
    print(first_window.get_frame_pose3(first_window.get_last_keyframe_id()))
    print("relative pose - with between method")
    print(relative_pose)
    print("Covariance:")
    print(first_window.get_covariance())


if __name__ == '__main__':
    ex6_run()
