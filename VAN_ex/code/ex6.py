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
    bundle_adjustment = BundleAdjustment(2560, 20, db)
    bundle_adjustment.optimize_all_windows()
    first_window = bundle_adjustment.get_first_window()
    marginals = first_window.get_marginals()
    # print(marginals)
    result = first_window.get_current_values()
    plot.plot_trajectory(0, result, marginals=marginals, scale=1, title="Covariance poses for first bundle",
                         save_file=f"{OUTPUT_DIR}Poses rel_covs.png"
                         # , d2_view=False
                         ) #todo: fix graph

    first_camera = first_window.get_frame_pose3(first_window.get_first_keyframe_id())
    second_camera = first_window.get_frame_pose3(first_window.get_last_keyframe_id())
    relative_pose = first_camera.between(second_camera)

    print("Relative pose between first two keyframes:")
    print(first_window.get_frame_pose3(first_window.get_last_keyframe_id()))
    print("relative pose - with between method")
    print(relative_pose)
    print("Covariance:")
    print(first_window.get_covariance())

    # 6.2
    pose_graph = PoseGraph(bundle_adjustment)
    global_poses = pose_graph.get_global_keyframes_poses()
    locations = BundleAdjustment.from_poses_to_locations(global_poses)
    utils.show_localization(locations, [[], [], []], f"{OUTPUT_DIR}initial_poses.png")
    pose_graph.optimize()
    global_poses_after_optimize = pose_graph.get_global_keyframes_poses()
    locations = BundleAdjustment.from_poses_to_locations(global_poses)
    utils.show_localization()

if __name__ == '__main__':
    ex6_run()