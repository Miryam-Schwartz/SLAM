import os

OUTPUT_DIR = 'results/run_over_other_data/'
os.makedirs(OUTPUT_DIR, exist_ok=True)



def run_over_other_data():
    pass
    # ground_truth_mats = np.array(utils.read_ground_truth_matrices(utils.GROUND_TRUTH_PATH))
    # ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_mats)
    #
    # db = DataBase()
    #
    # db.fill_database(utils.FRAMES_NUM)
    # db.save_database(utils.DB_PATH)
    #
    # # db.read_database(utils.DB_PATH)
    #
    # pnp_mats = db.get_left_camera_mats_in_world_coordinates()
    # pnp_locations = utils.calculate_ground_truth_locations_from_matrices(pnp_mats)
    #
    # bundle_adjustment = BundleAdjustment(utils.FRAMES_NUM, 20, db)
    # bundle_adjustment.optimize_all_windows()
    # bundle_adjustment_poses = bundle_adjustment.get_global_keyframes_poses()  # dict
    # bundle_adjustment_mats = from_gtsam_poses_to_world_coordinates_mats(bundle_adjustment_poses)
    # bundle_adjustment_locations = \
    #     BundleAdjustment.from_poses_to_locations(bundle_adjustment_poses)
    #
    # pose_graph = PoseGraph(bundle_adjustment)
    # loop_closure = LoopClosure(db, pose_graph, threshold_close=500,
    #                            threshold_inliers_rel=0.4)
    # full_cov_before_LC = np.array(pose_graph.get_covraince_all_poses())
    # loops_dict = loop_closure.find_loops('', plot=False)
    # full_cov_after_LC = np.array(pose_graph.get_covraince_all_poses())
    # loop_closure_poses = pose_graph.get_global_keyframes_poses()  # dict
    # loop_closure_mats = from_gtsam_poses_to_world_coordinates_mats(loop_closure_poses)
    # loop_closure_locations = BundleAdjustment.from_poses_to_locations(loop_closure_poses)
    #
    # show_locations_trajectory(ground_truth_locations, pnp_locations, bundle_adjustment_locations,
    #                           loop_closure_locations)
    #
    # present_tracking_statistics(db)
    #
    # keyframes_list = bundle_adjustment.get_keyframes()
    # ground_truth_mats_of_keyframes = ground_truth_mats[keyframes_list]
    # ground_truth_locations_of_keyframes = ground_truth_locations[keyframes_list]
    #
    # plot_location_error_along_axes(ground_truth_locations_of_keyframes, loop_closure_locations, 'LC', keyframes_list)
    # plot_angle_error(ground_truth_mats_of_keyframes, loop_closure_mats, 'LC')
    #
    # # uncertainty size vs keyframes
    # covs_angle_before, covs_location_before = slice_cov_mat_to_localization_angle(full_cov_before_LC)
    # covs_angle_after, covs_location_after = slice_cov_mat_to_localization_angle(full_cov_after_LC)
    # plot_uncertainty(covs_angle_before, covs_angle_after, keyframes_list, "Angle")
    # plot_uncertainty(covs_location_before, covs_location_after, keyframes_list, "Location")

if __name__ == '__main__':
    run_over_other_data()