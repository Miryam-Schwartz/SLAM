import gtsam
import numpy as np

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleWindow import CAMERA_SYMBOL, POINT_SYMBOL
from Bundle.BundleWindow import BundleWindow
from VAN_ex.code.ex7 import OUTPUT_DIR


def subtract_lines(a, b):
    s = np.maximum(a.max(0) + 1, b.max(0) + 1)
    return a[~np.isin(a.dot(s), b.dot(s))]


class LoopClosure:
    def __init__(self, db, pose_graph, threshold_close, threshold_inliers_num):
        self._db = db
        self._pose_graph = pose_graph
        self._threshold_close = threshold_close
        self._threshold_inliers_num = threshold_inliers_num

    def detect_possible_candidates(self, n_keyframe, prev_keyframes):
        # 7.1
        # assume this func return list of keyframes that are candidates of n_keyframe
        possible_cantidates = []
        n_pose = self._pose_graph.get_pose_obj(n_keyframe)
        for i_keyframe in prev_keyframes:
            i_pose = self._pose_graph.get_pose_obj(i_keyframe)
            rel_cov = self._pose_graph.get_relative_covariance(i_keyframe, n_keyframe)
            rel_pose = n_pose.between(i_pose)       # todo: make sure
            rotation = rel_pose.rotation()
            rel_pose =\
                np.array([rotation.pitch(), rotation.roll(), rotation.yaw(), rel_pose.x(), rel_pose.y(), rel_pose.z()])
            mahalanubis_dist = rel_pose.T @ rel_cov @ rel_pose
            if mahalanubis_dist < self._threshold_close:
                possible_cantidates.append(i_keyframe)
        return possible_cantidates

    def consensus_matching(self, i_keyframe, n_keyframe):
        # 7.2
        n_des_left = self._db.get_frame_obj(n_keyframe).get_des_left()
        i_des_left = self._db.get_frame_obj(i_keyframe).get_des_left()
        matches = utils.find_matches(i_des_left, n_des_left)
        matches = [(match.queryIdx, match.trainIdx) for match in matches]
        extrinsic_camera_mat_i_to_n_left, inliers_matches, inliers_percentage = \
            self._db.RANSAC(i_keyframe, n_keyframe, matches)
        return inliers_matches, subtract_lines(np.array(matches), inliers_matches), extrinsic_camera_mat_i_to_n_left

    def find_loops(self):
        loops_dict = dict()
        keyframes_list = self._pose_graph.get_keyframes()
        interval_len = int(len(keyframes_list) / 6)
        for j, n_keyframe in enumerate(keyframes_list):
            prev_keyframes = [kf for kf in keyframes_list if kf < n_keyframe]
            candidates = self.detect_possible_candidates(n_keyframe, prev_keyframes)
            for i_keyframe in candidates:
                inliers_matches, outliers_matches, extrinsic_camera_mat_i_to_n_left = \
                    self.consensus_matching(i_keyframe, n_keyframe)
                if len(inliers_matches) > self._threshold_inliers_num:
                    loops_dict[(i_keyframe, n_keyframe)] = (inliers_matches, outliers_matches)
                    pose_i_to_n, cov_i_to_n = \
                        self.bundle_for_two_frames(i_keyframe, n_keyframe, extrinsic_camera_mat_i_to_n_left,
                                                   inliers_matches)
                    # 7.4
                    self._pose_graph.add_factor(i_keyframe, n_keyframe, pose_i_to_n, cov_i_to_n)
                    self._pose_graph.optimize()
            if j % interval_len == 0:
                self._pose_graph.show(f"{OUTPUT_DIR}pose_graph_after_iteration_{j}.png", j)
        return loops_dict

    def bundle_for_two_frames(self, i_keyframe, n_keyframe, extrinsic_i_to_n, matches):
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        # add prior
        sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [0.03, 0.003, 0.03])
        prior_cov = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        prior_factor = gtsam.PriorFactorPose3(gtsam.symbol(CAMERA_SYMBOL, i_keyframe), gtsam.Pose3(), prior_cov)
        graph.add(prior_factor)

        # add pose i and n to values
        values.insert(gtsam.symbol(CAMERA_SYMBOL, i_keyframe), gtsam.Pose3())
        inv_r, inv_t = utils.invert_extrinsic_matrix(extrinsic_i_to_n[:, :3], extrinsic_i_to_n[:, 3])
        n_pos3 = gtsam.Pose3(np.hstack((inv_r, inv_t.reshape(3, 1))))
        values.insert(gtsam.symbol(CAMERA_SYMBOL, n_keyframe), n_pos3)

        # add factors between each point (from matches) to i and n poses
        n_stereo_cam = gtsam.StereoCamera(n_pos3, BundleWindow.K)
        for i, match in enumerate(matches):
            # match[0] is kp idx in frame i, match[1] is kp idx in frame n
            n_pt_2d = utils.get_stereo_point2(self._db, n_keyframe, match[1])

            pt_3d = n_stereo_cam.backproject(n_pt_2d)
            if pt_3d[2] <= 0 or pt_3d[2] >= 200:
                continue

            values.insert(gtsam.symbol(POINT_SYMBOL, i), pt_3d)
            cov = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
            n_factor = gtsam.GenericStereoFactor3D \
                (n_pt_2d, cov, gtsam.symbol(CAMERA_SYMBOL, n_keyframe), gtsam.symbol(POINT_SYMBOL, i),
                 BundleWindow.K)

            i_pt_2d = utils.get_stereo_point2(self._db, i_keyframe, match[0])
            i_factor = gtsam.GenericStereoFactor3D \
                (i_pt_2d, cov, gtsam.symbol(CAMERA_SYMBOL, i_keyframe), gtsam.symbol(POINT_SYMBOL, i),
                 BundleWindow.K)
            graph.add(n_factor)
            graph.add(i_factor)

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
        optimized_values = optimizer.optimize()
        optimized_pose_i_to_n = optimized_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, n_keyframe))

        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol(CAMERA_SYMBOL, i_keyframe))
        keys.append(gtsam.symbol(CAMERA_SYMBOL, n_keyframe))
        marginals = gtsam.Marginals(graph, optimized_values)
        sliced_inform_mat = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
        cov_i_to_n = np.linalg.inv(sliced_inform_mat)
        return optimized_pose_i_to_n, cov_i_to_n
