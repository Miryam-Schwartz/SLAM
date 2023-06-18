import gtsam
import numpy as np

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleWindow import CAMERA_SYMBOL, POINT_SYMBOL
from Bundle.BundleWindow import BundleWindow


def setdiff_nd_positivenums(a, b):
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
        # assume this func return list of keyframes that are candidates of n_keyframe (7.1)


        pass

    def consensus_matching(self, i_keyframe, n_keyframe):
        # 7.2
        n_des_left = self._db.get_frame_obj(n_keyframe).get_des_left()
        i_des_left = self._db.get_frame_obj(i_keyframe).get_des_left()
        matches = utils.find_matches(i_des_left, n_des_left)
        matches = [(match.queryIdx, match.trainIdx) for match in matches]
        extrinsic_camera_mat_i_to_n_left, inliers_matches, inliers_percentage = \
            self._db.RANSAC(i_keyframe, n_keyframe, matches)
        return inliers_matches, setdiff_nd_positivenums(np.array(matches), inliers_matches), \
               extrinsic_camera_mat_i_to_n_left

    def find_loops(self):
        counter = 0
        loops_dict = dict()
        keyframes_list = self._pose_graph.get_keyframes_list()
        for n_keyframe in keyframes_list:
            prev_keyframes = [kf for kf in keyframes_list if kf < n_keyframe]
            candidates = self.detect_possible_candidates(n_keyframe, prev_keyframes)
            for i_keyframe in candidates:
                inliers_matches, outliers_matches, extrinsic_camera_mat_i_to_n_left = \
                    self.consensus_matching(i_keyframe, n_keyframe)
                if len(inliers_matches) > self._threshold_inliers_num:
                    counter += 1
                    loops_dict[(i_keyframe, n_keyframe)] = (inliers_matches, outliers_matches)
                    pose_i_to_n, cov_i_to_n = \
                        self.bundle_for_two_frames(i_keyframe, n_keyframe, extrinsic_camera_mat_i_to_n_left,
                                                   inliers_matches)
                    # 7.4
                    self._pose_graph.add_loop_factor(i_keyframe, n_keyframe, pose_i_to_n, cov_i_to_n)
        return loops_dict


    def bundle_for_two_frames(self, i_keyframe, n_keyframe, extrinsic_i_to_n, matches):
        sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [0.03, 0.003, 0.03])
        prior_cov = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        prior_factor = gtsam.PriorFactorPose3(gtsam.symbol(CAMERA_SYMBOL, i_keyframe), gtsam.Pose3(), prior_cov)
        values = gtsam.Values()
        graph = gtsam.NonlinearFactorGraph()
        graph.add(prior_factor)
        values.insert(gtsam.symbol(CAMERA_SYMBOL, i_keyframe), gtsam.Pose3())
        inv_r, inv_t = utils.invert_extrinsic_matrix(extrinsic_i_to_n[:, :3], extrinsic_i_to_n[:, 3])
        n_pos3 = gtsam.Pose3(np.hstack((inv_r, inv_t.reshape(3, 1))))
        values.insert(gtsam.symbol(CAMERA_SYMBOL, n_keyframe), n_pos3)
        n_stereo_cam = gtsam.StereoCamera(n_pos3, BundleWindow.K)
        for i, match in enumerate(matches):
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
