import gtsam
import numpy as np
import RANSAC

from VAN_ex.code import utils
from VAN_ex.code.Bundle.BundleWindow import CAMERA_SYMBOL, POINT_SYMBOL
from Bundle.BundleWindow import BundleWindow


def extract_matches(matches, kp_left_first, kp_right_first, kp_left_second, kp_right_second):
    """
    :param matches: np array in shape (x,2) in each row there is a single match between first to second frame,
    means the i-th match is matches[i], and matches[i][0] is index in kp_left_first that corsponds to kp_left_second
    in index matches[i][1].
    :param kp_left_first: list of key-points on left image of first frame. each keypoint is from shape (x,y) represents
    pixel on image.
    :param kp_right_first: same as above, for right image of first frame
    :param kp_left_second: same as above, for left image of first second
    :param kp_right_second: same as above, for right image of second frame
    :return: two np arrays, each on shape (x,3), while x is number of matches between first to second. the arrays are
    correlated, means first_pixels[i] is matched to second_pixels[i].
    each row is from the shape (x_left, x_right, y).
    """
    first_pixels = np.empty((matches.shape[0], 3))
    second_pixels = np.empty((matches.shape[0], 3))
    for i, match in enumerate(matches):
        idx_kp_first = match[0]
        idx_kp_second = match[1]
        first_pixels[i] = convert_to_shared_pixel(kp_left_first[idx_kp_first], kp_right_first[idx_kp_first])
        second_pixels[i] = convert_to_shared_pixel(kp_left_second[idx_kp_second], kp_right_second[idx_kp_second])
    return first_pixels, second_pixels


def convert_to_shared_pixel(kp_left, kp_right):
    x_left, y_left = kp_left
    x_right, y_right = kp_right
    return np.array([x_left, x_right, (y_left + y_right)/2])


class LoopClosure:
    """
        A class used to represent loop closure algorithm. Loop closure contains pose graph, and used to find and add
        loops in pose graph. loop is an area where the vehicle traveled and returned to.
        ...

        Attributes
        ----------
        _db : BundleAdjustment
           database object that saves the information about frames and points.
        _pose_graph :   PoseGraph
            skeleton graph of poses and factors between poses
        _threshold_close    :   int
            choose keyframe i to be candidate for loop of keyframe n, if the distance between them in mahalanubis
            distance is smaller than _threshold_close
        _threshold_inliers_rel  :   float
            add loop between i keyframe to n keyframe, if relation between inliers to all matches is bigger than
            _threshold_inliers_rel
           """
    def __init__(self, db, pose_graph, threshold_close=500, threshold_inliers_rel=0.4):
        self._db = db
        self._pose_graph = pose_graph
        self._threshold_close = threshold_close
        self._threshold_inliers_rel = threshold_inliers_rel

    def detect_possible_candidates(self, n_keyframe, prev_keyframes):   # 7.1
        """
        find from all prev_keyframes, keyframes that their mahalanubis distance from n_keyframe is not too big
        :param n_keyframe: keyframe id
        :param prev_keyframes: list of all keyframes that are between 0 to n_keyframe
        :return: list of keyframes that are possible candidates to be looped with n_keyframe
        """
        possible_cantidates = []
        n_pose = self._pose_graph.get_pose_obj(n_keyframe)
        for i_keyframe in prev_keyframes:
            # ignore too close keyframes
            if i_keyframe + 100 >= n_keyframe:
                continue
            i_pose = self._pose_graph.get_pose_obj(i_keyframe)
            rel_cov = self._pose_graph.get_relative_covariance(i_keyframe, n_keyframe)
            rel_pose = n_pose.between(i_pose)
            rotation = rel_pose.rotation()
            rel_pose =\
                np.array([rotation.pitch(), rotation.roll(), rotation.yaw(), rel_pose.x(), rel_pose.y(), rel_pose.z()])
            mahalanubis_dist = (rel_pose.T @ np.linalg.inv(rel_cov) @ rel_pose) ** 0.5
            if mahalanubis_dist < self._threshold_close:
                possible_cantidates.append(i_keyframe)
        return possible_cantidates

    def consensus_matching(self, i_keyframe, n_keyframe):         # 7.2
        """
        Run RANSAC between i_keyframe to n_keyframe for finding the motion matrix between them, inliers and outliers
        :param i_keyframe:
        :param n_keyframe:
        :return: extrinsic_camera_mat_second_frame_left- np matrix in shape (3,4), represents extrinsic camera matrix of
        second frame (left) in coordinates of first frame (left).
        and inliers_i, inliers_n, outliers_i, outliers_n
        """
        extrinsic_camera_mat_second_frame_left, inliers_matches, outliers_matches, \
            kp_left_first, kp_right_first, kp_left_second, kp_right_second = RANSAC.RANSAC(i_keyframe, n_keyframe)
        if kp_right_first is None:
            return None, None, None, None, None
        inliers_i, inliers_n =\
            extract_matches(inliers_matches, kp_left_first, kp_right_first, kp_left_second, kp_right_second)
        outliers_i, outliers_n =\
            extract_matches(outliers_matches, kp_left_first, kp_right_first, kp_left_second, kp_right_second)
        return extrinsic_camera_mat_second_frame_left, inliers_i, inliers_n, outliers_i, outliers_n

    def find_loops(self, output_dir, plot=True):
        """
        loop closure al gorithm to find loops in pose graph.
        Go over all keyframes, for each keyframe, find possible candidates by using the shortest path and mahalanubis
        distance, for each of the candidates, run RANSAC with pnp and filter ineligible candidates by using inliers
        percentage. Finally, add a loop in the appropriate place and optimize pose graph.
        :param output_dir: directory path to save plots there
        :param plot: boolean, specifies whether to output a graph
        :return: dictionary, key is tuple of two keyframe ids (i,n)of new loops that were added, while value is
        inliers_i, inliers_n, outliers_i, outliers_n
        """
        loops_dict = dict()
        keyframes_list = self._pose_graph.get_keyframes()
        interval_len = int(len(keyframes_list) / 6)
        for j, n_keyframe in enumerate(keyframes_list):
            prev_keyframes = [kf for kf in keyframes_list if kf < n_keyframe]
            candidates = self.detect_possible_candidates(n_keyframe, prev_keyframes)
            for i_keyframe in candidates:
                extrinsic_camera_mat_i_to_n_left, inliers_i, inliers_n, outliers_i, outliers_n = \
                    self.consensus_matching(i_keyframe, n_keyframe)
                if extrinsic_camera_mat_i_to_n_left is None:
                    continue
                if inliers_i.shape[0] / (inliers_i.shape[0] + outliers_i.shape[0]) > self._threshold_inliers_rel:
                    loops_dict[(i_keyframe, n_keyframe)] = (inliers_i, inliers_n, outliers_i, outliers_n)
                    pose_i_to_n, cov_i_to_n = \
                        LoopClosure.bundle_for_two_frames(i_keyframe, n_keyframe, extrinsic_camera_mat_i_to_n_left,
                                                   inliers_i, inliers_n)
                    # 7.4
                    self._pose_graph.add_factor(i_keyframe, n_keyframe, pose_i_to_n, cov_i_to_n)
                    self._pose_graph.optimize()
            if j % interval_len == 0 and plot:
                self._pose_graph.show(f"{output_dir}pose_graph_after_keyframe_{n_keyframe}.png", n_keyframe)
        return loops_dict

    @staticmethod
    def bundle_for_two_frames(i_keyframe, n_keyframe, extrinsic_i_to_n, inliers_i, inliers_n):
        """
        bundle adjustment algorithm version for only two frames, without all frames between them.
        :param i_keyframe:
        :param n_keyframe:
        :param extrinsic_i_to_n:
        :param inliers_i:
        :param inliers_n:
        :return: relative pose and covariance between i to n keyframe
        """

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
        for j in range(inliers_i.shape[0]):
            n_pt_2d = gtsam.StereoPoint2(inliers_n[j][0], inliers_n[j][1], inliers_n[j][2])

            pt_3d = n_stereo_cam.backproject(n_pt_2d)
            if pt_3d[2] <= 0 or pt_3d[2] >= 200:
                continue

            values.insert(gtsam.symbol(POINT_SYMBOL, j), pt_3d)
            cov = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
            n_factor = gtsam.GenericStereoFactor3D \
                (n_pt_2d, cov, gtsam.symbol(CAMERA_SYMBOL, n_keyframe), gtsam.symbol(POINT_SYMBOL, j),
                 BundleWindow.K)

            i_pt_2d = gtsam.StereoPoint2(inliers_i[j][0], inliers_i[j][1], inliers_i[j][2])
            i_factor = gtsam.GenericStereoFactor3D \
                (i_pt_2d, cov, gtsam.symbol(CAMERA_SYMBOL, i_keyframe), gtsam.symbol(POINT_SYMBOL, j),
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
