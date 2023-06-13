import gtsam

from VAN_ex.code.Bundle.BundleWindow import CAMERA_SYMBOL


class PoseGraph:
    def __init__(self, bundle_adjustment):
        self._bundle_adjustment = bundle_adjustment
        self._initial_estimate = self._init_initial_estimate()
        self._current_values = self._initial_estimate
        self._factors = self._init_factors()
        self._prior_factor =\
            gtsam.PriorFactorPose3(gtsam.symbol(CAMERA_SYMBOL, 0), gtsam.Pose3(), gtsam.noiseModel.Unit.Create(6))
        self._graph = self._init_graph()
        self._optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial_estimate)

    def _init_initial_estimate(self):
        initial_estimate = gtsam.Values()
        global_keyframes_poses = self._bundle_adjustment.get_global_keyframes_poses()
        for keyframe, pose in global_keyframes_poses.items():
            initial_estimate.insert(gtsam.symbol(CAMERA_SYMBOL, keyframe), pose)
        return initial_estimate

    def _init_factors(self):
        relative_motions, relative_covs = self._bundle_adjustment.get_relative_motion_and_covariance()
        factors = dict()
        for keyframes_tuple, motion in relative_motions.items():
            first_keyframe_id, second_keyframe_id = keyframes_tuple
            noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_covs[keyframes_tuple])
            factor = gtsam.BetweenFactorPose3\
                (gtsam.symbol(CAMERA_SYMBOL, first_keyframe_id), gtsam.symbol(CAMERA_SYMBOL, second_keyframe_id),
                 motion, noise_model)
            factors[keyframes_tuple] = factor
        return factors

    def _init_graph(self):
        graph = gtsam.NonlinearFactorGraph()
        for factor in self._factors.values():
            graph.add(factor)
        graph.add(self._prior_factor)
        return graph

    def optimize(self):
        self._current_values = self._optimizer.optimize()
        return self._current_values  # returns values object

    def get_global_keyframes_poses(self):
        poses = dict()  # key = keyframe, val = pose
        poses[0] = self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, 0))
        for keyframes_tuple in self._factors:
            first_keyframe_id, second_keyframe_id = keyframes_tuple
            global_pose_last_keyframe = self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, second_keyframe_id))
            poses[second_keyframe_id] = global_pose_last_keyframe
        return poses