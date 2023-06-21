import gtsam
import numpy as np
import networkx as nx

from gtsam.utils import plot

from VAN_ex.code.Bundle.BundleWindow import CAMERA_SYMBOL


def cov_weight_square_det(cov_mat):
    return np.square(np.linalg.det(cov_mat))


class PoseGraph:
    def __init__(self, bundle_adjustment):
        self._bundle_adjustment = bundle_adjustment
        self._initial_estimate = self._init_initial_estimate()
        self._current_values = self._initial_estimate
        self._shortest_path_graph = self._init_shortest_path_graph()
        self._factor_covariances = dict()
        self._factors = self._init_factors()
        sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [1e-1, 1, 1e-1])
        cov = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        self._prior_factor = \
            gtsam.PriorFactorPose3(gtsam.symbol(CAMERA_SYMBOL, 0), gtsam.Pose3(), cov)
        self._graph = self._init_graph()
        self._optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial_estimate)

    def _init_shortest_path_graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.get_keyframes())
        return G

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
            self.add_factor(first_keyframe_id, second_keyframe_id, motion, relative_covs[keyframes_tuple])
            # noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_covs[keyframes_tuple])
            # factor = gtsam.BetweenFactorPose3 \
            #     (gtsam.symbol(CAMERA_SYMBOL, first_keyframe_id), gtsam.symbol(CAMERA_SYMBOL, second_keyframe_id),
            #      motion, noise_model)
            # factors[keyframes_tuple] = factor
            # self._shortest_path_graph.add_edge(first_keyframe_id, second_keyframe_id,
            #                                    weight=cov_weight_square_det(relative_covs[keyframes_tuple]))
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

    def total_factor_error(self):  # returns error of current values
        error = 0
        for factor in self._factors.values():
            error += factor.error(self._current_values)
        return error

    def get_marginals(self):
        return gtsam.Marginals(self._graph, self._current_values)

    def get_current_values(self):
        return self._current_values

    def get_relative_covariance(self, i_keyframe, n_keyframe):
        # find the shortest path and sum the covariance
        path = nx.shortest_path(self._shortest_path_graph, source=i_keyframe, target=n_keyframe, weight='weight')
        sum_cov = np.zeros(6, 6)
        for i in range(len(path)-1):
            sum_cov += self._factor_covariances[(path[i], path[i+1])]
        return sum_cov

    def get_keyframes(self):
        return self._bundle_adjustment.get_keyframes()
        # keyframes_lst = set()
        # for keyframes_tuple in self._factors.keys():
        #     set.add(keyframes_tuple[0])
        #     set.add(keyframes_tuple[1])
        # return list(keyframes_lst)

    def add_factor(self, i_keyframe, n_keyframe, pose_i_to_n, cov):
        noise_model = gtsam.noiseModel.Gaussian.Covariance(cov)
        factor = gtsam.BetweenFactorPose3 \
            (gtsam.symbol(CAMERA_SYMBOL, i_keyframe), gtsam.symbol(CAMERA_SYMBOL, n_keyframe),
             pose_i_to_n, noise_model)
        self._factors[(i_keyframe, n_keyframe)] = factor
        self._factor_covariances[(i_keyframe, n_keyframe)] = cov
        self._graph.add(factor)
        self._shortest_path_graph.add_edge(i_keyframe, n_keyframe, weight=cov_weight_square_det(cov))

    def show(self, output_path, j):
        marginals = self.get_marginals()
        result = self.get_current_values()
        plot.plot_trajectory(0, result, marginals=marginals, scale=1, title=f"Loop closure after iteration {j}",
                             save_file=output_path, d2_view=True)

    # def get_covariance_n_given_i(self, i_keyframe, n_keyframe):
    #     keys = gtsam.KeyVector()
    #     keys.append(gtsam.symbol(CAMERA_SYMBOL, i_keyframe))
    #     keys.append(gtsam.symbol(CAMERA_SYMBOL, n_keyframe))
    #     marginals = self.get_marginals()
    #     sliced_inform_mat = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    #     return np.linalg.inv(sliced_inform_mat)

    def get_pose_obj(self, keyframe_id):
        return self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, keyframe_id))