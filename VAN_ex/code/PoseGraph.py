import gtsam
import numpy as np
import networkx as nx
from gtsam.utils import plot

from VAN_ex.code.Bundle.BundleWindow import CAMERA_SYMBOL


def cov_weight_square_det(cov_mat):
    """
    :param cov_mat: covariance matrix (np array)
    :return: square determinant of matrix
    """
    return np.square(np.linalg.det(cov_mat))


class PoseGraph:
    """
        A class used to represent Pose graph. A Pose graph is a skeleton graph that is created from bundle adjustment
        graph. to create pose graph we drop unnecessary data (frames that are not keyframes and points).
        This leaves us with a graph contains only keyframes and factors between them.
        ...

        Attributes
        ----------
        _bundle_adjustment : BundleAdjustment
            bundle adjustment instance
        _initial_estimate   :   gtsam.Values
            like a dictionary object. associate between keyframe to its initial guess position
            (taken from bundle adjustment).
        _current_values :   gtsam.Values
            associate between keyframe to its optimized value. before the optimization, initial estimate is saved here.
        _shortest_path_graph    :   nx.DiGraph
            simple graph with only nodes, edges and weights on edges. helps for finding the shortest path in pose graph.
        _factor_covariances :   dict
            covariance of factors. Key is tuple (i_keyframe_id, n_keyframe_id). Value is np matrix in shape (6,6),
            represents covariance of the factor/ motion between i-th keyframe to n-th keyframe.
        _graph  :   gtsam.NonlinearFactorGraph
            gtsam object that holds all factors between keyframes
        _factors    :   dict
            Key is tuple (i_keyframe_id, n_keyframe_id). value is factor object
        _prior_factor   :   gtsam.PriorFactorPose3
            factor for first keyframe (0,0,0)
        _optimizer  :   gtsam.LevenbergMarquardtOptimizer
            gtsam optimizer from type Levenberg-Marquardt
        """

    def __init__(self, bundle_adjustment):
        self._bundle_adjustment = bundle_adjustment
        self._initial_estimate = self._init_initial_estimate()
        self._current_values = self._initial_estimate
        self._shortest_path_graph = self._init_shortest_path_graph()
        self._factor_covariances = dict()
        self._graph = gtsam.NonlinearFactorGraph()
        self._factors = dict()
        self._init_factors()
        sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [10e-1, 10, 10e-1])
        cov = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        self._prior_factor = \
            gtsam.PriorFactorPose3(gtsam.symbol(CAMERA_SYMBOL, 0), gtsam.Pose3(), cov)
        self._graph.add(self._prior_factor)
        self._optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial_estimate)

    # ================ Initialization ================ #

    def _init_initial_estimate(self):
        """
        Go over all poses of keyframes from bundle adjustment algorithm and fill initial_estimate with B.A.
        optimized values.
        :return: gtsam.Values object initialized with values of global pose for each keyframe.
        """
        initial_estimate = gtsam.Values()
        global_keyframes_poses = self._bundle_adjustment.get_global_keyframes_poses()
        for keyframe, pose in global_keyframes_poses.items():
            initial_estimate.insert(gtsam.symbol(CAMERA_SYMBOL, keyframe), pose)
        return initial_estimate

    def _init_shortest_path_graph(self):
        """
        :return: nx.DiGraph that its nodes are the keyframes.
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.get_keyframes())
        return G

    def _init_factors(self):
        """
        Go over all the relative motions between consecutive keyframes and add factors to the pose graph.
        each factor saves motion (gtsam.Pose3) and covariance (np matrix in shape (6,6)) that were gotten from B.A.
        """
        relative_motions, relative_covs = self._bundle_adjustment.get_relative_motion_and_covariance()
        for keyframes_tuple, motion in relative_motions.items():
            first_keyframe_id, second_keyframe_id = keyframes_tuple
            self.add_factor(first_keyframe_id, second_keyframe_id, motion, relative_covs[keyframes_tuple])

    def add_factor(self, i_keyframe, n_keyframe, pose_i_to_n, cov):
        """
        add "between" factor to pose graph. a factor is information about motion between two keyframes.
        While adding an edge to pose graph, update factors dictionary and add an edge to _shortest_path_graph.
        the weight of this edge is square determinant of covariance matrix.
        :param i_keyframe:  the motion is from i-th keyframe to n-th keyframe
        :param n_keyframe:  same as above
        :param pose_i_to_n: gtsam.Pose3, represents the motion.
        :param cov: covariance of factor (np matrix in shape (6,6))
        """
        noise_model = gtsam.noiseModel.Gaussian.Covariance(cov)
        factor = gtsam.BetweenFactorPose3 \
            (gtsam.symbol(CAMERA_SYMBOL, i_keyframe), gtsam.symbol(CAMERA_SYMBOL, n_keyframe),
             pose_i_to_n, noise_model)
        self._factors[(i_keyframe, n_keyframe)] = factor
        self._factor_covariances[(i_keyframe, n_keyframe)] = cov
        self._graph.add(factor)
        self._shortest_path_graph.add_edge(i_keyframe, n_keyframe, weight=cov_weight_square_det(cov))

    # ================ Getters ================ #

    def get_pose_obj(self, keyframe_id):
        """
        :param keyframe_id:
        :return: gtsam.Pose3 that corresponding to keyframe id.
        """
        return self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, keyframe_id))

    def get_keyframes(self):
        """
        :return: np array of all Keyframe ids.
        """
        return self._bundle_adjustment.get_keyframes()

    def get_global_keyframes_poses(self):
        """
        get poses of keyframes from _current_values
        (initial estimate at first, and after optimization - optimized values)
        :return: poses dictionary. Key is keyframe id, value is gtsam.Pose3
        """
        poses = dict()  # key = keyframe, val = pose
        poses[0] = self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, 0))
        for keyframes_tuple in self._factors:
            first_keyframe_id, second_keyframe_id = keyframes_tuple
            poses[second_keyframe_id] = self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, second_keyframe_id))
        return poses

    def get_marginals(self):
        """
        :return: gtsam.Marginals of pose graph
        """
        return gtsam.Marginals(self._graph, self._current_values)

    def get_covariance_all_poses(self):
        """
        :return: list of np matrices in shape (6,6), represents covariance matrix of each keyframe.
        """
        marginals = self.get_marginals()
        keyframes = self._bundle_adjustment.get_keyframes()
        cov_list = []
        for id_keyframe in keyframes:
            cur_symbol = gtsam.symbol(CAMERA_SYMBOL, id_keyframe)
            cur_cov = marginals.marginalCovariance(cur_symbol)
            cov_list.append(cur_cov)
        return cov_list

    def get_current_values(self):
        """
        :return: gtsam.Values, saves current pose of each keyframe. At first saves initial estimation,
        and after optimization saves optimized values.
        """
        return self._current_values

    def get_relative_covariance(self, i_keyframe, n_keyframe):
        """
        find the shortest path from i-th keyframe to n-th keyframe and sum the covariances along the path
        :param i_keyframe:
        :param n_keyframe:
        :return: estimated relative covariance between i-th keyframe to n-th keyframe
        """
        path = nx.shortest_path(self._shortest_path_graph, source=i_keyframe, target=n_keyframe, weight='weight')
        sum_cov = np.zeros((6, 6))
        for i in range(len(path) - 1):
            sum_cov += self._factor_covariances[(path[i], path[i + 1])]
        return sum_cov

    def optimize(self):
        """
        :return: gtsam.Values after optimization (each time we call optimize, we create newly optimizer that gets the
        previous optimized values as initial estimation.
        """
        self._initial_estimate = self._current_values
        self._optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial_estimate)
        self._current_values = self._optimizer.optimize()
        return self._current_values

    def total_factor_error(self):  # returns error of current values
        """
        :return: sum of error of all factors in current values
        """
        error = 0
        for factor in self._factors.values():
            error += factor.error(self._current_values)
        return error

    def show(self, output_path, iteration):
        """
        plot current view of pose graph from bird eye, with covariances of each keyframe on it.
        :param output_path: where to save the graph file + file name
        :param iteration: loop closure iteration
        """
        marginals = self.get_marginals()
        result = self.get_current_values()
        plot.plot_trajectory(0, result, marginals=marginals, scale=1, title=f"Loop closure after iteration {iteration}",
                             save_file=output_path, d2_view=True)
