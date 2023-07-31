import statistics
import gtsam
import numpy as np

from VAN_ex.code.Bundle.BundleWindow import BundleWindow


class BundleAdjustment:
    """
        A class used to represent bundle adjustment algorithm. An instance of of this class containing collection of
        bundle windows objects.
        ...

        Attributes
        ----------
        _window_len  : int
            length of each bundle window
        _frames_num  : int
            number of frames in total
        _db  : DB
            database object that saves the information about frames and points.
        _bundle_windows  : dict
            containing all bundle windows that covers all the trajectory. Key is int - first keyframe id of the window.
            Value is bundle window object.
        """

    def __init__(self, frames_num, window_len, db):
        self._window_len = window_len
        self._frames_num = frames_num
        self._db = db
        self._bundle_windows = self._init_bundle_windows()  # key = first keyframe, val = window obj

    @staticmethod
    def from_poses_to_locations(poses):
        """
        convert gtsam poses to 3d location of the keyframe camera.
        :param poses: dict, key is keyframe id, value is gtsam Pose object that represents the position that corresponds
        to the keyframe. The pose should be in global coordinates.
        :return: np array in shape (x, 3) while x is number of keyframes.
        represents locations (x,y,z) of all keyframes in poses param. The location is in global coordinates.
        """
        # assume poses are in global coordinates. poses is dict
        locations = np.empty((len(poses), 3))
        for i, pose in enumerate(poses.values()):
            locations[i] = np.array([pose.x(), pose.y(), pose.z()])
        return locations

    def _init_bundle_windows(self):
        """
        create bundle windows in lengths of self._window_len.
        :return: dict of bundle windows. Key is first keyframe, value is bundle window object.
        """
        bundle_windows = dict()
        for first_keyframe in range(0, self._frames_num, self._window_len - 1):
            last_key_frame = min(first_keyframe + self._window_len - 1, self._frames_num - 1)
            bundle_windows[first_keyframe] = BundleWindow(self._db, first_keyframe, last_key_frame)
        return bundle_windows

    def optimize_all_windows(self):
        """
        optimize each bundle window
        """
        for window in self._bundle_windows.values():
            window.optimize()

    # ================ Getters ================ #

    def get_keyframes(self):
        """
        :return: np array of all Keyframe ids.
        """
        keyframes = list(self._bundle_windows.keys())
        keyframes.append(self._frames_num - 1)
        return np.array(keyframes)

    def get_last_window(self):
        """
        :return: bundle window object of last window.
        """
        max_frame_key = max(self._bundle_windows.keys())
        return self._bundle_windows[max_frame_key]

    def get_first_window(self):
        """
        :return: bundle window object of of window.
        """
        return self._bundle_windows[0]

    def get_window_len(self):
        """
        :return: single window length (number of frames per bundle window)
        """
        return self._window_len

    def get_global_3d_points(self, global_poses):
        """
        Go over all bundle windows, for each window, go over all 3d points and transform them to world coordinates
        :param global_poses: dictionary, key is frame id, value is pose of corresponding frame in global coordinates
        :return: np array in shape (x,3) of all 3d points in global coordinates
        """
        points_list = []
        for window in self._bundle_windows.values():
            pose3 = global_poses[window.get_first_keyframe_id()]
            points_of_window = window.get_points()
            for pt in points_of_window:
                pt_global = pose3.transformFrom(pt)
                points_list.append(pt_global)
        return np.array(points_list)

    def get_global_keyframes_poses(self):
        """
        return poses of all keyframe in global coordinates. To do this, concatenate pose of previous keyframe
        with current relative pose.
        :return: dictionary. Key is keyframe id, value is gtsam pose of the keyframe, in global coordinates.
        """
        poses = dict()
        poses[0] = gtsam.Pose3()
        for first_keyframe, window in self._bundle_windows.items():
            last_keyframe = window.get_last_keyframe_id()
            global_pose_last_keyframe = poses[first_keyframe] * window.get_frame_pose3(last_keyframe)
            poses[last_keyframe] = global_pose_last_keyframe
        return poses

    def get_relative_motion_and_covariance(self):
        """
        extract relative motion and covariance of each window
        :return: 2 dictionaries. Key is tuple of 2 ints in shape (first_keyframe, last_keyframe) in both.
        In first dictionary value gtsam pose of relative motion of corresponding window,
        in second dictionary, value is relative covariance np matrix of that window.
        """
        motions, covs = dict(), dict()
        for first_keyframe, window in self._bundle_windows.items():
            last_key_frame = window.get_last_keyframe_id()
            motions[(first_keyframe, last_key_frame)] = window.get_frame_pose3(last_key_frame)
            covs[(first_keyframe, last_key_frame)] = window.get_covariance()
        return motions, covs

    # ================ Errors ================ #

    def get_mean_factor_error_for_all_windows(self):
        """
        :return: one dimensional np array with mean factor error for each single bundle window.
        """
        mean_factor_error = np.empty(len(self._bundle_windows))
        for i, window in enumerate(self._bundle_windows.values()):
            mean_factor_error[i] = window.get_mean_factors_error()
        return mean_factor_error

    def get_median_projection_error_for_all_windows(self):
        """
        :return: one dimensional np array with median projection error for each single bundle window.
        """
        median_projection_error = np.empty(len(self._bundle_windows))
        for i, window in enumerate(self._bundle_windows.values()):
            median_projection_error[i] = window.get_median_projection_error()
        return median_projection_error

    def get_median_projection_error_per_distance(self):
        """
        calculate median projection error per distance from last frame od window. means, for all links that their
        distance between frame to last keyframe of bundle is d, takes the median of reprojection error.
        :return: distances - np array of all the distances between frame to last frame of bundle
                median_proj_err_left - dict. key is distance from last frame, value is list of projection errors in left
                median_proj_err_right - same as above for right
                links_num_per_distance - np array such that in the i-th index is the number of links that are far i from
                last frame.
        """
        projection_error_left = dict()  # key = distance from first frame, value = list of projection errors
        projection_error_right = dict()
        for window in self._bundle_windows.values():
            window.calc_projection_error_per_distance(projection_error_left, projection_error_right)
        distances = np.array(sorted(projection_error_right.keys()))
        median_proj_err_left = np.array([statistics.median(projection_error_left[d]) for d in distances])
        median_proj_err_right = np.array([statistics.median(projection_error_right[d]) for d in distances])
        links_num_per_distance = np.array([len(projection_error_right[d]) for d in distances])
        return distances, median_proj_err_left, median_proj_err_right, links_num_per_distance
