import statistics
import gtsam
import numpy as np

from VAN_ex.code.Bundle.BundleWindow import BundleWindow


class BundleAdjustment:
    def __init__(self, frames_num, window_len, db):
        self._window_len = window_len
        self._frames_num = frames_num
        self._db = db
        if window_len != 0:
            self._bundle_windows = self._init_bundle_windows()  # key = first keyframe, val = window obj
        else:
            self._bundle_windows = self._init_bundle_windows_by_median_track_len()

    @staticmethod
    def from_poses_to_locations(poses):
        # assume poses are in global coordinates. poses is dict
        locations = np.empty((len(poses), 3))
        for i, pose in enumerate(poses.values()):
            locations[i] = np.array([pose.x(), pose.y(), pose.z()])
        return locations

    def _init_bundle_windows(self):
        bundle_windows = dict()
        for first_keyframe in range(0, self._frames_num, self._window_len-1):
            last_key_frame = min(first_keyframe + self._window_len - 1, self._frames_num - 1)
            bundle_windows[first_keyframe] = BundleWindow(self._db, first_keyframe, last_key_frame)
        return bundle_windows

    def _init_bundle_windows_by_median_track_len(self):
        bundle_windows = dict()
        first_keyframe = 0
        last_keyframe = int(self._get_median_track_len_from_frame(first_keyframe))
        while last_keyframe < self._frames_num - 1:
            bundle_windows[first_keyframe] = BundleWindow(self._db, first_keyframe, last_keyframe)
            first_keyframe = last_keyframe
            last_keyframe = first_keyframe + int(self._get_median_track_len_from_frame(first_keyframe))

        last_keyframe = min(last_keyframe, self._frames_num - 1)
        bundle_windows[first_keyframe] = BundleWindow(self._db, first_keyframe, last_keyframe)
        return bundle_windows

    def get_keyframes(self):
        keyframes = list(self._bundle_windows.keys())
        keyframes.append(self._frames_num - 1)
        return np.array(keyframes)

    def optimize_all_windows(self):
        for window in self._bundle_windows.values():
            # print(f'optimize window {window.get_first_keyframe_id()} - {window.get_last_keyframe_id()}')
            window.optimize()

    def get_global_keyframes_poses(self):
        poses = dict()  # key = keyframe, val = pose
        poses[0] = gtsam.Pose3()
        for first_keyframe, window in self._bundle_windows.items():
            # todo
            last_keyframe = window.get_last_keyframe_id()
            global_pose_last_keyframe = poses[first_keyframe] * window.get_frame_pose3(last_keyframe)
            poses[last_keyframe] = global_pose_last_keyframe
            print(f'pose {last_keyframe}: {global_pose_last_keyframe}')
        return poses

    def get_relative_motion_and_covariance(self):
        motions, covs = dict(), dict()  # key = (first_keyframe, last_keyframe), val = relative motion / relative cov mat
        for first_keyframe, window in self._bundle_windows.items():
            last_key_frame = window.get_last_keyframe_id()
            motions[(first_keyframe, last_key_frame)] = window.get_frame_pose3(last_key_frame)
            covs[(first_keyframe, last_key_frame)] = window.get_covariance()
        return motions, covs

    def get_global_3d_points(self, global_poses):
        # assume gloal_poses is dict: key = frame_id, val = pose of frame in global coordinates
        points_list = []
        for window in self._bundle_windows.values():
            pose3 = global_poses[window.get_first_keyframe_id()]
            points_of_window = window.get_points()
            for pt in points_of_window:
                pt_global = pose3.transformFrom(pt)
                points_list.append(pt_global)
        return np.array(points_list)

    def get_last_window(self):
        max_frame_key = max(self._bundle_windows.keys())
        return self._bundle_windows[max_frame_key]

    def get_first_window(self):
        return self._bundle_windows[0]

    def get_mean_factor_error_for_all_windows(self):
        mean_factor_error = np.empty(len(self._bundle_windows))
        for i, window in enumerate(self._bundle_windows.values()):
            mean_factor_error[i] = window.get_mean_factors_error()
        return mean_factor_error

    def get_median_projection_error_for_all_windows(self):
        median_projection_error = np.empty(len(self._bundle_windows))
        for i, window in enumerate(self._bundle_windows.values()):
            median_projection_error[i] = window.get_median_projection_error()
        return median_projection_error

    def get_median_projection_error_per_distance(self):
        projection_error_left = dict()       # key = distance from first frame, value = list of projection errors
        projection_error_right = dict()
        for window in self._bundle_windows.values():
            window.calc_projection_error_per_distance(projection_error_left, projection_error_right)
        distances = np.array(sorted(projection_error_right.keys()))
        median_proj_err_left = np.array([statistics.median(projection_error_left[d]) for d in distances])
        median_proj_err_right = np.array([statistics.median(projection_error_right[d]) for d in distances])
        links_num_per_distance = np.array([len(projection_error_right[d]) for d in distances])
        return distances, median_proj_err_left, median_proj_err_right, links_num_per_distance

    def get_window_len(self):
        return self._window_len

    def _get_median_track_len_from_frame(self, frame_id):
        frame_obj = self._db.get_frame_obj(frame_id)
        track_lens = frame_obj.get_lens_outgoing_tracks()
        return statistics.median(track_lens)








