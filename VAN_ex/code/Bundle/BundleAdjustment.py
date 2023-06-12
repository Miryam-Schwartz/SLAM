import gtsam
import numpy as np

# from BundleWindow import BundleWindow
from VAN_ex.code.Bundle.BundleWindow import BundleWindow


class BundleAdjustment:
    def __init__(self, frames_num, window_len, db):
        self._window_len = window_len
        self._frames_num = frames_num
        self._db = db
        self._bundle_windows = self._init_bundle_windows()  # key = first keyframe, val = window obj

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
            last_keyframe = min(first_keyframe + self._window_len - 1, self._frames_num - 1)
            global_pose_last_keyframe = poses[first_keyframe] * window.get_frame_pose3(last_keyframe)
            poses[last_keyframe] = global_pose_last_keyframe
        return poses

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
