import random

import gtsam
import numpy as np
from VAN_ex.code import utils

CAMERA_SYMBOL = 'c'
POINT_SYMBOL = 'q'




class BundleWindow:
    k, m_left, m_right = utils.read_cameras()
    indentation_right_cam = m_right[0][3]
    K = gtsam.Cal3_S2Stereo(k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], -indentation_right_cam)

    def __init__(self, db, first_keyframe_id, last_keyframe_id):
        self._db = db
        self._first_keyframe_id = first_keyframe_id
        self._last_keyframe_id = last_keyframe_id
        self._tracks = self._init_tracks()  # set of tracks ids of all tracks of window
        self._initial_estimate = self._init_initial_estimate()
        self._current_values = self._initial_estimate       # before optimization - current values is initial estimate
        self._factors = self._init_factors()
        self._graph = self._init_graph()
        self._optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial_estimate)

    def _init_tracks(self):
        tracks = set()
        for frame_id in range(self._first_keyframe_id, self._last_keyframe_id + 1):
            frame_obj = self._db.get_frame_obj(frame_id)
            frame_tracks = set(frame_obj.get_tracks_dict().keys())
            tracks = tracks.union(frame_tracks)
        return tracks

    def _init_initial_estimate(self):
        values = gtsam.Values()
        self._init_initial_estimate_cameras(values)  # add cameras nodes
        self._init_initial_estimate_points(values)  # add 3d_points nodes
        return values

    def _init_initial_estimate_cameras(self, values):
        is_first_frame = True
        concat_r = np.identity(3)
        concat_t = np.zeros(3)
        for frame_id in range(self._first_keyframe_id, self._last_keyframe_id + 1):
            if is_first_frame:
                is_first_frame = False
            else:
                mat = self._db.get_frame_obj(frame_id).get_left_camera_pose_mat()
                concat_r = mat[:, :3] @ concat_r  # in coordinates of first frame of window
                concat_t = mat[:, :3] @ concat_t + mat[:, 3]
            inv_r, inv_t = utils.invert_extrinsic_matrix(concat_r, concat_t)
            cur_pos3 = gtsam.Pose3(np.hstack((inv_r, inv_t.reshape(3, 1))))
            values.insert(gtsam.symbol(CAMERA_SYMBOL, frame_id), cur_pos3)

    def _init_factors(self):
        factors = dict()
        cov = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        for track_id in self._tracks:
            track_obj = self._db.get_track_obj(track_id)
            frames_dict = track_obj.get_frames_dict()
            for frame_id, kp_idx in frames_dict.items():
                if frame_id > self._last_keyframe_id:
                    break
                pt_2d_real = utils.get_stereo_point2(self._db, frame_id, kp_idx)

                factor = gtsam.GenericStereoFactor3D \
                    (pt_2d_real, cov, gtsam.symbol(CAMERA_SYMBOL, frame_id), gtsam.symbol(POINT_SYMBOL, track_id),
                     BundleWindow.K)
                factors[(frame_id, track_id)] = factor
        return factors

    def _init_graph(self):
        graph = gtsam.NonlinearFactorGraph()
        for factor in self._factors.values():
            graph.add(factor)
        origin = gtsam.Pose3()
        sigma6d = gtsam.noiseModel.Unit.Create(6)
        graph.add(gtsam.PriorFactorPose3(gtsam.symbol(CAMERA_SYMBOL, 0), origin, sigma6d))
        return graph

    def _init_initial_estimate_points(self, values):
        for track_id in self._tracks:
            track_obj = self._db.get_track_obj(track_id)
            last_frame_id, _ = track_obj.get_last_frame_id_and_kp_idx()
            last_frame_id = min(last_frame_id, self._last_keyframe_id)
            last_frame_pose = values.atPose3(gtsam.symbol(CAMERA_SYMBOL, last_frame_id))
            last_frame_stereo_cam = gtsam.StereoCamera(last_frame_pose, BundleWindow.K)
            last_kp_idx = track_obj.get_kp_idx_of_frame(last_frame_id)
            pt_2d = utils.get_stereo_point2(self._db, last_frame_id, last_kp_idx)
            pt_3d = last_frame_stereo_cam.backproject(pt_2d)
            values.insert(gtsam.symbol(POINT_SYMBOL, track_id), pt_3d)

    def optimize(self):
        self._current_values = self._optimizer.optimize()
        return self._current_values  # returns values object

    def total_factor_error(self):       # returns error of current values
        error = 0
        for factor in self._factors.values():
            error += factor.error(self._current_values)
        return error

    def get_random_factor(self):
        factor_key = random.choice(list(self._factors.keys()))
        return factor_key

    def factor_error(self, frame_id, track_id):
        factor_key = (frame_id, track_id)
        factor_obj = self._factors[factor_key]
        return factor_obj.error(self._current_values)

    def get_pixels_before_and_after_optimize(self, frame_id, track_id):
        left_pixel, right_pixel = self._db.get_feature(frame_id, track_id)
        x_l, x_r, y = left_pixel[0], right_pixel[0], (left_pixel[1] + right_pixel[1]) / 2
        measurement_pixel = np.array([x_l, x_r, y])
        before_pixel = self._project_point(frame_id, track_id, self._initial_estimate)
        after_pixel = self._project_point(frame_id, track_id, self._current_values)
        return measurement_pixel, before_pixel, after_pixel



    def _project_point(self, frame_id, track_id, values):
        pt_3d = values.atPoint3(gtsam.symbol(POINT_SYMBOL, track_id))
        pose3 = values.atPose3(gtsam.symbol(CAMERA_SYMBOL, frame_id))
        cam = gtsam.StereoCamera(pose3, BundleWindow.K)
        proj = cam.project(pt_3d)
        pixel = np.array([proj.uL(), proj.uR(), proj.v()])
        return pixel