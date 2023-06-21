import random

import gtsam
import numpy as np
from matplotlib import pyplot as plt

from VAN_ex.code import utils
from gtsam.utils import plot

CAMERA_SYMBOL = 'c'
POINT_SYMBOL = 'q'





class BundleWindow:
    k, _, m_right = utils.read_cameras()
    indentation_right_cam = m_right[0][3]
    K = gtsam.Cal3_S2Stereo(k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], -indentation_right_cam)

    def __init__(self, db, first_keyframe_id, last_keyframe_id):
        # print(f'window {first_keyframe_id} - {last_keyframe_id}')
        self._db = db
        self._first_keyframe_id = first_keyframe_id
        self._last_keyframe_id = last_keyframe_id
        self._tracks = self._init_tracks()  # set of tracks ids of all tracks of window
        self._initial_estimate = self._init_initial_estimate()
        self._current_values = self._initial_estimate  # before optimization - current values is initial estimate
        self._factors = self._init_factors()
        sigmas = np.array([(1 * np.pi / 180) ** 2] * 3 + [0.03, 0.003, 0.03])
        cov = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        self._prior_factor = gtsam.PriorFactorPose3(gtsam.symbol(CAMERA_SYMBOL, first_keyframe_id), gtsam.Pose3(), cov)
        self._graph = self._init_graph()
        self._optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial_estimate)

    def get_first_keyframe_id(self):
        return self._first_keyframe_id

    def get_last_keyframe_id(self):
        return self._last_keyframe_id

    def get_points(self):
        points = []
        for track_id in self._tracks:
            pt = self._current_values.atPoint3(gtsam.symbol(POINT_SYMBOL, track_id))
            points.append(pt)
        return points

    def _init_tracks(self):
        tracks = set()
        for frame_id in range(self._first_keyframe_id, self._last_keyframe_id + 1):
            frame_obj = self._db.get_frame_obj(frame_id)
            frame_tracks = set(frame_obj.get_tracks_dict().keys())
            tracks = tracks.union(frame_tracks)
        tracks_to_delete = set()
        for track_id in tracks:
            counter = 0
            track_obj = self._db.get_track_obj(track_id)
            frames_dict_of_track = track_obj.get_frames_dict()
            for frame_id in frames_dict_of_track.keys():
                if self._last_keyframe_id >= frame_id >= self._first_keyframe_id:
                    counter += 1
            if counter <= 1:
                tracks_to_delete.add(track_id)
        # print(f"track len: {len(tracks)}, tracks to delete: {len(tracks_to_delete)}")
        return tracks-tracks_to_delete

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
        cov = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)

        # print(f'len of tracks of window: {self._first_keyframe_id} - {self._last_keyframe_id}: {len(self._tracks)}')
        for track_id in self._tracks:
            track_obj = self._db.get_track_obj(track_id)
            frames_dict = track_obj.get_frames_dict()
            for frame_id, kp_idx in frames_dict.items():
                if frame_id > self._last_keyframe_id or frame_id < self._first_keyframe_id:
                    continue
                pt_2d_real = utils.get_stereo_point2(self._db, frame_id, kp_idx)
                # print(f"add factor : frame_id-{frame_id}, track_id-{track_id}")
                factor = gtsam.GenericStereoFactor3D \
                    (pt_2d_real, cov, gtsam.symbol(CAMERA_SYMBOL, frame_id), gtsam.symbol(POINT_SYMBOL, track_id),
                     BundleWindow.K)
                factors[(frame_id, track_id)] = factor
        return factors

    def _init_graph(self):
        graph = gtsam.NonlinearFactorGraph()
        for factor in self._factors.values():
            graph.add(factor)
        graph.add(self._prior_factor)
        return graph

    def _init_initial_estimate_points(self, values):
        updated_tracks = set()
        for track_id in self._tracks:
            track_obj = self._db.get_track_obj(track_id)
            last_frame_id, _ = track_obj.get_last_frame_id_and_kp_idx()
            last_frame_id = min(last_frame_id, self._last_keyframe_id)
            last_frame_pose = values.atPose3(gtsam.symbol(CAMERA_SYMBOL, last_frame_id))
            last_frame_stereo_cam = gtsam.StereoCamera(last_frame_pose, BundleWindow.K)
            last_kp_idx = track_obj.get_kp_idx_of_frame(last_frame_id)
            pt_2d = utils.get_stereo_point2(self._db, last_frame_id, last_kp_idx)
            pt_3d = last_frame_stereo_cam.backproject(pt_2d)

            if pt_3d[2] <= 0 or pt_3d[2] >= 200:
                continue
            else:
                values.insert(gtsam.symbol(POINT_SYMBOL, track_id), pt_3d)
                updated_tracks.add(track_id)
        self._tracks = updated_tracks

    def optimize(self):
        # print(len(self._tracks))
        # print(len(self._factors.keys()))
        # print(len(self._current_values.keys()))
        self._current_values = self._optimizer.optimize()
        return self._current_values  # returns values object

    def total_factor_error(self):  # returns error of current values
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

    def get_frame_location(self, frame_id):
        pose3 = self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, frame_id))
        return np.array([pose3.x(), pose3.y(), pose3.z()])

    def get_frame_pose3(self, frame_id):
        return self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, frame_id))

    def get_prior_factor_error(self):
        return self._prior_factor.error(self._current_values)

    def plot_3d_positions_graph(self, output_path):
        gtsam.utils.plot.plot_trajectory(fignum=0, values=self._current_values,
                                         title=f'Estimated pose for frames {self._first_keyframe_id} - {self._last_keyframe_id}',
                                         save_file=output_path)
    def plot_2d_positions_graph(self, output_path):
        x_poses = np.empty(self._last_keyframe_id - self._first_keyframe_id + 1)
        z_poses = np.empty(self._last_keyframe_id - self._first_keyframe_id + 1)
        for i, frame_id in enumerate(range(self._first_keyframe_id, self._last_keyframe_id + 1)):
            pose3 = self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, frame_id))
            x_poses[i] = pose3.x()
            z_poses[i] = pose3.z()
        x_points = np.empty(len(self._tracks))
        z_points = np.empty(len(self._tracks))
        for i, track_id in enumerate(self._tracks):
            point3 = self._current_values.atPoint3(gtsam.symbol(POINT_SYMBOL, track_id))
            x_points[i] = point3[0]
            z_points[i] = point3[2]

        fig, ax = plt.subplots()
        ax.scatter(x=x_poses, y=z_poses,
                   c='tab:blue', label='Cameras positions', s=0.3, alpha=0.5)
        ax.scatter(x=x_points, y=z_points,
                   c='tab:orange', label='Points', s=0.5, alpha=0.7)
        ax.set_ylim(0, 200)
        ax.legend()
        plt.title('Cameras & Points - View from above')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.savefig(output_path)

    def get_marginals(self):
        # print("get marginals of window: ", self._first_keyframe_id, ", ", self._last_keyframe_id)
        return gtsam.Marginals(self._graph, self._current_values)

    def get_current_values(self):
        return self._current_values

    def get_covariance(self):
        # the returned covariance is of first anf last frames in window
        # covariance of last frame in condition of first frame
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol(CAMERA_SYMBOL, self._first_keyframe_id))
        keys.append(gtsam.symbol(CAMERA_SYMBOL, self._last_keyframe_id))
        marginals = self.get_marginals()
        sliced_inform_mat = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
        return np.linalg.inv(sliced_inform_mat)
