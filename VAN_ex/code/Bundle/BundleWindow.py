import random
import statistics
import gtsam
import numpy as np
from matplotlib import pyplot as plt
from VAN_ex.code import utils
from gtsam.utils import plot

CAMERA_SYMBOL = 'c'
POINT_SYMBOL = 'q'
MAX_Z_VALUE = 100


class BundleWindow:
    """
        A class used to represent single bundle window. Bundle window is sequence of some frames with their key points.
        Given initial estimate (from database), we can optimize positions of cameras and points.
        All positions in the window are in first keyframe coordinates.
        ...

        Attributes
        ----------
        _db  : DB
            database instance includes all the data about the tracking (tracks frames ect.)
        _first_keyframe_id  :   int
            id of first keyframe of the window
        _last_keyframe_id   :   int
            id of ladt keyframe of the window
        _tracks :   set(ints)
            set of track ids of all tracks that are in the window
        _initial_estimate   :   gtsam.Values
            like a dictionary object. saves the data about all frames and points that are in this window. each frame
            and point has initial guess about its position.
            all the positions in the window are in _first_keyframe_id coordinates.
        _current_values :   gtsam.Values
            saves the data about cameras and points positions after optimization.
            before optimization, it holds initial estimate values
        _factors    :   dict
            saves data about factors in the window (factor is link between camera and point).
            key is (frame_id, track_id). Value is gtsam factor object.
        _prior_factor   :   gtsam.PriorFactorPose3
            factor for first keyframe (0,0,0)
        _graph  :   gtsam.NonlinearFactorGraph
            gtsam object that holds all factors
        _optimizer  :   gtsam.LevenbergMarquardtOptimizer
            gtsam optimizer from type Levenberg-Marquardt

        Static
        ----------
        indentation_right_cam   :   float
            indentation between left and right camera on each frame
        K   :   gtsam.Cal3_S2Stereo
            intrinsic camera - calibration

        """
    k, _, m_right = utils.read_cameras()
    indentation_right_cam = m_right[0][3]
    K = gtsam.Cal3_S2Stereo(k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], -indentation_right_cam)

    def __init__(self, db, first_keyframe_id, last_keyframe_id):
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

    def optimize(self):
        """
        :return: gtsam.Values object with values of points and cameras, after optimization
        """
        self._current_values = self._optimizer.optimize()
        return self._current_values

    # ================ Initialization ================ #

    def _init_tracks(self):
        """
        Find all tracks that are part of the window, means tracks that associated to more than one frame in the window
        :return: set of track ids of all tracks that are in the window
        """
        tracks = set()
        # add to tracks set all tracks that associated with one of the window frames
        for frame_id in range(self._first_keyframe_id, self._last_keyframe_id + 1):
            frame_obj = self._db.get_frame_obj(frame_id)
            frame_tracks = set(frame_obj.get_tracks_dict().keys())
            tracks = tracks.union(frame_tracks)
        # delete from tracks set, all tracks that are associated to only one frame
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
        return tracks - tracks_to_delete

    def _init_initial_estimate(self):
        values = gtsam.Values()
        self._init_initial_estimate_cameras(values)  # add cameras nodes
        self._init_initial_estimate_points(values)  # add 3d_points nodes
        return values

    def _init_initial_estimate_cameras(self, values):
        """
        add to initial estimate camera matrices of all frames (calculated from pnp algorithm)
        :param values: gtsam.Values object to fill with the initial estimate
        """
        is_first_frame = True
        concat_r = np.identity(3)
        concat_t = np.zeros(3)
        for frame_id in range(self._first_keyframe_id, self._last_keyframe_id + 1):
            # skip over first frame (it is trivial, initialized up)
            if is_first_frame:
                is_first_frame = False
            else:
                # convert camera matrix to be in global coordinates
                mat = self._db.get_frame_obj(frame_id).get_left_camera_pose_mat()
                concat_r = mat[:, :3] @ concat_r  # in coordinates of first frame of window
                concat_t = mat[:, :3] @ concat_t + mat[:, 3]
            # convert camera matrix to be a matrix that transforms from pose to world coordinates (used by gtsam)
            inv_r, inv_t = utils.invert_extrinsic_matrix(concat_r, concat_t)
            cur_pos3 = gtsam.Pose3(np.hstack((inv_r, inv_t.reshape(3, 1))))
            values.insert(gtsam.symbol(CAMERA_SYMBOL, frame_id), cur_pos3)

    def _init_initial_estimate_points(self, values):
        """
        add to initial estimate 3d points of all tracks that are in the window (accept filtered points by z value)
        :param values: gtsam.Values object to fill with the initial estimate
        """
        updated_tracks = set()
        for track_id in self._tracks:
            # make triangulation from min(last track frame, last window frame)
            track_obj = self._db.get_track_obj(track_id)
            last_frame_id, _ = track_obj.get_last_frame_id_and_kp_idx()
            last_frame_id = min(last_frame_id, self._last_keyframe_id)
            last_frame_pose = values.atPose3(gtsam.symbol(CAMERA_SYMBOL, last_frame_id))
            last_frame_stereo_cam = gtsam.StereoCamera(last_frame_pose, BundleWindow.K)
            last_kp_idx = track_obj.get_kp_idx_of_frame(last_frame_id)
            pt_2d = utils.get_stereo_point2(self._db, last_frame_id, last_kp_idx)
            pt_3d = last_frame_stereo_cam.backproject(pt_2d)

            # filter points with negative or too big z value
            if pt_3d[2] <= 0 or pt_3d[2] >= MAX_Z_VALUE:
                continue
            else:
                values.insert(gtsam.symbol(POINT_SYMBOL, track_id), pt_3d)
                updated_tracks.add(track_id)
        # update tracks have only track ids of points that have not been filtered
        self._tracks = updated_tracks

    def _init_factors(self):
        """
        create factors between all frames in window, to all key-points in each frame.
        :return: dictionary with data about factors. key is (frame_id, track_id). Value is gtsam factor object.
        """
        factors = dict()
        cov = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)

        for track_id in self._tracks:
            track_obj = self._db.get_track_obj(track_id)
            frames_dict = track_obj.get_frames_dict()
            for frame_id, kp_idx in frames_dict.items():
                if frame_id > self._last_keyframe_id or frame_id < self._first_keyframe_id:
                    continue
                pt_2d_real = utils.get_stereo_point2(self._db, frame_id, kp_idx)
                factor = gtsam.GenericStereoFactor3D \
                    (pt_2d_real, cov, gtsam.symbol(CAMERA_SYMBOL, frame_id), gtsam.symbol(POINT_SYMBOL, track_id),
                     BundleWindow.K)
                factors[(frame_id, track_id)] = factor
        return factors

    def _init_graph(self):
        """
        add all factors in factors dictionary and prior factor to graph
        :return: gtsam object that holds all factors
        """
        graph = gtsam.NonlinearFactorGraph()
        for factor in self._factors.values():
            graph.add(factor)
        graph.add(self._prior_factor)
        return graph

    # ================ getters ================ #

    def get_first_keyframe_id(self):
        return self._first_keyframe_id

    def get_last_keyframe_id(self):
        return self._last_keyframe_id

    def get_points(self):
        """
        :return: list of all 3d points in the window, as gtsam.Point3 objects.
        """
        points = []
        for track_id in self._tracks:
            pt = self._current_values.atPoint3(gtsam.symbol(POINT_SYMBOL, track_id))
            points.append(pt)
        return points

    def get_random_factor(self):
        factor_key = random.choice(list(self._factors.keys()))
        return factor_key

    def get_frame_location(self, frame_id):
        """
        :param frame_id:
        :return: np array in shape (1,3), represents location of frame. the location is calculated from current values
        """
        pose3 = self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, frame_id))
        return np.array([pose3.x(), pose3.y(), pose3.z()])

    def get_frame_pose3(self, frame_id):
        """
        :param frame_id:
        :return: gtsam.Pose3 object, represents frame pose
        """
        return self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, frame_id))

    def get_marginals(self):
        return gtsam.Marginals(self._graph, self._current_values)

    def get_current_values(self):
        return self._current_values

    def get_covariance(self):
        """
        :return: np array in shape (6,6) represents covariance matrix of last keyframe, in condition of first frame
        """
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol(CAMERA_SYMBOL, self._first_keyframe_id))
        keys.append(gtsam.symbol(CAMERA_SYMBOL, self._last_keyframe_id))
        marginals = self.get_marginals()
        sliced_inform_mat = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
        return np.linalg.inv(sliced_inform_mat)

    def get_pixels_before_and_after_optimize(self, frame_id, track_id):
        """
        :param frame_id:
        :param track_id:
        :return: 3 np arrays with the pixel of track_id point on frame_id image. the pixels are: measurment (key point),
        before optimization (project 3d point on frame image) and after bundle adjustment optimization
        """
        left_pixel, right_pixel = self._db.get_feature(frame_id, track_id)
        x_l, x_r, y = left_pixel[0], right_pixel[0], (left_pixel[1] + right_pixel[1]) / 2
        measurement_pixel = np.array([x_l, x_r, y])
        before_pixel = BundleWindow._project_point(frame_id, track_id, self._initial_estimate)
        after_pixel = BundleWindow._project_point(frame_id, track_id, self._current_values)
        return measurement_pixel, before_pixel, after_pixel

    @staticmethod
    def _project_point(frame_id, track_id, values):
        """
        :param frame_id:
        :param track_id:
        :param values: gtsam.Values
        :return: projected pixel of track_id point on frame_id image
        """
        pt_3d = values.atPoint3(gtsam.symbol(POINT_SYMBOL, track_id))
        pose3 = values.atPose3(gtsam.symbol(CAMERA_SYMBOL, frame_id))
        cam = gtsam.StereoCamera(pose3, BundleWindow.K)
        proj = cam.project(pt_3d)
        return np.array([proj.uL(), proj.uR(), proj.v()])

    # ================ errors ================ #

    def get_prior_factor_error(self):
        return self._prior_factor.error(self._current_values)

    def get_mean_factors_error(self):
        """
        :return: mean error of all factors in graph
        """
        return self.total_factor_error() / len(self._factors)

    def total_factor_error(self):
        """
        :return: sum of error in all factors graph. error is calculated according to current values
        """
        error = 0
        for factor in self._factors.values():
            error += factor.error(self._current_values)
        return error

    def factor_error(self, frame_id, track_id):
        """
        :param frame_id:
        :param track_id:
        :return: factor error of single factor
        """
        factor_key = (frame_id, track_id)
        factor_obj = self._factors[factor_key]
        return factor_obj.error(self._current_values)

    def get_median_projection_error(self):
        """
        sample randomly some tracks, for each track, go over all its frames, project the 3d point to the frame and
        calculate the difference between real and projected pixel. calculate median over all those differences.
        :return: median projection error of window. The error is calculated from random sampled tracks.
        """
        projection_errors_list = []
        random_track_ids = random.sample(self._tracks, 200)
        for track_id in random_track_ids:
            pt_3d = self._current_values.atPoint3(gtsam.symbol(POINT_SYMBOL, track_id))
            track_obj = self._db.get_track_obj(track_id)
            frames_dict = track_obj.get_frames_dict()
            for frame_id, kp_idx in frames_dict.items():
                if frame_id > self._last_keyframe_id or frame_id < self._first_keyframe_id:
                    continue
                # project the 3d point to each frame in track
                pt_2d_real = utils.get_stereo_point2(self._db, frame_id, kp_idx)
                cam_pose = self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, frame_id))
                stereo_cam = gtsam.StereoCamera(cam_pose, BundleWindow.K)
                pt_2d_proj = stereo_cam.project(pt_3d)
                diff = pt_2d_real - pt_2d_proj
                diff = np.array([diff.uL(), diff.uR(), diff.v()])
                projection_errors_list.append(np.linalg.norm(diff))
        return statistics.median(projection_errors_list)

    def calc_projection_error_per_distance(self, projection_error_left, projection_error_right):
        """
        fill the given dictionaries with left and right error. the error is calculated from some sampled tracks.
        :param projection_error_left: dict. Key is distance from min(track last frame, window last frame),
                value is list of errors in left cameras
        :param projection_error_right: same as above, for right cameras
        """
        random_track_ids = random.sample(self._tracks, 8)
        for track_id in random_track_ids:
            pt_3d = self._current_values.atPoint3(gtsam.symbol(POINT_SYMBOL, track_id))
            track_obj = self._db.get_track_obj(track_id)
            frames_dict = track_obj.get_frames_dict()
            last_frame_id, _ = track_obj.get_last_frame_id_and_kp_idx()
            last_frame_id = min(last_frame_id, self._last_keyframe_id)
            for frame_id, kp_idx in frames_dict.items():
                if frame_id > self._last_keyframe_id or frame_id < self._first_keyframe_id:
                    continue
                pt_2d_real = utils.get_stereo_point2(self._db, frame_id, kp_idx)
                cam_pose = self._current_values.atPose3(gtsam.symbol(CAMERA_SYMBOL, frame_id))
                stereo_cam = gtsam.StereoCamera(cam_pose, BundleWindow.K)
                pt_2d_proj = stereo_cam.project(pt_3d)
                diff = pt_2d_real - pt_2d_proj
                diff = np.array([diff.uL(), diff.uR(), diff.v()])
                d = last_frame_id - frame_id
                if d in projection_error_left:
                    projection_error_left[d].append(np.linalg.norm(diff[[0, 2]]))
                    projection_error_right[d].append(np.linalg.norm(diff[[1, 2]]))
                else:
                    projection_error_left[d] = [np.linalg.norm(diff[[0, 2]])]
                    projection_error_right[d] = [np.linalg.norm(diff[[1, 2]])]

    # ================ plot ================ #

    def plot_3d_positions_graph(self, output_path):
        """
        plot trajectory of window
        :param output_path: path to save the figure
        """
        gtsam.utils.plot.plot_trajectory(fignum=0, values=self._current_values,
                                         title=f'Estimated pose for frames {self._first_keyframe_id} - {self._last_keyframe_id}',
                                         save_file=output_path)

    def plot_2d_positions_graph(self, output_path):
        """
        plot cameras and points of window from bird eye view
        :param output_path:
        """
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
