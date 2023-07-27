import csv
import sys
import random
import cv2 as cv
import numpy as np

from VAN_ex.code import utils
from VAN_ex.code.DB.Frame import Frame
from VAN_ex.code.DB.Track import Track


class DataBase:
    """
        A class used to represent a DataBase that holds within all the data of tracking
        ...

        Attributes
        ----------
        _tracks_dict : dict
            Key is track_id, value is track object. Holds all the tracks that are existing.
        _frames_dict : dict
            Key is frame_id, value is frame object. Holds all the frames that are existing.
        _dict_matches_between_frames: dict
            Holds data about matching between frames
            key is tuple from the shape: (first frame, second frame), value is list of tuples is tuple holds two
            indexes, the idx of keypoint in first frame and the idx of keypoint in second frame.
        _concat_r: np.Array in shape (3,3)
            concatenating of R matrices multiplication
        _concat_t: np.Array in length 3
            concatenating of t matrices
        """

    def __init__(self):
        self._tracks_dict = dict()
        self._frames_dict = dict()
        self._dict_matches_between_frames = dict()  # key = (first frame, second frame), value = list of tuples of idx
        self._concat_r = np.identity(3)
        self._concat_t = np.zeros(3)

    def get_tracks_of_frame(self, frame_id):
        """
        :param frame_id:
        :return: id-s of tracks that are associated with specific frame
        """
        if frame_id not in self._frames_dict:
            raise "invalid frame id"
        return list(self._frames_dict[frame_id].get_tracks_dict().keys())

    def get_frames_of_track(self, track_id):
        """
        :param track_id:
        :return: id-s of frames that are associated with specific track
        """
        if track_id not in self._tracks_dict:
            raise "invalid track id"
        return list(self._tracks_dict[track_id].get_frames_dict().keys())

    def get_feature(self, frame_id, track_id):
        """
        :param frame_id:
        :param track_id:
        :return: given frame_id and track_id, return pixels of feature in the frame, that is associated with this track
        """
        if frame_id not in self._frames_dict:
            raise "invalid frame id"
        if track_id not in self._tracks_dict:
            raise "invalid track id"
        track = self._tracks_dict[track_id]
        frames_to_kp_of_track = track.get_frames_dict()
        if frame_id not in frames_to_kp_of_track:
            raise "frame does not exist in track"
        idx_kp_left = frames_to_kp_of_track[frame_id]
        return self._frames_dict[frame_id].get_feature_pixels(idx_kp_left)

    def get_tracks_number(self):
        """
        :return: Total number of tracks
        """
        return len(self._tracks_dict)

    def get_frames_number(self):
        """
        :return: Total number of frames
        """
        return len(self._frames_dict)

    def get_random_track_in_len(self, length):
        """
        :param length:
        :return: random track that its length (frames number) is bigger than or equal to param length
        """
        tracks = []
        for track in self._tracks_dict.values():
            if track.get_track_len() >= length:
                tracks.append(track)
        rand_idx = random.randint(0, len(tracks) - 1)
        return tracks[rand_idx]

    def get_frame_obj(self, frame_id):
        assert (frame_id in self._frames_dict)
        return self._frames_dict[frame_id]

    def get_track_obj(self, track_id):
        assert (track_id in self._tracks_dict)
        return self._tracks_dict[track_id]

    def get_camera_locations(self):
        """
        :return: np.Array with all camera locations of all frames
        """
        concat_r, concat_t = np.identity(3), np.zeros(3)
        locations = np.empty((self.get_frames_number(), 3))
        for frame_id, frame_obj in self._frames_dict.items():
            pose_mat = frame_obj.get_left_camera_pose_mat()
            cur_r, cur_t = pose_mat[:, :3], pose_mat[:, 3]
            concat_r = cur_r @ concat_r
            concat_t = cur_r @ concat_t + cur_t
            left_cam_location = ((-concat_r.T @ concat_t).reshape(1, 3))[0]
            locations[frame_id] = left_cam_location
        return locations

    def get_left_camera_mats_in_world_coordinates(self):
        """
        :return: np array include all camera matrices, calculated in global coordinates
        """
        concat_r, concat_t = np.identity(3), np.zeros(3)
        mats = []
        for frame_id, frame_obj in self._frames_dict.items():
            pose_mat = frame_obj.get_left_camera_pose_mat()
            cur_r, cur_t = pose_mat[:, :3], pose_mat[:, 3]
            concat_r = cur_r @ concat_r
            concat_t = cur_r @ concat_t + cur_t
            mats.append(np.hstack((concat_r, concat_t.reshape(3, 1))))
        return np.array(mats)

    # ================ Tracking Statistics ================ #

    def get_mean_max_and_min_track_len(self):
        """
        :return: Mean track length, maximum and minimum track lengths
        """
        sum_len = 0
        min_len = sys.maxsize
        max_len = 0
        for track in self._tracks_dict.values():
            track_len = track.get_track_len()
            sum_len += track_len
            min_len = min(min_len, track_len)
            max_len = max(max_len, track_len)
        return sum_len / self.get_tracks_number(), max_len, min_len

    def get_mean_number_of_tracks_on_frame(self):
        """
        :return: Mean number of frame links (number of tracks on an average image)
        """
        sum_tracks = 0
        for frame in self._frames_dict.values():
            sum_tracks += frame.get_number_of_tracks()
        return sum_tracks / self.get_frames_number()

    def get_median_track_len(self):
        """
        :return: Median track len over all the tracks
        """
        lengths = np.empty(len(self._tracks_dict))
        for i, track in enumerate(self._tracks_dict.values()):
            track_len = track.get_track_len()
            lengths[i] = track_len
        return np.median(lengths)

    def set_initial_camera(self, k, m_left, m_right):
        Frame.k = k
        Frame.m_left = m_left
        Frame.m_right = m_right
        Frame.INDENTATION_RIGHT_CAM_MAT = m_right[0][3]

    # ================ Tracking ================ #

    def fill_database(self, frame_number):
        """
        Read images from dataset, find matches, make triangulation, and calculate the motion between each frame to the
        next frame. Assume that after calling to this function, the database will be filled with all the details
        (frames, tracks and such as this). To calculate the relative motion between each frame to the next,
        we use RANSAC with PnP.
        :param frame_number: The calculation will be from frame number 0, till this frame
        """
        k, m_left, m_right = utils.read_cameras()
        self.set_initial_camera(k, m_left, m_right)
        for i in range(frame_number):
            print(f'---- frame iteration {i}----')
            left_kp, left_des, right_kp, right_des = DataBase._match_in_frame(i)
            left_kp = [kp.pt for kp in left_kp]
            right_kp = [kp.pt for kp in right_kp]
            self.add_frame(i, left_kp, right_kp, left_des)

    @staticmethod
    def _match_in_frame(im_idx, threshold=1):
        """
        Read frame images, find KeyPoints on both images, find matches between them, and use rectified stereo pattern to
        reject not suitable matches
        :param im_idx: index of frame to find matches on it
        :param threshold: rectified stereo pattern threshold, i.e. by how many pixels the keypoints in a match can be
        far from each other.
        :return: np arrays of: left_key-points, left_descriptors, right_key-points, right_descriptors
        """
        left_img, right_img = utils.read_images(im_idx)
        left_kp, left_des, right_kp, right_des = utils.detect_and_compute(left_img, right_img)
        matches = utils.find_matches(left_des, right_des)
        left_kp, left_des, right_kp, right_des, _ =\
            utils.get_correlated_kps_and_des_from_matches(left_kp, left_des, right_kp, right_des, matches)
        left_kp, left_des, right_kp, right_des =\
            utils.rectified_stereo_pattern_test(left_kp, left_des, right_kp, right_des, threshold)
        return np.array(left_kp), np.array(left_des), np.array(right_kp), np.array(right_des)

    def add_frame(self, frame_id, kp_left, kp_right, des_left):
        frame = Frame(frame_id, kp_left, kp_right, des_left)
        self._frames_dict[frame_id] = frame
        if frame_id == 0:
            frame.set_inliers_percentage(100)
            pose0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
            frame.set_left_camera_pose_mat(pose0)
        else:
            self._update_tracks(frame_id - 1, frame_id)

    # ================ Helpers for add_frame Method ================ #

    def _update_tracks(self, first_frame_id, second_frame_id):
        """
        Find matches between first and second frames, calculate second frame camera location using RANSAC & PnP,
        update inliers percentage of second frame and remove outliers. for each match - if the keypoint in first
        frame is associated with existing track - add to the track a new frame. else - create new track.
        :param first_frame_id:
        :param second_frame_id:
        """
        first_des = self._frames_dict[first_frame_id].get_des_left()
        second_des = self._frames_dict[second_frame_id].get_des_left()
        matches = utils.find_matches(first_des, second_des)
        matches = [(match.queryIdx, match.trainIdx) for match in matches]

        extrinsic_camera_mat_second_frame_left, matches, inliers_percentage = \
            self.RANSAC(first_frame_id, second_frame_id, matches)

        # cur_r = extrinsic_camera_mat_second_frame_left[:, :3]
        # cur_t = extrinsic_camera_mat_second_frame_left[:, 3]
        # self._find_cam_location_and_concat_mats(second_frame_id, cur_r, cur_t)

        self._dict_matches_between_frames[(first_frame_id, second_frame_id)] = matches

        first_frame = self._frames_dict[first_frame_id]
        second_frame = self._frames_dict[second_frame_id]
        second_frame.set_inliers_percentage(inliers_percentage)
        second_frame.set_left_camera_pose_mat(extrinsic_camera_mat_second_frame_left)

        for first_idx_kp_left, second_idx_kp_left in matches:
            is_exist_track = False
            for _, track in first_frame.get_tracks_dict().items():
                if track.get_kp_idx_of_frame(first_frame_id) == first_idx_kp_left:
                    is_exist_track = True
                    # update track, update second_frame tracks
                    track.add_frame(second_frame_id, second_idx_kp_left)
                    second_frame.add_track(track)
                    break
            # if we were not in the if - open new track, update tracks of first and second, and update tracks_dict
            if not is_exist_track:
                new_track = Track(first_frame_id, second_frame_id, first_idx_kp_left, second_idx_kp_left)
                first_frame.add_track(new_track)
                second_frame.add_track(new_track)
                self._tracks_dict[new_track.get_id()] = new_track

    # def _find_cam_location_and_concat_mats(self, second_frame_id, cur_r, cur_t):
    #     self._concat_r = cur_r @ self._concat_r
    #     self._concat_t = cur_r @ self._concat_t + cur_t
    #     left_cam_location = ((-self._concat_r.T @ self._concat_t).reshape(1, 3))[0]
    #     self._frames_dict[second_frame_id].set_left_camera_location(left_cam_location)

    def RANSAC(self, first_frame_id, second_frame_id, matches):
        p, eps = 0.99, 0.99  # eps = prob to be outlier
        i = 0
        size = len(matches)
        # print("matches number between frames: ", size)
        max_supporters_num = 0
        idxs_max_supports_matches = None
        while eps > 0 and i < utils.calc_max_iterations(p, eps, 4):
            points_2d, points_3d = self._sample_4_points(first_frame_id, second_frame_id, matches)
            extrinsic_camera_mat_second_frame_left, extrinsic_camera_mat_second_frame_right = \
                utils.estimate_second_frame_mats_pnp(points_2d, points_3d, Frame.INDENTATION_RIGHT_CAM_MAT, Frame.k)
            if extrinsic_camera_mat_second_frame_left is None:
                continue
            idxs_of_supporters_matches = self._find_supporters(first_frame_id, second_frame_id, matches,
                                                               extrinsic_camera_mat_second_frame_left,
                                                               extrinsic_camera_mat_second_frame_right)
            supporters_num = len(idxs_of_supporters_matches)
            if first_frame_id == 2387:
                print("supporters num ", supporters_num)
            if supporters_num > max_supporters_num:
                max_supporters_num = supporters_num
                idxs_max_supports_matches = idxs_of_supporters_matches
                # update eps
                eps = (size - max_supporters_num) / size
            i += 1
        points_3d_supporters = np.empty((max_supporters_num, 3))
        points_2d_supporters = np.empty((max_supporters_num, 2))
        for i in range(max_supporters_num):
            cur_match = matches[idxs_max_supports_matches[i]]
            points_3d_supporters[i] = self._frames_dict[first_frame_id].get_3d_point(cur_match[0])
            pixel_left, pixel_right = self._frames_dict[second_frame_id].get_feature_pixels(cur_match[1])
            points_2d_supporters[i] = np.array(pixel_left)
        extrinsic_camera_mat_second_frame_left, extrinsic_camera_mat_second_frame_right = \
            utils.estimate_second_frame_mats_pnp(points_2d_supporters, points_3d_supporters,
                                                 Frame.INDENTATION_RIGHT_CAM_MAT, Frame.k,
                                                 flag=cv.SOLVEPNP_ITERATIVE)

        idxs_max_supports_matches = self._find_supporters(first_frame_id, second_frame_id, matches,
                                                          extrinsic_camera_mat_second_frame_left,
                                                          extrinsic_camera_mat_second_frame_right)
        matches = np.array(matches)
        idxs_max_supports_matches = np.array(idxs_max_supports_matches)
        # print("supporters num ", idxs_max_supports_matches.shape[0])
        return extrinsic_camera_mat_second_frame_left, matches[idxs_max_supports_matches], \
            (len(idxs_max_supports_matches) / len(matches)) * 100

    def _sample_4_points(self, first_frame_id, second_frame_id, matches):
        rand_idxs = random.sample(range(len(matches)), 4)
        points_3d = np.empty((4, 3))
        points_2d = np.empty((4, 2))
        for i in range(4):
            cur_match = matches[rand_idxs[i]]  # kp_idx of first frame, kp_idx of second frame
            points_3d[i] = self._frames_dict[first_frame_id].get_3d_point(cur_match[0])
            pixel_left, pixel_right = self._frames_dict[second_frame_id].get_feature_pixels(cur_match[1])
            points_2d[i] = np.array(pixel_left)
        return points_2d, points_3d

    def _sample_4_points_with_filter(self, first_frame_id, second_frame_id, matches):
        rand_idxs = set()
        points_3d = np.empty((4, 3))
        points_2d = np.empty((4, 2))
        i = 0
        while i < 4:
            rand_idx = random.randint(0, len(matches)-1)
            if rand_idx in rand_idxs:
                continue
            rand_idxs.add(rand_idx)
            cur_match = matches[rand_idx]  # kp_idx of first frame, kp_idx of second frame
            point_3d = self._frames_dict[first_frame_id].get_3d_point(cur_match[0])
            if point_3d[2] > 200 or point_3d[2] <= 0:
                continue
            points_3d[i] = point_3d
            pixel_left, pixel_right = self._frames_dict[second_frame_id].get_feature_pixels(cur_match[1])
            points_2d[i] = np.array(pixel_left)
            i += 1
        return points_2d, points_3d

    def _find_supporters(self, first_frame_id, second_frame_id, matches, extrinsic_camera_mat_second_frame_left,
                         extrinsic_camera_mat_second_frame_right):
        """
        Given extrinsic camera matrices of a frame (right and left mats), and matches, return indexes of all the
        matches that are supporters. means, the distance from the pixel to the projected pixel
        (after making triangulation and finding the 3d point), is not bigger than a threshold.
        :param first_frame_id: id of first frame
        :param second_frame_id: id of second frame
        :param matches: list of matches between key-points in first frame, to key-points in second frame. each match is
        a tuple: (idx of kp in first frame, idx of kp in second frame)
        :param extrinsic_camera_mat_second_frame_left: second frame extrinsic left camera matrix, represents the relative
        motion from first to second frame.
        :param extrinsic_camera_mat_second_frame_right: second frame extrinsic right camera matrix
        :return: indices of all matches from matches list that are supporters
        """
        idxs_supports_matches = []
        for i in range(len(matches)):
            pt_3d = self._frames_dict[first_frame_id].get_3d_point(matches[i][0])
            pixel_second_left = utils.project_3d_pt_to_pixel(Frame.k, extrinsic_camera_mat_second_frame_left, pt_3d)
            pixel_left, pixel_right = self._frames_dict[second_frame_id].get_feature_pixels(matches[i][1])
            real_pixel_second_left = np.array(pixel_left)
            pixel_second_right = utils.project_3d_pt_to_pixel(Frame.k, extrinsic_camera_mat_second_frame_right, pt_3d)
            real_pixel_second_right = np.array(pixel_right)
            if np.linalg.norm(real_pixel_second_left - pixel_second_left) <= 2 \
                    and np.linalg.norm(real_pixel_second_right - pixel_second_right) <= 2:
                idxs_supports_matches.append(i)
        return idxs_supports_matches

    # ================ Serializing and Reading ================ #

    def save_database(self, path, db_id=''):
        self._save_tracks(f'{path}tracks{db_id}.csv')
        self._save_match_in_frame(f'{path}match_in_frame{db_id}.csv')
        self._save_match_between_frames(f'{path}match_between_frames{db_id}.csv')
        self._save_left_camera_poses(f'{path}left_camera_poses{db_id}.csv')

    def _save_tracks(self, path):
        header = ['track_id', 'frame_id', 'idx_kp']
        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for track_id, track in self._tracks_dict.items():
                for frame_id, idx_kp in track.get_frames_dict().items():
                    row = [track_id, frame_id, idx_kp]
                    writer.writerow(row)

    def _save_match_in_frame(self, path):
        header = ['frame_id', 'idx_kp', 'x_left', 'y_left', 'x_right', 'y_right']
        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for frame_id, frame in self._frames_dict.items():
                for i in range(frame.get_kp_len()):
                    pixel_left, pixel_right = frame.get_feature_pixels(i)
                    row = [frame_id, i, pixel_left[0], pixel_left[1], pixel_right[0], pixel_right[1]]
                    writer.writerow(row)

    def _save_match_between_frames(self, path):
        header = ['first_frame_id', 'second_frame_id', 'first_idx_kp', 'second_idx_kp']
        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for frames, matches_list in self._dict_matches_between_frames.items():
                first_frame_id, second_frame_id = frames
                for first_idx_kp, second_idx_kp in matches_list:
                    row = [first_frame_id, second_frame_id, first_idx_kp, second_idx_kp]
                    writer.writerow(row)

    def _save_left_camera_poses(self, path):
        header = ['frame_id', 'm00', 'm01', 'm02', 'm03', 'm10', 'm11', 'm12', 'm13', 'm20', 'm21', 'm22', 'm23', ]
        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for frame_id, frame in self._frames_dict.items():
                pose_mat = frame.get_left_camera_pose_mat()
                row = [frame_id]
                row.extend(list(pose_mat.flatten()))
                writer.writerow(row)

    def read_database(self, path, db_id=''):
        # assume database is empty
        assert (self.get_frames_number() == 0 and self.get_tracks_number() == 0)
        k, m_left, m_right = utils.read_cameras()
        self.set_initial_camera(k, m_left, m_right)
        self._read_tracks(f'{path}tracks{db_id}.csv')
        self._read_match_in_frame(f'{path}match_in_frame{db_id}.csv')
        self._read_match_between_frames(f'{path}match_between_frames{db_id}.csv')
        self._read_left_camera_poses(f'{path}left_camera_poses{db_id}.csv')
        # assume that the db will not be extended (descriptors are not been saved)
        # inliers percentage are also not been saved

    def _read_tracks(self, path):
        with open(path, 'r') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                track_id, frame_id, idx_kp = int(row[0]), int(row[1]), int(row[2])
                if track_id not in self._tracks_dict:
                    # add new track to dict
                    self._tracks_dict[track_id] = Track(track_id, frame_id, idx_kp)
                else:
                    # add to this track a new frame
                    self._tracks_dict[track_id].add_frame(frame_id, idx_kp)

    def _read_match_in_frame(self, path):
        with open(path, 'r') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                frame_id, idx_kp, x_l, y_l, x_r, y_r = \
                    int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
                if frame_id not in self._frames_dict:
                    self._frames_dict[frame_id] = Frame(frame_id, [(x_l, y_l)], [(x_r, y_r)], [])
                else:
                    self._frames_dict[frame_id].add_kp(idx_kp, (x_l, y_l), (x_r, y_r))
        # after creating frames objects, add the tracks object to their dict
        for track_id, track in self._tracks_dict.items():
            for frame_id in track.get_frames_dict():
                assert (frame_id in self._frames_dict)
                self._frames_dict[frame_id].add_track(track)

    def _read_match_between_frames(self, path):
        with open(path, 'r') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                first_frame_id, second_frame_id, first_idx_kp, second_idx_kp = \
                    int(row[0]), int(row[1]), int(row[2]), int(row[3])
                if (first_frame_id, second_frame_id) not in self._dict_matches_between_frames:
                    self._dict_matches_between_frames[(first_frame_id, second_frame_id)] = \
                        [(first_idx_kp, second_idx_kp)]
                else:
                    self._dict_matches_between_frames[(first_frame_id, second_frame_id)] \
                        .append((first_idx_kp, second_idx_kp))

    def _read_left_camera_poses(self, path):
        with open(path, 'r') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                frame_id, row = int(row[0]), row[1:]
                row = [float(n) for n in row]
                row = np.array(row)
                row = row.reshape((3, 4))
                # assume frames_dict is full
                self._frames_dict[frame_id].set_left_camera_pose_mat(row)

