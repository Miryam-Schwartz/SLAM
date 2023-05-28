import csv
import sys
import random
import cv2 as cv
import numpy as np

from VAN_ex.code import utils
from VAN_ex.code.DB.Frame import Frame
from VAN_ex.code.DB.Track import Track


class DataBase:
    def __init__(self):
        self._tracks_dict = dict()
        self._frames_dict = dict()
        self._dict_matches_between_frames = dict()  # key = (first frame, second frame), value = list of tuples of idx
        self._concat_r = np.identity(3)
        self._concat_t = np.zeros(3)

    def find_cam_location_and_concat_mats(self, second_frame_id, cur_r, cur_t):
        self._concat_r = cur_r @ self._concat_r
        self._concat_t = cur_r @ self._concat_t + cur_t
        left_cam_location = ((-self._concat_r.T @ self._concat_t).reshape(1, 3))[0]
        self._frames_dict[second_frame_id].set_left_camera_location(left_cam_location)

    def add_frame(self, frame_id, kp_left, kp_right, des_left):
        frame = Frame(frame_id, kp_left, kp_right, des_left)
        self._frames_dict[frame_id] = frame
        if frame_id != 0:
            self._update_tracks(frame_id - 1, frame_id)
        else:
            frame.set_inliers_percentage(100)
            frame.set_left_camera_location(np.zeros(3))

    def _update_tracks(self, first_frame_id, second_frame_id):
        first_des = self._frames_dict[first_frame_id].get_des_left()
        second_des = self._frames_dict[second_frame_id].get_des_left()
        matches = utils.find_matches(first_des, second_des)
        matches = [(match.queryIdx, match.trainIdx) for match in matches]
        extrinsic_camera_mat_second_frame_left, matches, inliers_percentage = \
            self.RANSAC(first_frame_id, second_frame_id, matches)
        cur_r = extrinsic_camera_mat_second_frame_left[:, :3]
        cur_t = extrinsic_camera_mat_second_frame_left[:, 3]
        self.find_cam_location_and_concat_mats(second_frame_id, cur_r, cur_t)
        self._dict_matches_between_frames[(first_frame_id, second_frame_id)] = matches
        first_frame = self._frames_dict[first_frame_id]
        second_frame = self._frames_dict[second_frame_id]
        second_frame.set_inliers_percentage(inliers_percentage)
        for first_idx_kp_left, second_idx_kp_left in matches:
            is_exist_track = False
            for _, track in first_frame.get_tracks_dict().items():
                if track.get_kp_idx_of_frame(first_frame_id) == first_idx_kp_left:
                    is_exist_track = True
                    # print(second_frame_id, " added to track id ", _)
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

    def RANSAC(self, first_frame_id, second_frame_id, matches):
        p, eps = 0.99, 0.99  # eps = prob to be outlier
        i = 0
        size = len(matches)
        max_supporters_num = 0
        idxs_max_supports_matches = None
        while eps > 0 and i < utils.calc_max_iterations(p, eps, 4):
            points_2d, points_3d = self.sample_4_points(first_frame_id, second_frame_id, matches)
            extrinsic_camera_mat_left1, extrinsic_camera_mat_right1 = \
                utils.estimate_frame1_mats_pnp(points_2d, points_3d, Frame.INDENTATION_RIGHT_CAM_MAT, Frame.k)
            if extrinsic_camera_mat_left1 is None:
                continue
            idxs_of_supporters_matches = self.find_supporters(first_frame_id, second_frame_id, matches,
                                                              extrinsic_camera_mat_left1,
                                                              extrinsic_camera_mat_right1)
            supporters_num = len(idxs_of_supporters_matches)
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
        extrinsic_camera_mat_left1, extrinsic_camera_mat_right1 = \
            utils.estimate_frame1_mats_pnp(points_2d_supporters, points_3d_supporters,
                                           Frame.INDENTATION_RIGHT_CAM_MAT, Frame.k,
                                           flag=cv.SOLVEPNP_ITERATIVE)

        idxs_max_supports_matches = self.find_supporters(first_frame_id, second_frame_id, matches,
                                                         extrinsic_camera_mat_left1,
                                                         extrinsic_camera_mat_right1)
        matches = np.array(matches)
        idxs_max_supports_matches = np.array(idxs_max_supports_matches)
        return extrinsic_camera_mat_left1, matches[idxs_max_supports_matches], \
            (len(idxs_max_supports_matches) / len(matches)) * 100

    def sample_4_points(self, first_frame_id, second_frame_id, matches):
        rand_idxs = random.sample(range(len(matches)), 4)
        points_3d = np.empty((4, 3))
        points_2d = np.empty((4, 2))
        for i in range(4):
            cur_match = matches[rand_idxs[i]]  # kp_idx of first frame, kp_idx of second frame
            points_3d[i] = self._frames_dict[first_frame_id].get_3d_point(cur_match[0])
            pixel_left, pixel_right = self._frames_dict[second_frame_id].get_feature_pixels(cur_match[1])
            points_2d[i] = np.array(pixel_left)
        return points_2d, points_3d

    def get_tracks_of_frame(self, frame_id):
        if frame_id not in self._frames_dict:
            raise "invalid frame id"
        return list(self._frames_dict[frame_id].get_tracks_dict().keys())

    def get_frames_of_track(self, track_id):
        if track_id not in self._tracks_dict:
            raise "invalid track id"
        return list(self._tracks_dict[track_id].get_frames_dict().keys())

    def get_feature(self, frame_id, track_id):
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

    @staticmethod
    def _match_in_frame(im_idx, threshold=1):
        img1, img2 = utils.read_images(im_idx)
        kp1, des1, kp2, des2 = utils.detect_and_compute(img1, img2)
        matches = utils.find_matches(des1, des2)
        kp1, des1, kp2, des2, _ = utils.get_correlated_kps_and_des_from_matches(kp1, des1, kp2, des2, matches)
        kp1, des1, kp2, des2 = utils.rectified_stereo_pattern_test(kp1, des1, kp2, des2, threshold)
        return np.array(kp1), np.array(des1), np.array(kp2), np.array(des2)

    def fill_database(self, frame_number):
        for i in range(frame_number):
            print(f'---- frame iteration{i}----')
            kp1, des1, kp2, des2 = DataBase._match_in_frame(i)
            kp1 = [kp.pt for kp in kp1]
            kp2 = [kp.pt for kp in kp2]
            self.add_frame(i, kp1, kp2, des1)

    def save_database(self, path, db_id=''):
        self._save_tracks(f'{path}tracks{db_id}.csv')
        self._save_match_in_frame(f'{path}match_in_frame{db_id}.csv')
        self._save_match_between_frames(f'{path}match_between_frames{db_id}.csv')
        self._save_left_camera_locations(f'{path}left_camera_locations{db_id}.csv')

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

    def _save_left_camera_locations(self, path):
        header = ['frame_id', 'x', 'y', 'z']
        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for frame_id, frame in self._frames_dict.items():
                location = frame.get_left_camera_location()
                row = [frame_id, location[0], location[1], location[2]]
                writer.writerow(row)

    def read_database(self, path, db_id=''):
        # assume database is empty
        assert (self.get_frames_number() == 0 and self.get_tracks_number() == 0)
        self._read_tracks(f'{path}tracks{db_id}.csv')
        self._read_match_in_frame(f'{path}match_in_frame{db_id}.csv')
        self._read_match_between_frames(f'{path}match_between_frames{db_id}.csv')
        self._read_left_camera_location(f'{path}left_camera_locations{db_id}.csv')
        # assume that the db will not be extended (descriptors are not been saved)

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
                frame_id, idx_kp, x_l, y_l, x_r, y_r =\
                    int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
                if frame_id not in self._frames_dict:
                    self._frames_dict[frame_id] = Frame(frame_id, [(x_l, y_l)], [(x_r, y_r)], [])
                else:
                    self._frames_dict[frame_id].add_kp(idx_kp, (x_l, y_l), (x_r, y_r))
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

    def _read_left_camera_location(self, path):
        with open(path, 'r') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                frame_id, x, y, z = int(row[0]), float(row[1]), float(row[2]), float(row[3])
                # assume frames_dict is full
                self._frames_dict[frame_id].set_left_camera_location(np.array([x, y, z]))

    def get_tracks_number(self):
        return len(self._tracks_dict)

    def get_frames_number(self):
        return len(self._frames_dict)

    def get_mean_max_and_min_track_len(self):
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
        sum_tracks = 0
        for frame in self._frames_dict.values():
            sum_tracks += frame.get_number_of_tracks()
        return sum_tracks / self.get_frames_number()

    def get_random_track_in_len(self, length):
        tracks = []
        for track in self._tracks_dict.values():
            if track.get_track_len() >= length:
                tracks.append(track)
        rand_idx = random.randint(0, len(tracks)-1)
        return tracks[rand_idx]

    def get_frame_obj(self, frame_id):
        assert (frame_id in self._frames_dict)
        return self._frames_dict[frame_id]

    def find_supporters(self, first_frame_id, second_frame_id, matches, extrinsic_camera_mat_left1,
                        extrinsic_camera_mat_right1):
        idxs_supports_matches = []
        for i in range(len(matches)):
            pt_3d = self._frames_dict[first_frame_id].get_3d_point(matches[i][0])
            pixel_second_left = utils.project_3d_pt_to_pixel(Frame.k, extrinsic_camera_mat_left1, pt_3d)
            pixel_left, pixel_right = self._frames_dict[second_frame_id].get_feature_pixels(matches[i][1])
            real_pixel_second_left = np.array(pixel_left)
            pixel_second_right = utils.project_3d_pt_to_pixel(Frame.k, extrinsic_camera_mat_right1, pt_3d)
            real_pixel_second_right = np.array(pixel_right)
            if np.linalg.norm(real_pixel_second_left - pixel_second_left) <= 2 \
                    and np.linalg.norm(real_pixel_second_right - pixel_second_right) <= 2:
                idxs_supports_matches.append(i)
        return idxs_supports_matches

    def get_track_obj(self, track_id):
        assert (track_id in self._tracks_dict)
        return self._tracks_dict[track_id]

    def get_camera_locations(self):
        locations = np.empty((self.get_frames_number(), 3))
        for frame_id, frame_obj in self._frames_dict.items():
            locations[frame_id] = frame_obj.get_left_camera_location()
        return locations



