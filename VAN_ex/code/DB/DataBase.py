import csv

import numpy as np

from VAN_ex.code import utils
from VAN_ex.code.DB.Frame import Frame
from VAN_ex.code.DB.Track import Track


class DataBase:
    def __init__(self):
        self._tracks_dict = dict()
        self._frames_dict = dict()
        self._dict_matches_between_frames = dict()  # key = (first frame, second frame), value = list of tuples of idx

    def add_frame(self, frame_id, kp_left, kp_right, des_left):
        frame = Frame(frame_id, kp_left, kp_right, des_left)
        self._frames_dict[frame_id] = frame
        if frame_id != 0:
            self._update_tracks(frame_id - 1, frame_id)

    def _update_tracks(self, first_frame_id, second_frame_id):
        first_des = self._frames_dict[first_frame_id].get_des_left()
        second_des = self._frames_dict[second_frame_id].get_des_left()
        matches = utils.find_matches(first_des, second_des)
        matches = [(match.queryIdx, match.trainIdx) for match in matches]
        self._dict_matches_between_frames[(first_frame_id, second_frame_id)] = matches
        first_frame = self._frames_dict[first_frame_id]
        second_frame = self._frames_dict[second_frame_id]
        for first_idx_kp_left, second_idx_kp_left in matches:
            is_exist_track = False
            for _, track in first_frame.get_tracks_dict().items():
                if track.get_kp_idx_of_frame(first_frame_id) == first_idx_kp_left:
                    is_exist_track = True
                    print(second_frame_id, " added to track id ", _)
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

    def save_database(self, path):
        self._save_tracks(path)
        self._save_match_in_frame(path)
        self._save_match_between_frames(path)

    def _save_tracks(self, path):
        header = ['track_id', 'frame_id', 'idx_kp']
        with open(path + 'tracks.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for track_id, track in self._tracks_dict.items():
                for frame_id, idx_kp in track.get_frames_dict().items():
                    row = [track_id, frame_id, idx_kp]
                    writer.writerow(row)

    def _save_match_in_frame(self, path):
        header = ['frame_id', 'idx_kp', 'x_left', 'x_right', 'y']
        with open(path + 'match_in_frame.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for frame_id, frame in self._frames_dict.items():
                for i in range(frame.get_kp_len()):
                    x_l, x_r, y = frame.get_feature_pixels(i)
                    row = [frame_id, i, x_l, x_r, y]
                    writer.writerow(row)

    def _save_match_between_frames(self, path):
        header = ['first_frame_id', 'second_frame_id', 'first_idx_kp', 'second_idx_kp']
        with open(path + 'match_between_frames.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for frames, matches_list in self._dict_matches_between_frames.items():
                first_frame_id, second_frame_id = frames
                for first_idx_kp, second_idx_kp in matches_list:
                    row = [first_frame_id, second_frame_id, first_idx_kp, second_idx_kp]
                    writer.writerow(row)

    def read_database(self, path):
        # assume database is empty
        self._read_tracks(path)
        self._read_match_in_frame(path)
        self._read_match_between_frames(path)

    def _read_tracks(self, path):
        with open(path + 'tracks.csv', 'r') as f:
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
        with open(path + 'match_in_frame.csv', 'r') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                frame_id, idx_kp, x_l, x_r, y = int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4])
                if frame_id not in self._frames_dict:
                    # todo think about restore des
                    self._frames_dict[frame_id] = Frame(frame_id, [(x_l, y)], [(x_r, y)], [])
                else:
                    self._frames_dict[frame_id].add_kp(idx_kp, x_l, x_r, y)
        for track_id, track in self._tracks_dict.items():
            for frame_id in track.get_frames_dict():
                assert (frame_id in self._frames_dict)
                self._frames_dict[frame_id].add_track(track)

    def _read_match_between_frames(self, path):
        with open(path + 'match_between_frames.csv', 'r') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            for row in csvreader:
                first_frame_id, second_frame_id, first_idx_kp, second_idx_kp =\
                    int(row[0]), int(row[1]), int(row[2]), int(row[3])
                if (first_frame_id, second_frame_id) not in self._dict_matches_between_frames:
                    self._dict_matches_between_frames[(first_frame_id, second_frame_id)] =\
                        [(first_idx_kp, second_idx_kp)]
                else:
                    self._dict_matches_between_frames[(first_frame_id, second_frame_id)]\
                        .append((first_idx_kp, second_idx_kp))
