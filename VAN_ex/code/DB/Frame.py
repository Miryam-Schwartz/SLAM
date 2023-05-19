class Frame:
    def __init__(self, frame_id, kp_left, kp_right, des_left):
        self._frame_id = frame_id
        # self._kp_left = [kp.pt for kp in kp_left]
        # self._kp_right = [kp.pt for kp in kp_right]    # kp_right and kp_left are correlated after matching
        self._kp_left = kp_left
        self._kp_right = kp_right
        self._des_left = des_left
        self._tracks_dict = dict()              # key = track_id, val = track object

    def add_track(self, new_track):
        self._tracks_dict[new_track.get_id()] = new_track

    def get_tracks_dict(self):
        return self._tracks_dict

    def get_des_left(self):
        return self._des_left

    def get_feature_pixels(self, kp_idx):
        x_l = self._kp_left[kp_idx][0]
        y_l = self._kp_left[kp_idx][1]
        x_r = self._kp_right[kp_idx][0]
        y_r = self._kp_right[kp_idx][1]
        y = (y_l + y_r) / 2
        return x_l, x_r, y

    def get_kp_len(self):
        return len(self._kp_left)

    def add_kp(self, idx_kp, x_l, x_r, y):
        assert (idx_kp in range(len(self._kp_left) + 1))
        if idx_kp == len(self._kp_left):
            self._kp_left.append((x_l, y))
            self._kp_right.append((x_r, y))
        else:
            self._kp_left[idx_kp] = (x_l, y)
            self._kp_right[idx_kp] = (x_r, y)

