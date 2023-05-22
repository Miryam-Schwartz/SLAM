class Track:
    tracks_counter = 0

    def __init__(self, *args):
        assert (len(args) == 4 or len(args) == 3)
        self._frames_dict = dict()  # key = id of frame, val = idx of kp
        if len(args) == 4:
            first_frame_id, second_frame_id, first_idx_kp_left, second_idx_kp_left = args
            assert (second_frame_id == first_frame_id + 1)
            self._track_id = Track.tracks_counter
            self._frames_dict[first_frame_id] = first_idx_kp_left
            self._frames_dict[second_frame_id] = second_idx_kp_left
        else:
            track_id, frame_id, idx_kp = args
            self._track_id = track_id
            self._frames_dict[frame_id] = idx_kp
        Track.tracks_counter += 1

    def get_frames_dict(self):
        return self._frames_dict

    def get_id(self):
        return self._track_id

    # def get_last_kp_frame_idx(self):
    #     max_key = max(self._frames_dict.keys())
    #     return self._frames_dict[max_key]      # the last frame and its idx of left kp

    def get_kp_idx_of_frame(self, frame_id):
        assert (frame_id in self._frames_dict)
        return self._frames_dict[frame_id]

    def add_frame(self, frame_id, idx_kp_left):
        max_key = max(self._frames_dict.keys())
        assert (frame_id == max_key + 1), f"max key = {max_key}, frame id = {frame_id}"
        self._frames_dict[frame_id] = idx_kp_left

    def get_track_len(self):
        return len(self._frames_dict)

