
class Track:
    """
        A class used to represent a Track (matched feature between frames)

        ...

        Attributes
        ----------
        _track_id : int
            unique id of the track
        _frames_dict : dict
            a dictionary of all the frames that are associated with this track.
            the key is frame_id and the value is the idx of the tracked keypoint in its frame.

        Static
        ----------
        tracks_counter : int
            number of track objects that were created till now.
        """
    tracks_counter = 0

    def __init__(self, *args):
        """
        :param args: There are two options of constructing new track:
            1. while creating the database from frames images and finding new match, then args should be:
                args = (first_frame_id, second_frame_id, first_idx_kp_left, second_idx_kp_left)
            2. while reading the database from a file line by line, at first we create the track with only one frame
                args = (track_id, frame_id, idx_kp)
        """
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

    def get_last_frame_id_and_kp_idx(self):
        """
        :return: id of last frame of the track, and its idx of the left keypoint
        """
        max_key = max(self._frames_dict.keys())
        return max_key, self._frames_dict[max_key]  # the last frame and its idx of left kp

    def get_first_frame_id(self):
        return min(self._frames_dict.keys())

    def get_kp_idx_of_frame(self, frame_id):
        """
        :param frame_id:
        :return: index of the keypoint of specific frame from the track
        """
        assert (frame_id in self._frames_dict)
        return self._frames_dict[frame_id]

    def get_track_len(self):
        return len(self._frames_dict)

    def add_frame(self, frame_id, idx_kp_left):
        """
        Add new frame to the track
        :param frame_id: id of frame to add to the track (should be successive to the last frame in the track)
        :param idx_kp_left: index of keypoint in the frame that associated to this track
        """
        max_key = max(self._frames_dict.keys())
        assert (frame_id == max_key + 1), f"max key = {max_key}, frame id = {frame_id}"
        self._frames_dict[frame_id] = idx_kp_left


