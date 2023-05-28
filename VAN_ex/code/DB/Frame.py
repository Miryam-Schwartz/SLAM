from VAN_ex.code import utils


class Frame:
    """
        A class used to represent a Frame

        ...

        Attributes
        ----------
        _frame_id : int
            unique id of the frame (according to image number)
        _kp_left : lst
            List of tuples of pixels. each element in the list is from the shape (x,y),
            and represents feature pixels in the left image of the frame
        _kp_right: lst
            as _kp_left. represents feature pixels in the right image. _kp_left and _kp_right are lists in the
            same length. The i-th kp in _kp_left is matched to the i-th kp in _kp_right.
        _des_left: lst
            List of match descriptors. The i-th element in the list, is the left descriptor of the matching
            between the i-th keypoint in _kp_left to the i-th keypoint in _kp_right.
        _tracks_dict: dict
            Dictionary of all the tracks that are associated to this frame.
            Key is track_id, and value is the track object.
        _inliers_percentage: float
             The percentage of inliers out of all the features that were matched to this frame.
        _left_camera_location: np.array
            [x,y,z] represents the location of left camera of this frame, in relation to the initial position.


        Static
        ----------
        k, m_left, m_right: np. Array
            matrices of left_0 camera
        INDENTATION_RIGHT_CAM_MAT
            the indentation between left and right camera in every frame
        """

    k, m_left, m_right = utils.read_cameras()
    INDENTATION_RIGHT_CAM_MAT = m_right[0][3]

    def __init__(self, frame_id, kp_left, kp_right, des_left):
        self._frame_id = frame_id
        self._kp_left = kp_left  # kp_right and kp_left are correlated after matching
        self._kp_right = kp_right
        self._des_left = des_left
        self._tracks_dict = dict()  # key = track_id, val = track object
        self._inliers_percentage = None
        self._left_camera_location = None  # in left0 coordinates system

    def get_tracks_dict(self):
        return self._tracks_dict

    def get_des_left(self):
        return self._des_left

    def get_3d_point(self, kp_idx):
        """
        calculate by triangulation, the location of the 3-d object in the world,
        according to this frame coordinates system
        :param kp_idx: the index of the keypoint to triangulate
        :return: 3-d point
        """
        pt_4d = utils.triangulation_single_match \
            (Frame.k @ Frame.m_left, Frame.k @ Frame.m_right, self._kp_left[kp_idx], self._kp_right[kp_idx])
        return pt_4d[:3] / pt_4d[3]

    def get_feature_pixels(self, kp_idx):
        """
        :param kp_idx: index of keypoint to return its feature pixels
        :return: (x_l, y_l), (x_r, y_r)
        """
        return self._kp_left[kp_idx], self._kp_right[kp_idx]

    def get_kp_len(self):
        return len(self._kp_left)

    def get_number_of_tracks(self):
        """
        :return: number of tracks that are associated to this frame.
        """
        return len(self._tracks_dict)

    def get_number_outgoing_tracks(self):
        """
        :return: the number of tracks outgoing to the next frame
        (the number of tracks on the frame with links also in the next frame)
        """
        counter = 0
        for track in self._tracks_dict.values():
            if self._frame_id + 1 in track.get_frames_dict():
                counter += 1
        return counter

    def get_left_camera_location(self):
        if self._left_camera_location is None:
            raise "location is None"
        return self._left_camera_location

    def get_inliers_percentage(self):
        if self._inliers_percentage is None:
            raise "inliers_percentage is None"
        return self._inliers_percentage

    def set_inliers_percentage(self, percentage):
        self._inliers_percentage = percentage

    def set_left_camera_location(self, location):
        self._left_camera_location = location

    def add_track(self, new_track):
        """
        add track to the frame
        :param new_track: track object to add
        """
        assert (new_track.get_id() not in self._tracks_dict)
        self._tracks_dict[new_track.get_id()] = new_track

    def add_kp(self, idx_kp, pixel_left, pixel_right):
        """
        Add keypoint to the frame lists (both left and right keypoints lists)
        :param idx_kp: index of keypoint to add
        :param pixel_left:
        :param pixel_right:
        """
        assert (idx_kp == len(self._kp_left))
        self._kp_left.append(pixel_left)
        self._kp_right.append(pixel_right)

