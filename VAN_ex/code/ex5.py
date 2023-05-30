import gtsam
import numpy as np
import utils
from DB.DataBase import DataBase

def invert_extrinsic_matrix(r_mat, t_vec):
    #origin extrinsic_mat take world coordinate -> return camera/pose coordinate
    #new extrinsic_mat (used by gtsam) take camera/pose coordinate -> return world coordinate
    new_t_vec = -np.transpose(r_mat) @ t_vec
    new_r_mat = r_mat.T
    return new_r_mat, new_t_vec

def reprojection_error_gtsam(db):
    track = db.get_random_track_in_len(10)
    frames_cameras = []
    last_frame_id, last_kp_idx = track.get_last_frame_id_and_kp_idx()
    frames_dict = track.get_frames_dict()
    is_first_frame = True
    k, identation_right_cam = db.get_initial_camera()
    K = gtsam.Cal3_S2Stereo(k[0][0], k[1][1], k[0][1], k[0][2], k[1][2], -identation_right_cam)
    concat_r = np.identity(3)
    concat_t = np.zeros(3)
    for frame_id, kp_idx in frames_dict.items():
        if is_first_frame:
            is_first_frame = False
        else:
            mat = db.get_frame_obj(frame_id).get_left_camera_pose_mat()
            concat_r = mat[:, :3] @ concat_r   # in coordinates of first frame of track
            concat_t = mat[:, :3] @ concat_t + mat[:, 3]
        inv_r, inv_t = invert_extrinsic_matrix(concat_r, concat_t)
        cur_pos3 = gtsam.Pose3(np.hstack((inv_r, inv_t.reshape(3, 1))))
        cur_stereo_camera = gtsam.StreoCamera(cur_pos3, K)
        frames_cameras.append(cur_stereo_camera)

    last_frame_camera = frames_cameras[-1]
    left_pixel, right_pixel = db.get_frame_obj(last_frame_id).get_feature_pixels(last_kp_idx)
    x_l, x_r, y = left_pixel[0], right_pixel[0], (left_pixel[1] + right_pixel[1])/2
    pt_3d = last_frame_camera.backproject(gtsam.StereoPoint2(x_l, x_r, y))   # in coordinates of last frame
    # todo- check which object is returned from backproject
    world_coor_3d_pt = concat_r.T @ (pt_3d - concat_t)
    for cam_frame in frames_cameras:
        pt_2d = cam_frame.project(world_coor_3d_pt)



    pt_3d = db.get_frame_obj(last_frame_id).get_3d_point(kp_idx)  # in coordinates of last frame
    mat = ground_truth_matrices[last_frame_id]
    r, t = mat[:, :3], mat[:, 3]
    world_coor_3d_pt = r.T @ (pt_3d - t)
    track_len = track.get_track_len()
    reprojection_error_left = np.empty(track_len)
    reprojection_error_right = np.empty(track_len)
    frames_dict = track.get_frames_dict()
    i = 0
    for frame_id, kp_idx in frames_dict.items():
        frame_obj = db.get_frame_obj(frame_id)
        extrinsic_camera_mat_right = np.array(ground_truth_matrices[frame_id], copy=True)
        extrinsic_camera_mat_right[0][3] += Frame.INDENTATION_RIGHT_CAM_MAT
        proj_left_pixel = utils.project_3d_pt_to_pixel(Frame.k, ground_truth_matrices[frame_id], world_coor_3d_pt)
        proj_right_pixel = utils.project_3d_pt_to_pixel(Frame.k, extrinsic_camera_mat_right, world_coor_3d_pt)
        pixel_left, pixel_right = frame_obj.get_feature_pixels(kp_idx)
        reprojection_error_left[i] = np.linalg.norm(proj_left_pixel - np.array(pixel_left))
        reprojection_error_right[i] = np.linalg.norm(proj_right_pixel - np.array(pixel_right))
        i += 1
    reprojection_error_left = reprojection_error_left[:len(reprojection_error_left) - 1]
    reprojection_error_right = reprojection_error_right[:len(reprojection_error_right) - 1]

    frames_arr = np.array(list(frames_dict.keys()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frames_arr, y=reprojection_error_left, mode='lines+markers', name='Left error'
                   # title=f"Reprojection error size (L2 norm) over track {track.get_id()} images",
                   # labels={'x': 'Frame', 'y': 'Reprojection error'})
                   ))
    fig.add_trace(
        go.Scatter(x=frames_arr, y=reprojection_error_right, mode='lines+markers', name='Right error'
                   # title=f"Reprojection error size (L2 norm) over track {track.get_id()} images",
                   # labels={'x': 'Frame', 'y': 'Reprojection error'})
                   ))
    fig.update_layout(title=f"Reprojection error size (L2 norm) over track {track.get_id()} images",
                      xaxis_title='Frame id', yaxis_title='Reprojection error')
    fig.write_image(f"{OUTPUT_DIR}reprojection_error.png")

def run_ex5():
    # todo - import cameras file to linux computer
    K, m_left, m_right = utils.read_cameras()
    db.set_initial_camera(K, m_left, m_right)