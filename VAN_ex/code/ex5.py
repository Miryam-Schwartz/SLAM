import os
import gtsam
import numpy as np
import utils
import plotly.graph_objs as go
from DB.DataBase import DataBase

OUTPUT_DIR = 'results\\ex5\\'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def invert_extrinsic_matrix(r_mat, t_vec):
    # origin extrinsic_mat take world coordinate -> return camera/pose coordinate
    # new extrinsic_mat (used by gtsam) take camera/pose coordinate -> return world coordinate
    new_t_vec = -np.transpose(r_mat) @ t_vec
    new_r_mat = r_mat.T
    return new_r_mat, new_t_vec


def get_stereo_point2(db, frame_id, kp_idx):
    left_pixel, right_pixel = db.get_frame_obj(frame_id).get_feature_pixels(kp_idx)
    x_l, x_r, y = left_pixel[0], right_pixel[0], (left_pixel[1] + right_pixel[1]) / 2
    return gtsam.StereoPoint2(x_l, x_r, y)


def reprojection_error_gtsam(db):
    track = db.get_random_track_in_len(10)
    track_len = track.get_track_len()
    frames_cameras = []
    real_pts_2d = []
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
            concat_r = mat[:, :3] @ concat_r  # in coordinates of first frame of track
            concat_t = mat[:, :3] @ concat_t + mat[:, 3]
        inv_r, inv_t = invert_extrinsic_matrix(concat_r, concat_t)
        cur_pos3 = gtsam.Pose3(np.hstack((inv_r, inv_t.reshape(3, 1))))
        cur_stereo_camera = gtsam.StereoCamera(cur_pos3, K)
        frames_cameras.append(cur_stereo_camera)
        real_pts_2d.append(get_stereo_point2(db, frame_id, kp_idx))

    last_frame_camera = frames_cameras[-1]
    last_frame_pt_2d = get_stereo_point2(db, last_frame_id, last_kp_idx)
    # triangulation - 3d_pt in coordinates of first frame in track
    pt_3d = last_frame_camera.backproject(last_frame_pt_2d)
    reprojection_error = np.empty(track_len)
    factor_error = np.empty(track_len)
    for i, cam_frame in enumerate(frames_cameras):
        pt_2d_proj = cam_frame.project(pt_3d)
        pt_2d_real = real_pts_2d[i]
        diff = pt_2d_real - pt_2d_proj
        diff = np.array([diff.uL(), diff.uR(), diff.v()])
        reprojection_error[i] = np.linalg.norm(diff)

        factor = gtsam.GenericStereoFactor3D\
            (pt_2d_real, cov, gtsam.symbol('c', i), gtsam.symbol('q', track.get_id()), K)

    frames_arr = np.array(list(frames_dict.keys()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frames_arr, y=reprojection_error, mode='lines+markers', name='error'))
    fig.update_layout(title=f"Reprojection error size (L2 norm) over track {track.get_id()} images",
                      xaxis_title='Frame id', yaxis_title='Reprojection error')
    fig.write_image(f"{OUTPUT_DIR}reprojection_error.png")




def run_ex5():
    db = DataBase()
    db.read_database(utils.DB_PATH)

    # 5.1
    reprojection_error_gtsam(db)

    # db.save_database(utils.DB_PATH, 2)
    return 0


run_ex5()
