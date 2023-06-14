import os
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objs as go

from VAN_ex.code.DB.Frame import Frame
from VAN_ex.code import utils
from VAN_ex.code.DB.DataBase import DataBase

OUTPUT_DIR = 'results\\ex4\\'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def crop_img(img, x, y, crop_size=50):
    """
    Crop the image around the pixel (x,y) in size of crop_size
    return the cropped image and the new pixel of the same feature in the cropped image
    :param img: opened image in cv
    :param x: x-th coordinate of the feature
    :param y: y-th coordinate of the feature
    :param crop_size: The size to crop around the feature from each direction. 50 by default.
    :return: the cropped image, new x and y (the pixels of the same feature in the cropped image)
    """
    up_y = int(max(0, y - crop_size))
    down_y = int(min(len(img), y + crop_size))
    left_x = int(max(x - crop_size, 0))
    right_x = int(min(x + crop_size, len(img[0])))
    img = img[up_y:down_y, left_x:right_x]
    new_x = min(x, crop_size)
    new_y = min(y, crop_size)
    return img, new_x, new_y


def show_feature_track(db, num_frames):
    """
    Choose random track in len of num_frames at least, create and save an image, that shows the feature
    the first 10 frames of the track.
    :param db: database
    :param num_frames: track length at least to choose
    """
    track = db.get_random_track_in_len(num_frames)
    if track is None:
        raise "there is no track in length >= num_frames"
    fig, grid = plt.subplots(num_frames, 2)
    fig.set_figwidth(4)
    fig.set_figheight(12)
    fig.suptitle(f"Track {track.get_id()}, with length of {num_frames} frames")
    plt.rcParams["font.size"] = "10"
    plt.subplots_adjust(wspace=-0.55, hspace=0.1)
    grid[0, 0].set_title("Left")
    grid[0, 1].set_title("Right")
    i = 0
    for frame_id, kp_idx in track.get_frames_dict().items():
        pixel_left, pixel_right = db.get_frame_obj(frame_id).get_feature_pixels(kp_idx)
        img_left, img_right = utils.read_images(frame_id)
        img_left, x_left, y_left = crop_img(img_left, pixel_left[0], pixel_left[1])
        img_right, x_right, y_right = crop_img(img_right, pixel_right[0], pixel_right[1])
        grid[i, 0].axes.xaxis.set_visible(False)
        grid[i, 0].axes.yaxis.set_label_text(f"frame: {frame_id}")
        grid[i, 0].set_yticklabels([])
        grid[i, 0].imshow(img_left, cmap="gray")
        grid[i, 0].scatter(x_left, y_left, s=10, c="r")
        grid[i, 1].axes.xaxis.set_visible(False)
        grid[i, 1].imshow(img_right, cmap="gray")
        grid[i, 1].scatter(x_right, y_right, s=10, c="r")
        grid[i, 1].set_yticklabels([])
        i = i + 1
        if i == 10:
            break
    fig.savefig(f'{OUTPUT_DIR}track.png')


def connectivity_graph(db):
    """
    Create and save a connectivity graph from the dataa base
    (For each frame, the number of tracks outgoing to the next frame)
    :param db: database
    """
    frames_num = db.get_frames_number()
    outgoing_frames = np.empty(frames_num)
    for i in range(frames_num):
        outgoing_frames[i] = db.get_frame_obj(i).get_number_outgoing_tracks()
    fig = px.line(x=np.arange(frames_num), y=outgoing_frames, title="Connectivity",
                  labels={'x': 'Frame', 'y': 'Outgoing tracks'})
    fig.write_image(f"{OUTPUT_DIR}connectivity_graph.png")


def inliers_percentage_graph(db):
    """
    Create and save inliers percentage graph -
    for each frame, the percentage of inliers out of all the features that were matched.
    Note: only inliers were been saved in the db, then, inliers percentage were calculated
    and saved before removing outliers
    :param db: database
    """
    frames_num = db.get_frames_number()
    inliers_percentage = np.empty(frames_num)
    for i in range(frames_num):
        inliers_percentage[i] = db.get_frame_obj(i).get_inliers_percentage()
    fig = px.line(x=np.arange(frames_num), y=inliers_percentage, title="Inliers percentage per frame",
                  labels={'x': 'Frame', 'y': 'Inliers Percentage'})
    fig.write_image(f"{OUTPUT_DIR}inliers_percentage_graph.png")


def tracks_length_histogram(db):
    """
    Create and save tracks length histogram (how many tracks are in a specific length)
    :param db:
    """
    tracks_number = db.get_tracks_number()
    tracks_length = np.empty(tracks_number)
    for i in range(tracks_number):
        tracks_length[i] = db.get_track_obj(i).get_track_len()
    unique, count = np.unique(tracks_length, return_counts=True)
    fig = px.line(x=unique, y=count, title='Tracks Length Histogram',
                  labels={'x': 'Track length', 'y': 'Track #'})
    fig.write_image(f'{OUTPUT_DIR}tracks_length_histogram.png')


def reprojection_error(db):
    track = db.get_random_track_in_len(10)
    ground_truth_matrices = utils.read_ground_truth_matrices("C:\\Users\\Miryam\\SLAM\\VAN_ex\\dataset\\poses\\05.txt")
    last_frame_id, kp_idx = track.get_last_frame_id_and_kp_idx()
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


def ex4_run():

    # 4.1 Create, fill and serialize the database from the frames images while removing outliers
    db = DataBase()
    db.fill_database(2560)
    db.save_database('C:\\Users\\Miryam\\SLAM\\VAN_ex\\code\\DB\\')

    # 4.2 Present tracking statistics
    print("Total number of tracks: ", db.get_tracks_number())
    print("Number of frames: ", db.get_frames_number())
    mean, max_t, min_t = db.get_mean_max_and_min_track_len()
    print(f"Mean track length: {mean}\nMaximum track length: {max_t}\nMinimum track length: {min_t}")
    print("Mean number of tracks on average frame: ", db.get_mean_number_of_tracks_on_frame())

    # 4.3 Present feature for the 1-st 10 images of a track
    show_feature_track(db, 10)

    # 4.4 Present a connectivity graph: For each frame, the number of tracks outgoing to the next frame
    connectivity_graph(db)

    # 4.5 Present a graph of the percentage of inliers per frame
    inliers_percentage_graph(db)

    # 4.6 Present a track length histogram graph
    tracks_length_histogram(db)

    # 4.7 Present a graph of the reprojection error size (L2 norm) over the track’s images.
    # We’ll define the reprojection error for a given camera as the difference between the projection
    # and the tracked feature location on that camera.
    reprojection_error(db)

    # use the database for localization (as we did in the previous exercise)
    locations = db.get_camera_locations()
    ground_truth_matrices = utils.read_ground_truth_matrices("C:\\Users\\Miryam\\SLAM\\VAN_ex\\dataset\\poses\\05.txt")
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_matrices)
    utils.show_localization(locations, ground_truth_locations, f'{OUTPUT_DIR}localization.png')

    return 0

if __name__ == '__main__':
    ex4_run()
