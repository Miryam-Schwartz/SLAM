import os
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt

from VAN_ex.code import utils
from VAN_ex.code.DB.DataBase import DataBase

OUTPUT_DIR = 'results\\ex4\\'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def crop_img(img, x, y):
    up_y = int(max(0, y - 50))
    down_y = int(min(len(img), y + 50))
    left_x = int(max(x - 50, 0))
    right_x = int(min(x + 50, len(img[0])))
    img = img[up_y:down_y, left_x:right_x]
    new_x = min(x, 50)
    new_y = min(y, 50)
    return img, new_x, new_y


def show_feature_track(db, num_frames):
    track = db.get_track_in_len(num_frames)
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
        x_left, x_right, y = db.get_frame_obj(frame_id).get_feature_pixels(kp_idx)
        img_left, img_right = utils.read_images(frame_id)
        img_left, x_left, y_left = crop_img(img_left, x_left, y)
        img_right, x_right, y_right = crop_img(img_right, x_right, y)
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
    frames_num = db.get_frames_number()
    outgoing_frames = np.empty(frames_num)
    for i in range(frames_num):
        outgoing_frames[i] = db.get_frame_obj(i).get_number_outgoing_tracks()
    fig = px.scatter(x=np.arange(frames_num), y=outgoing_frames, title="Connectivity",
                     labels={'x': 'Frame', 'y': 'Outgoing tracks'})
    fig.write_image(f"{OUTPUT_DIR}connectivity_graph.png")


def ex4_run():
    db = DataBase()
    db.fill_database(11)
    db.save_database('C:\\Users\\Miryam\\SLAM\\VAN_ex\\code\\DB\\')
    # db = DataBase()
    # db.read_database('C:\\Users\\Miryam\\SLAM\\VAN_ex\\code\\DB\\', 1)
    # db.save_database('C:\\Users\\Miryam\\SLAM\\VAN_ex\\code\\DB\\', 2)

    # 4.2
    print("Total number of tracks: ", db.get_tracks_number())
    print("Number of frames: ", db.get_frames_number())
    mean, max_t, min_t = db.get_mean_max_and_min_track_len()
    print(f"Mean track length: {mean}\nMaximum track length: {max_t}\nMinimum track length: {min_t}")
    print("Mean number of tracks on average frame: ", db.get_mean_number_of_tracks_on_frame())

    # 4.3
    show_feature_track(db, 10)

    # 4.4
    connectivity_graph(db)
    return 0


ex4_run()
