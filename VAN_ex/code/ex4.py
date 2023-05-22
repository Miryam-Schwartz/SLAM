from matplotlib import pyplot as plt

from VAN_ex.code import utils
from VAN_ex.code.DB.DataBase import DataBase

def show_feature_track(db, num_frames):
    track = db.get_track_in_len(num_frames)
    if track is None:
        raise "there is no track in length >= num_frames"
    for frame_id, kp_idx in track.get_frames_dict().items():
        x_left, x_right, y = db.get_frame_obj(frame_id).get_feature_pixels(kp_idx)
        mark_pixel_on_img(frame_id, x_left, x_right, y)

def mark_pixel_on_img(frame_id, x_left, x_right, y):
    img_name = '{:06d}.png'.format(frame_id)
    img_left = plt.imread(utils.DATA_PATH + 'image_0\\' + img_name)
    img_right = plt.imread(utils.DATA_PATH + 'image_1\\' + img_name)
    size = 50  # size of marker

    plt.imshow(img_left, cmap="gray")  # plot image
    plt.scatter(x_left, y, size, c="r")  # plot markers
    plt.show()



def ex4_run():
    db = DataBase()
    db.fill_database(200)
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
    return 0

ex4_run()