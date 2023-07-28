import os
import time

from VAN_ex.code import utils
from VAN_ex.code.DB.DataBase import DataBase
import numpy as np
from matplotlib import pyplot as plt

OUTPUT_DIR = 'results/compare_feature_detector/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def present_db_statistics(db, title):
    print(f'--- {title} detector ---')
    print("Total number of tracks: ", db.get_tracks_number())
    print("Number of frames: ", db.get_frames_number())
    mean, max_t, min_t = db.get_mean_max_and_min_track_len()
    print(f"Mean track length: {mean}\nMaximum track length: {max_t}\nMinimum track length: {min_t}")


def get_inliers_percentage_per_frame(db):
    frames_num = db.get_frames_number()
    inliers_percentage = np.empty(frames_num)
    for i in range(frames_num):
        inliers_percentage[i] = db.get_frame_obj(i).get_inliers_percentage()
    return inliers_percentage


if __name__ == '__main__':
    frames_num = 2560
    # SIFT
    start = time.time()
    db_sift = DataBase('SIFT')
    db_sift.fill_database(frames_num)
    end = time.time()
    sift_time = end - start

    #ORB
    #start = time.time()
    #db_orb = DataBase('ORB')
    #db_orb.fill_database(frames_num)
    #end = time.time()
    #orb_time = end - start

    #AKAZE
    start = time.time()
    db_akaze = DataBase('AKAZE')
    db_akaze.fill_database(frames_num)
    end = time.time()
    akaze_time = end - start

    #BRIEF
    start = time.time()
    db_brief = DataBase('BRIEF')
    db_brief.fill_database(frames_num)
    end = time.time()
    brief_time = end - start

    inliers_percentage_per_frame_sift = get_inliers_percentage_per_frame(db_sift)
    present_db_statistics(db_sift, 'SIFT')
    #inliers_percentage_per_frame_orb = get_inliers_percentage_per_frame(db_orb)
    #present_db_statistics(db_orb, 'ORB')
    inliers_percentage_per_frame_akaze = get_inliers_percentage_per_frame(db_akaze)
    present_db_statistics(db_akaze, 'AKAZE')
    inliers_percentage_per_frame_brief = get_inliers_percentage_per_frame(db_brief)
    present_db_statistics(db_brief, 'BRIEF')

    # compare inliers percentage
    fig = plt.figure()
    plt.title("Inliers percentage with different feature detectors")
    plt.plot(np.arange(frames_num), inliers_percentage_per_frame_sift, label="SIFT")
    #plt.plot(np.arange(frames_num), inliers_percentage_per_frame_orb, label="ORB")
    plt.plot(np.arange(frames_num), inliers_percentage_per_frame_akaze, label="AKAZE")
    plt.plot(np.arange(frames_num), inliers_percentage_per_frame_brief, label="BRIEF")
    plt.ylabel("Inliers percentage")
    plt.xlabel("frameID")
    plt.legend()
    fig.savefig(f'{OUTPUT_DIR}compare_inliers_percentage.png')
    plt.close(fig)

    # show pnp localization
    ground_truth_matrices = utils.read_ground_truth_matrices(utils.GROUND_TRUTH_PATH)
    ground_truth_locations = utils.calculate_ground_truth_locations_from_matrices(ground_truth_matrices)

    sift_locations = db_sift.get_camera_locations()
    #orb_locations = db_orb.get_camera_locations()
    akaze_locations = db_akaze.get_camera_locations()
    brief_locations = db_brief.get_camera_locations()

    fig, ax = plt.subplots(figsize=(19.2, 14.4))
    ax.scatter(x=ground_truth_locations[:frames_num, 0], y=ground_truth_locations[:frames_num, 2],
               label='Ground Truth', s=30, alpha=0.4, c='tab:gray')
    ax.scatter(x=sift_locations[:, 0], y=sift_locations[:, 2],
               label='SIFT', s=30, alpha=0.7, c='tab:blue')
    #ax.scatter(x=orb_locations[:, 0], y=orb_locations[:, 2],
    #           label='ORB', s=30, alpha=0.7, c='tab:orange')
    ax.scatter(x=akaze_locations[:, 0], y=akaze_locations[:, 2],
               label='AKAZE', s=30, alpha=0.7, c='tab:orange')
    ax.scatter(x=brief_locations[:, 0], y=brief_locations[:, 2],
               label='BRIEF', s=30, alpha=0.7, c='tab:green')
    ax.legend(fontsize='20')
    ax.set_title('Loclization Trajectory - comparing feature detectors - bird eye view')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(f'{OUTPUT_DIR}localization_comparing_feature_detectors.png')

    # calculate PnP localization error
    gt_locations_sliced = ground_truth_locations[:frames_num]
    sift_error = np.sum(np.abs(gt_locations_sliced - sift_locations) ** 2, axis=-1) ** 0.5
    #orb_error = np.sum(np.abs(gt_locations_sliced - orb_locations) ** 2, axis=-1) ** 0.5
    akaze_error = np.sum(np.abs(gt_locations_sliced - akaze_locations) ** 2, axis=-1) ** 0.5
    brief_error = np.sum(np.abs(gt_locations_sliced - brief_locations) ** 2, axis=-1) ** 0.5

    fig = plt.figure()
    plt.title("Compare estimation location error with different feature detectors")
    plt.plot(np.arange(frames_num), sift_error, label="SIFT")
    #plt.plot(np.arange(frames_num), orb_error, label="ORB")
    plt.plot(np.arange(frames_num), akaze_error, label="AKAZE")
    plt.plot(np.arange(frames_num), brief_error, label="BRIEF")
    plt.ylabel("location error")
    plt.xlabel("frameID")
    plt.legend()
    fig.savefig(f'{OUTPUT_DIR}compare_location_error.png')
    plt.close(fig)

    # print runtime
    print(f"SIFT runtime: {sift_time}")
    #print(f"ORB runtime: {orb_time}")
    print(f"AKAZE runtime: {akaze_time}")
    print(f"BRIEF runtime: {brief_time}")