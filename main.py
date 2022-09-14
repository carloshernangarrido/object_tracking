import os
import pickle
import matplotlib.pyplot as plt

from utils.object_tracking import main_object_tracking


flags = {'webcam': False,
         'update_roi': True,
         'auto_play': False}
# path = os.curdir
# filename = 'con_AFO_3.mp4'
path = 'C:/TRABAJO/CONICET/videos/'
filename = 'vid_2022-09-13_12-54-44.mp4'
actual_fps = 500  # Ignored if flags['webcam'] == True or if actual_fps is None
start_time_ms = 1000
dump_filename = 'txy.dat'


if __name__ == '__main__':
    full_filename = os.path.join(path, filename)

    txy, txy_refined = main_object_tracking(flags, full_filename, start_time_ms, actual_fps=None)

    # Save to disk
    with open(dump_filename, 'wb') as dump_file:
        saving_list = [txy, txy_refined]
        pickle.dump(saving_list, dump_file)

    plt.plot(txy[:, 0], txy[:, 1])
    plt.plot(txy[:, 0], txy[:, 2])
    plt.plot(txy_refined[:, 0], txy_refined[:, 1])
    plt.plot(txy_refined[:, 0], txy_refined[:, 2])
    plt.show()
    print(txy)


