import os
import pickle
import matplotlib.pyplot as plt

from utils.object_tracking import main_object_tracking
from utils.signal_processing import plot_time_domain, plot_frequency_domain, plot_emd

flags = {'webcam': False,
         'update_roi': True,
         'auto_play': False,
         'perform_object_tracking': False,
         'perform_dsp': True}
# path = os.curdir
# filename = 'con_AFO_3.mp4'
video_path = 'C:/TRABAJO/CONICET/videos/'
video_filename = 'vid_2022-09-13_12-54-44.mp4'
actual_fps = 500  # Ignored if flags['webcam'] == True or if actual_fps is None
start_time_ms = 1000
dump_filename = 'txy.dat'

dsp_input_filenames = ['txy_dof1_ok.dat',
                       'txy_dof2_ok.dat']

if __name__ == '__main__':
    if flags['perform_object_tracking']:
        # object_tracking
        video_full_filename = os.path.join(video_path, video_filename)
        txy, txy_refined = main_object_tracking(flags, video_full_filename, start_time_ms, actual_fps=None)
        # Save to disk
        with open(dump_filename, 'wb') as dump_file:
            saving_list = [txy, txy_refined]
            pickle.dump(saving_list, dump_file)
        # plot_time_domain(txy, txy_refined)

    if flags['perform_dsp']:
        for dsp_input_filename in dsp_input_filenames:
            figs = []
            with open(dsp_input_filename, 'rb') as load_file:
                saving_list = pickle.load(load_file)
                txy, txy_refined = saving_list[0], saving_list[1]
            plot_time_domain(txy, txy_refined, title=f'time domain positions for {dsp_input_filename}')
            plot_frequency_domain(txy, txy_refined, title=f'frequency domain displacements {dsp_input_filename}')
            plot_emd(txy, txy_refined, title=f'empirical mode decomposition of positions for {dsp_input_filename}')
        plt.show()






