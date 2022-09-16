import os
import pickle
import matplotlib.pyplot as plt
from utils.object_tracking import main_object_tracking
from utils.signal_processing import plot_time_domain, plot_frequency_domain, plot_emd, perform_kalman_filter

flags = {'webcam': False,
         'update_roi': True,
         'auto_play': False,
         'perform_object_tracking': None,
         'perform_dsp': True}

video_path = 'C:/TRABAJO/CONICET/videos/'
video_filename = 'vid_2022-09-13_12-54-44.mp4'
actual_fps = 500  # Ignored if flags['webcam'] == True or if actual_fps is None
start_time_ms = 0
ot_output_filename = 'txy.dat'

dsp_input_filenames = ['txy_dof1_ok.dat',
                       'txy_dof2_ok.dat']
kalman_param = {'freq_of_interest': 1,
                'measurement_error_std': 1}
dsp_param = {'emd_mask_freqs': [7, 2.5],
             'emd_max_imfs': 3}

if __name__ == '__main__':
    if flags['perform_object_tracking']:
        # object_tracking
        video_full_filename = os.path.join(video_path, video_filename)
        txy, txy_refined = main_object_tracking(flags, video_full_filename, start_time_ms, actual_fps=actual_fps)
        # Save to disk
        with open(ot_output_filename, 'wb') as dump_file:
            saving_list = [txy, txy_refined]
            pickle.dump(saving_list, dump_file)
        if not flags['perform_dsp']:
            plot_time_domain(txy, txy_refined)
            plt.show()

    if flags['perform_dsp']:
        for dsp_input_filename in dsp_input_filenames:
            figs = []
            with open(dsp_input_filename, 'rb') as load_file:
                saving_list = pickle.load(load_file)
                txy, txy_refined = saving_list[0], saving_list[1]

            txy_smoothed = perform_kalman_filter(txy_refined, kalman_param)

            plot_time_domain(txy, txy_refined, txy_smoothed=txy_smoothed,
                             title=f'time domain positions for {dsp_input_filename}')

            plot_frequency_domain(txy, txy_refined, txy_smoothed,
                                  title=f'frequency domain displacements {dsp_input_filename}')
            plot_emd(txy_smoothed, mask_freqs=dsp_param['emd_mask_freqs'], max_imfs=dsp_param['emd_max_imfs'],
                     title=f'empirical mode decomposition of positions for {dsp_input_filename}')
        plt.show()
        a = 0






