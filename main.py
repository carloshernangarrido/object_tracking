import os
import pickle
import warnings

import matplotlib.pyplot as plt
from utils.object_tracking import main_object_tracking
from utils.multi_tracking import main_multi_tracking
from utils.signal_processing import plot_time_domain, plot_frequency_domain, plot_emd, perform_kalman_filter

flags = {'webcam': False,
         'update_roi': True,
         'auto_play': False,
         'perform_object_tracking': False,
         'perform_multi_tracking': 3,  # 0 to avoid multi_tracking, 3 or more to specify and perform multi-tracking
         'perform_dsp': False}

video_path = 'C:/TRABAJO/CONICET/videos/'
video_filename = 'vid_2022-09-13_12-54-44.mp4'
actual_fps = 500  # Ignored if flags['webcam'] == True or if actual_fps is None
start_time_ms = 0
ot_output_filename = 'txytheta.dat'

# dsp_input_filenames = ['txy.dat']
dsp_input_filenames = ['txy_dof1_m.dat',
                       'txy_dof2_m.dat']
kalman_param = {'freq_of_interest': 10,
                'measurement_error_std': 0.00047*0.5}
dsp_param = {'emd_mask_freqs': [7, 2.5],
             'emd_max_imfs': 3}

if __name__ == '__main__':
    video_full_filename = os.path.join(video_path, video_filename)

    if flags['perform_object_tracking']:
        # object_tracking
        txy, txy_refined = main_object_tracking(flags, video_full_filename, start_time_ms, actual_fps=actual_fps)
        txy_smoothed = perform_kalman_filter(txy_refined, kalman_param)
        # Save to disk
        with open(ot_output_filename, 'wb') as dump_file:
            saving_list = [txy, txy_refined, txy_smoothed]
            pickle.dump(saving_list, dump_file)
        if not flags['perform_dsp']:
            plot_time_domain(txy, txy_refined, txy_smoothed)
            plt.show()

    if flags['perform_multi_tracking'] == 0:
        pass
    elif flags['perform_multi_tracking'] >= 3:
        txytheta, txytheta_refined = main_multi_tracking(flags, video_full_filename, start_time_ms,
                                                         n_points=flags['perform_multi_tracking'], actual_fps=actual_fps)
        ####################################################
        try:
            txytheta_smoothed = perform_kalman_filter(txytheta_refined, kalman_param)
        except AssertionError:
            txytheta_smoothed = txytheta_refined
            warnings.warn('txytheta_smoothed = txytheta_refined')
        # Save to disk
        with open(ot_output_filename, 'wb') as dump_file:
            saving_list = [txytheta, txytheta_refined, txytheta_smoothed]
            pickle.dump(saving_list, dump_file)
        if not flags['perform_dsp']:
            plot_time_domain(txytheta, txytheta_refined, txytheta_smoothed)
            plt.show()
        ####################################################
    else:
        raise ValueError("flags['perform_multi_tracking'] must be 0, 3 or more.")

    if flags['perform_dsp']:
        for dsp_input_filename in dsp_input_filenames:
            figs = []
            with open(dsp_input_filename, 'rb') as load_file:
                saving_list = pickle.load(load_file)
                txy, txy_refined, txy_smoothed = saving_list[0], saving_list[1], saving_list[2]

            plot_time_domain(txy, txy_refined, txy_smoothed=txy_smoothed,
                             title=f'time domain positions for {dsp_input_filename}')
            plot_frequency_domain(txy, txy_refined, txy_smoothed,
                                  title=f'frequency domain displacements {dsp_input_filename}')
            plot_emd(txy_smoothed, mask_freqs=dsp_param['emd_mask_freqs'], max_imfs=dsp_param['emd_max_imfs'],
                     title=f'empirical mode decomposition of positions for {dsp_input_filename}')
        plt.show()
        ...






