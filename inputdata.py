import os


flags = {'webcam': False,
         'update_roi': True,
         'auto_play': False,
         'perform_object_tracking': True,
         'perform_multi_tracking': 0,  # 0 to avoid multi_tracking, 3 or more to specify and perform multi-tracking
         'perform_dsp': False}

case = '1000mm_rep1'
video_path = r"C:\TRABAJO\CONICET\videos\2023-08-17"  # r"C:\TRABAJO\CONICET\videos\2022-11-16"
video_filename = "vid_2023-08-17_15-58-00.mp4"  # f'case_{case}.mp4'
actual_fps = 5000  # Ignored if flags['webcam'] == True or if actual_fps is None
start_time_ms = 8
finish_time_ms = None
conf_th = 0.9
ot_output_filename = f'case_{case}_.dat'
ot_output_path = r'C:\Users\joses\Mi unidad\TRABAJO\48_FG_protection\TRABAJO\characterization\Experimental tests\2023-08-17'
ot_output_filename = os.path.join(ot_output_path, ot_output_filename)

# dsp_input_filenames = ['txy.dat']
dsp_input_filenames = ['txy_dof1_m.dat',
                       'txy_dof2_m.dat']
kalman_param = {'freq_of_interest': 100,
                'measurement_error_std': 0.00045*0.5,  # 0.5,
                'override_low_pass_f': 0}  # 0 to perform kalman filter
dsp_param = {'emd_mask_freqs': [7, 2.5],
             'emd_max_imfs': 3}
