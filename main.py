import pickle
import sys
import os
import cv2
import numpy as np

from utils import template_match, frame_slice, bbox2rect
import matplotlib.pyplot as plt


flags = {'webcam': False,
         'update_roi': True,
         'auto_play': False}
# path = os.curdir
# filename = 'con_AFO_3.mp4'
path = 'C:/TRABAJO/CONICET/videos/'
filename = 'vid_2022-09-13_12-54-44.mp4'
actual_fps = 500  # Ignored if flags['webcam'] == True or if actual_fps is None
start_time_ms = 0
dump_filename = 'txy_dof1.dat'


if __name__ == '__main__':
    txy = []
    full_filename = os.path.join(path, filename)

    # Read video
    if flags['webcam']:
        video = cv2.VideoCapture(0)  # for using CAM
    else:
        video = cv2.VideoCapture(full_filename)
        video.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    actual_fps = video_fps if actual_fps is None or flags['webcam'] else actual_fps
    time_scale = video_fps/actual_fps
    total_time = time_scale * video.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()  # c = frame[y, x, :]
    if not ok:
        print('Cannot read video file')
        sys.exit()
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Select Region Of Interest
    bbox_roi = [0, 0, 0, 0]
    while bbox_roi[2] == 0 or bbox_roi[3] == 0:
        cv2.namedWindow('Select ROI, or press any key to forward', cv2.WINDOW_KEEPRATIO)
        bbox_roi = list(cv2.selectROI('Select ROI, or press any key to forward', frame, False))  # [x, y, with, height]
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
    roi = frame_slice(frame, bbox_roi)

    # Select Template
    cv2.namedWindow('Select template to track', cv2.WINDOW_NORMAL)
    bbox_template = list(cv2.selectROI('Select template to track', roi, True))  # (x, y, with, height)
    if bbox_template[0] == 0 or bbox_template[1] == 0:
        print('No template was selected.')
        sys.exit()
    bbox_template[0], bbox_template[1] = bbox_template[0] + bbox_roi[0], bbox_template[1] + bbox_roi[1]
    template = frame_slice(frame, bbox_template)

    bbox = bbox_template
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Template matching
        roi = frame_slice(frame, bbox_roi)
        bbox = template_match(roi, template, bbox_roi, bbox_template, frame, verbose=False)
        # Update roi
        if flags['update_roi']:
            bbox_roi[0], bbox_roi[1] = bbox[0] - bbox_roi[2]//2 + bbox[2]//2, bbox[1] - bbox_roi[3]//2 + bbox[3]//2
            bbox_roi[0] = 0 if bbox_roi[0] < 0 else bbox_roi[0]
            bbox_roi[1] = 0 if bbox_roi[1] < 0 else bbox_roi[1]
            bbox_roi[0] = video_width - bbox_roi[2] - 1 if bbox_roi[0] >= video_width - bbox_roi[2] else bbox_roi[0]
            bbox_roi[1] = video_height - bbox_roi[3] - 1 if bbox_roi[1] >= video_height - bbox_roi[3] else bbox_roi[1]
            roi = frame_slice(frame, bbox_roi)
        # Save time (s) and point (px)
        t = time_scale*video.get(cv2.CAP_PROP_POS_MSEC)/1000
        if t == 0:
            break
        txy.append([t, bbox[0], bbox[1]])

        # Draw bounding box for roi
        p1, p2 = bbox2rect(bbox_roi)
        frame_matching_rect = cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        # Draw bounding box for matched template
        p1, p2 = bbox2rect(bbox)
        frame_matching_rect = cv2.rectangle(frame_matching_rect, p1, p2, (0, 255, 0), 1, 1)
        # Display current and total time
        cv2.putText(frame_matching_rect, f't = {np.round(t, 4)} s',
                    (10, 100), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 0))
        cv2.putText(frame_matching_rect, f'of T = {np.round(total_time, 2)} s',
                    (10, 150), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 0))
        cv2.putText(frame_matching_rect, f'video @ {np.round(video_fps,1)} fps',
                    (10, 200), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 0))
        cv2.putText(frame_matching_rect, f'actual @ {np.round(actual_fps,1)} fps',
                    (10, 250), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 0))
        cv2.namedWindow('Frame matching. Press bar to forward, enter to play, or Q to quit.', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Frame matching. Press bar to forward, enter to play, or Q to quit.', frame_matching_rect)

        if flags['auto_play']:
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey()
        if key & 0xFF == ord('q') or key & 0xFF == ord('Q'):
            break
        elif key == ord('\r'):
            flags['auto_play'] = True

    video.release()
    cv2.destroyAllWindows()

    txy = np.array(txy)

    # Save to disk
    with open(dump_filename, 'wb') as dump_file:
        pickle.dump(txy, dump_file)

    plt.plot(txy[:, 0], txy[:, 1])
    plt.plot(txy[:, 0], txy[:, 2])
    plt.show()
    print(txy)


