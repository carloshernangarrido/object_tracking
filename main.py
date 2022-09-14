import pickle
import sys
import os
import cv2
import numpy as np

from utils import template_match, frame_slice, bbox2rect
import matplotlib.pyplot as plt


webcam = False
# path = os.curdir
path = 'C:/TRABAJO/CONICET/videos/'
filename = 'vid_2022-09-13_12-54-44.mp4'
actual_fps = 500
start_time_ms = 0
dump_filename = 'txy_dof1.dat'
update_roi = True

if __name__ == '__main__':
    txy = []
    full_filename = os.path.join(path, filename)

    # Read video
    if webcam:
        video = cv2.VideoCapture(0)  # for using CAM
    else:
        video = cv2.VideoCapture(full_filename)
        video.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    time_scale = video_fps/actual_fps
    T = video.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()  # c = frame[y, x, :]
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Select Region Of Interest
    cv2.namedWindow('Select region of interest', cv2.WINDOW_KEEPRATIO)
    bbox_roi = list(cv2.selectROI('Select region of interest', frame, False))  # [x, y, with, height]
    roi = frame_slice(frame, bbox_roi)

    # Select Template
    cv2.namedWindow('Select template to track', cv2.WINDOW_NORMAL)
    bbox_template = cv2.selectROI('Select template to track', roi, True)  # (x, y, with, height)
    bbox_template = list(bbox_template)
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
        bbox_old = bbox
        bbox = template_match(roi, template, bbox_roi, bbox_template, frame, verbose=False)
        # Save time (s) and point (px)
        t = time_scale*video.get(cv2.CAP_PROP_POS_MSEC)/1000
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
        cv2.putText(frame_matching_rect, f'of T = {np.round(T, 2)} s @ {video_fps} fps',
                    (10, 150), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 0))
        cv2.putText(frame_matching_rect, f'video @ {np.round(video_fps,1)} fps',
                    (10, 200), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 0))
        cv2.putText(frame_matching_rect, f'actual @ {np.round(actual_fps,1)} fps',
                    (10, 250), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 0))
        cv2.namedWindow('Frame matching. Press any key to forward or Q to quit.', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Frame matching. Press any key to forward or Q to quit.', frame_matching_rect)
        key = cv2.waitKey()
        if key & 0xFF == ord('q') or key & 0xFF == ord('Q'):
            break

        # Update roi
        if update_roi:
            delta_x, delta_y = bbox[0] - bbox_old[0], bbox[1] - bbox_old[1]
            bbox_roi[0], bbox_roi[1] = bbox_roi[0] + delta_x, bbox_roi[1] + delta_y
            roi = frame_slice(frame, bbox_roi)

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


