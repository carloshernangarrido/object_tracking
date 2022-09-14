import sys
import cv2
import numpy as np
import scipy as sp
from typing import List, Union, Tuple


def frame_slice(frame_to_slice, bbox):
    if frame_to_slice.ndim == 2:
        return frame_to_slice[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
    elif frame_to_slice.ndim == 3:
        return frame_to_slice[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2]), :]
    else:
        raise ValueError('frame_to_slice must be 2D or 3D ndarray')


def template_match(roi, template, bbox_roi, bbox_template):
    matching = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(matching)
    bbox_match = (max_loc[0] + bbox_roi[0], max_loc[1] + bbox_roi[1], bbox_template[2], bbox_template[3])

    # Refinement
    window_halfsize = 10
    bbox_ref = [max_loc[0] - window_halfsize, max_loc[1] - window_halfsize,
                2*window_halfsize + 1, 2*window_halfsize + 1]
    refined_yx = sp.ndimage.center_of_mass(frame_slice(matching, bbox_ref))
    refined_xy = (refined_yx[1] - window_halfsize, refined_yx[0] - window_halfsize)
    bbox_match_refined = (bbox_match[0] + refined_xy[0], bbox_match[1] + refined_xy[1], bbox_match[2], bbox_match[3])

    return bbox_match, bbox_match_refined


def bbox2rect(bbox: Union[List, Tuple]):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    return p1, p2


def main_object_tracking(flags, full_filename, start_time_ms, actual_fps=None):
    txy, txy_refined = [], []

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
        # [x, y, with, height]
        bbox_roi = list(cv2.selectROI('Select ROI, or press any key to forward', frame, False))
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
        bbox, bbox_refined = template_match(roi, template, bbox_roi, bbox_template)
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
        txy_refined.append([t, bbox_refined[0], bbox_refined[1]])

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
        cv2.namedWindow('Frame matching. Press bar to forward, enter to play/pause, or Q to quit.', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Frame matching. Press bar to forward, enter to play/pause, or Q to quit.', frame_matching_rect)

        if flags['auto_play']:
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey()
        if key & 0xFF == ord('q') or key & 0xFF == ord('Q'):
            break
        elif key == ord('\r'):
            flags['auto_play'] = not flags['auto_play']

    video.release()
    cv2.destroyAllWindows()

    txy = np.array(txy)
    txy_refined = np.array(txy_refined)

    return txy, txy_refined
