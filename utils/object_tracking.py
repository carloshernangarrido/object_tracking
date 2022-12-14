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
                2 * window_halfsize + 1, 2 * window_halfsize + 1]
    refined_yx = sp.ndimage.center_of_mass(frame_slice(matching, bbox_ref))
    refined_xy = (refined_yx[1] - window_halfsize, refined_yx[0] - window_halfsize)
    bbox_match_refined = (bbox_match[0] + refined_xy[0], bbox_match[1] + refined_xy[1], bbox_match[2], bbox_match[3])

    return bbox_match, bbox_match_refined


def bbox2rect(bbox: Union[List, Tuple]):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    return p1, p2


class MousePts:
    def __init__(self, windowname, img):
        self.windowname = windowname
        self.img1 = img.copy()
        self.img = self.img1.copy()
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(windowname, img)
        self.curr_pt = []
        self.point = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x, y])
            # print(self.point)
            cv2.circle(self.img, (x, y), 5, (0, 255, 0), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.curr_pt = [x, y]
            # print(self.point)

    def getpt(self, count=1, img=None):
        if img is not None:
            self.img = img
        else:
            self.img = self.img1.copy()
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname, self.img)
        cv2.setMouseCallback(self.windowname, self.select_point)
        self.point = []
        while 1:
            cv2.imshow(self.windowname, self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.point) >= count:
                break
            # print(self.point)
        cv2.setMouseCallback(self.windowname, lambda *args: None)
        # cv2.destroyAllWindows()
        return self.point, self.img


def main_object_tracking(flags, full_filename, start_time_ms, finish_time_ms=None, actual_fps=None):
    txy, txy_refined = [], []

    # Read video
    if flags['webcam']:
        video = cv2.VideoCapture(0)  # for using CAM
    else:
        video = cv2.VideoCapture(full_filename)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    actual_fps = video_fps if actual_fps is None or flags['webcam'] else actual_fps
    time_scale = video_fps / actual_fps
    total_time = time_scale * video.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
    video.set(cv2.CAP_PROP_POS_MSEC, start_time_ms / time_scale)
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
    cv2.namedWindow('Select ROI', cv2.WINDOW_KEEPRATIO)
    while bbox_roi[2] == 0 or bbox_roi[3] == 0:
        # [x, y, with, height]
        bbox_roi = list(cv2.selectROI('Select ROI', frame, False))
    roi = frame_slice(frame, bbox_roi)

    # Select scale
    cv2.namedWindow('Select 2 points and enter the scale between them', cv2.WINDOW_NORMAL)
    coordinate_store = MousePts('Select 2 points and enter the scale between them', roi)
    pts, _ = coordinate_store.getpt(2)
    cv2.destroyAllWindows()
    distance = None
    while distance is None:
        try:
            distance = float(input('Enter the distance or 0 to use pixel units: '))
        except ValueError:
            print('Enter a numeric value')
    delta_x = pts[1][0] - pts[0][0]
    delta_y = pts[1][1] - pts[0][1]
    distance_scale = distance/np.sqrt(delta_x ** 2 + delta_y ** 2) if distance != 0 else 1
    print(f'points coordinates: {pts}')
    print(f'{delta_x=}, {delta_y=}')
    print(f'distance scale: {distance_scale} distance units per px')

    # Select Template
    cv2.namedWindow('Select template to track', cv2.WINDOW_NORMAL)
    bbox_template = list(cv2.selectROI('Select template to track', roi, True))  # (x, y, with, height)
    if bbox_template[0] == 0 or bbox_template[1] == 0:
        print('No template was selected.')
        sys.exit()
    bbox_template[0], bbox_template[1] = bbox_template[0] + bbox_roi[0], bbox_template[1] + bbox_roi[1]
    template = frame_slice(frame, bbox_template)

    time_offset = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        t = time_scale * video.get(cv2.CAP_PROP_POS_FRAMES) / video_fps
        t_ms = t * 1000
        if finish_time_ms is None:
            if not ok:
                break
        else:
            if start_time_ms < finish_time_ms:
                if finish_time_ms <= t_ms or not ok:
                    break
            else:  # ring playback
                if not ok:
                    time_offset = time_scale * video.get(cv2.CAP_PROP_POS_FRAMES) / video_fps
                    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = video.read()
                    t = time_scale * video.get(cv2.CAP_PROP_POS_FRAMES) / video_fps
                if finish_time_ms <= t_ms < start_time_ms:
                    break

        # Template matching
        roi = frame_slice(frame, bbox_roi)
        bbox, bbox_refined = template_match(roi, template, bbox_roi, bbox_template)
        # Update roi
        if flags['update_roi']:
            bbox_roi[0], bbox_roi[1] = bbox[0] - bbox_roi[2] // 2 + bbox[2] // 2, bbox[1] - bbox_roi[3] // 2 + bbox[
                3] // 2
            bbox_roi[0] = 0 if bbox_roi[0] < 0 else bbox_roi[0]
            bbox_roi[1] = 0 if bbox_roi[1] < 0 else bbox_roi[1]
            bbox_roi[0] = video_width - bbox_roi[2] - 1 if bbox_roi[0] >= video_width - bbox_roi[2] else bbox_roi[0]
            bbox_roi[1] = video_height - bbox_roi[3] - 1 if bbox_roi[1] >= video_height - bbox_roi[3] else bbox_roi[1]

        # Save time (s) and point (distance_scale*px)
        txy.append([t + time_offset, distance_scale*bbox[0], distance_scale*bbox[1]])
        txy_refined.append([t + time_offset, distance_scale*bbox_refined[0], distance_scale*bbox_refined[1]])

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
        cv2.putText(frame_matching_rect, f'video @ {np.round(video_fps, 1)} fps',
                    (10, 200), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 0))
        cv2.putText(frame_matching_rect, f'actual @ {np.round(actual_fps, 1)} fps',
                    (10, 250), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 0))
        cv2.namedWindow('Frame matching. Press bar to forward, enter to play/pause, or Q to quit.',
                        cv2.WINDOW_KEEPRATIO)
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
