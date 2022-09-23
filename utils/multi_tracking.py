import sys
import cv2
import numpy as np
from utils.object_tracking import frame_slice, bbox2rect, MousePts, template_match


def read_video_frame(video):
    ok, frame = video.read()  # c = frame[y, x, :]
    if not ok:
        print('Cannot read video file')
        sys.exit()
    else:
        return frame


class TrackPoint:
    def __init__(self, frame):
        self.video_width = frame.shape[1]
        self.video_height = frame.shape[0]
        self.txy, self.txy_refined = [], []
        self.bbox_roi = [0, 0, 0, 0]
        self.roi = frame_slice(frame, self.bbox_roi)
        self.bbox_template = [0, 0, 0, 0]
        self.template = frame_slice(frame, self.bbox_template)
        self.bbox = [0, 0, 0, 0]
        self.bbox_refined = [0, 0, 0, 0]

    def update_roi(self, frame, bbox_roi=None):
        if bbox_roi is not None:
            self.bbox_roi = bbox_roi
        self.roi = frame_slice(frame, self.bbox_roi)

    def update_template(self, frame, bbox_template):
        self.bbox_template = bbox_template
        self.template = frame_slice(frame, self.bbox_template)

    def get_txy_ndarrays(self):
        txy = np.array(self.txy)
        txy_refined = np.array(self.txy_refined)
        return txy, txy_refined

    def match(self):
        self.bbox, self.bbox_refined = template_match(self.roi, self.template, self.bbox_roi, self.bbox_template)

    def update_bbox_roi(self):
        self.bbox_roi[0] = self.bbox[0] - self.bbox_roi[2] // 2 + self.bbox[2] // 2
        self.bbox_roi[1] = self.bbox[1] - self.bbox_roi[3] // 2 + self.bbox[3] // 2
        self.bbox_roi[0] = 0 if self.bbox_roi[0] < 0 else self.bbox_roi[0]
        self.bbox_roi[1] = 0 if self.bbox_roi[1] < 0 else self.bbox_roi[1]
        if self.bbox_roi[0] >= self.video_width - self.bbox_roi[2]:
            self.bbox_roi[0] = self.video_width - self.bbox_roi[2] - 1
        if self.bbox_roi[1] >= self.video_height - self.bbox_roi[3]:
            self.bbox_roi[1] = self.video_height - self.bbox_roi[3] - 1

    def save_time_and_point(self, t, distance_scale=1):
        self.txy.append([t, distance_scale * self.bbox[0], distance_scale * self.bbox[1]])
        self.txy_refined.append([t, distance_scale * self.bbox_refined[0], distance_scale * self.bbox_refined[1]])


def main_multi_tracking(flags, full_filename, start_time_ms, n_points, actual_fps=None):
    # Read video
    if flags['webcam']:
        video = cv2.VideoCapture(0)  # for using CAM
    else:
        video = cv2.VideoCapture(full_filename)
        video.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    actual_fps = video_fps if actual_fps is None or flags['webcam'] else actual_fps
    time_scale = video_fps / actual_fps
    total_time = time_scale * video.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
    if not video.isOpened():  # Exit if video not opened.
        print("Could not open video")
        sys.exit()
    # Read first frame.
    frame = read_video_frame(video)

    # Select scale
    win_name = 'Select 2 points and enter the scale between them, or double click for unit scale'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    coordinate_store = MousePts(win_name, frame)
    pts, _ = coordinate_store.getpt(2)
    cv2.destroyAllWindows()
    delta_x = pts[1][0] - pts[0][0]
    delta_y = pts[1][1] - pts[0][1]
    if delta_y == 0 and delta_x == 0:
        distance_scale = 1
    else:
        distance = None
        while distance is None:
            try:
                distance = float(input('Enter the distance or 0 to use pixel units: '))
            except ValueError:
                print('Enter a numeric value')
        distance_scale = distance/np.sqrt(delta_x ** 2 + delta_y ** 2) if distance != 0 else 1
    print(f'points coordinates: {pts}')
    print(f'{delta_x=}, {delta_y=}')
    print(f'distance scale: {distance_scale} distance units per px')

    # Create track point objects
    tps = [TrackPoint(frame) for _ in range(n_points)]
    for i_tp, tp in enumerate(tps):
        # Select Region Of Interest
        while tp.bbox_roi[2] == 0 or tp.bbox_roi[3] == 0:
            win_name = f'Select ROI for TrackPoint {i_tp}, or press any key to forward'
            cv2.namedWindow(win_name, cv2.WINDOW_KEEPRATIO)  # [x, y, with, height]
            tp.update_roi(frame, list(cv2.selectROI(win_name, frame, False)))
            # Read a new frame
            frame = read_video_frame(video)

        # Select Template
        win_name = f'Select template to track for TrackPoint {i_tp}'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        bbox_template = list(cv2.selectROI(win_name, tp.roi, True))  # (x, y, with, height)
        if bbox_template[0] == 0 or bbox_template[1] == 0:
            print('No template was selected.')
            sys.exit()
        bbox_template[0] = bbox_template[0] + tp.bbox_roi[0]
        bbox_template[1] = bbox_template[1] + tp.bbox_roi[1]
        tp.update_template(frame, bbox_template)

    # Main loop
    while True:
        # Read a new frame
        frame = read_video_frame(video)

        # Template matching
        t = 0
        for tp in tps:
            tp.update_roi(frame)
            tp.match()
            if flags['update_roi']:
                tp.update_bbox_roi()
            t = time_scale*video.get(cv2.CAP_PROP_POS_MSEC)/1000
            tp.save_time_and_point(t=t, distance_scale=distance_scale)

        # Affine
        src = np.array([[tp.txy_refined[0][1], tp.txy_refined[0][2]] for tp in tps], dtype=np.float)
        dst = np.array([[tp.txy_refined[-1][1], tp.txy_refined[-1][2]] for tp in tps], dtype=np.float)
        affine_transform = cv2.estimateAffine2D(src, dst)[0]
        angle = (180/np.pi)*np.arctan2(affine_transform[1, 0], affine_transform[0, 0])

        for i_tp, tp in enumerate(tps):
            # Draw bounding box for roi and matched template
            p1, p2 = bbox2rect(tp.bbox_roi)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            p1, p2 = bbox2rect(tp.bbox)
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 1, 1)
            cv2.putText(frame, f'{i_tp}', (tp.bbox[0], tp.bbox[1]),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0))

        # Display current and total time
        cv2.putText(frame, f't = {np.round(t, 4)} s',
                    (10, 100), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))
        cv2.putText(frame, f'of T = {np.round(total_time, 2)} s',
                    (10, 150), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))
        cv2.putText(frame, f'video @ {np.round(video_fps, 1)} fps',
                    (10, 200), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))
        cv2.putText(frame, f'actual @ {np.round(actual_fps, 1)} fps',
                    (10, 250), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0))
        for i, (p_src, p_dst) in enumerate(zip(src, dst)):
            cv2.putText(frame, f'src {[round(_,1) for _ in p_src]}, dst {[round(_,1) for _ in p_dst]}',
                        (10, 300 + 25*i), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))
        cv2.putText(frame, f'affine_transform {affine_transform[0].round(1)}',
                    (10, 375), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))
        cv2.putText(frame, f'affine_transform {affine_transform[1].round(1)}',
                    (10, 400), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))
        cv2.putText(frame, f'angle {angle}',
                    (10, 450), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))
        win_name = 'Frame matching. Press bar to forward, enter to play/pause, or Q to quit.'
        cv2.namedWindow(win_name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(win_name, frame)

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

    txy_list = [tp.get_txy_ndarrays()[0] for tp in tps]
    txy_refined_list = [tp.get_txy_ndarrays()[1] for tp in tps]

    return txy_list[0], txy_refined_list[0]
