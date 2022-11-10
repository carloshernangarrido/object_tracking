import sys
import cv2
import numpy as np
from utils.object_tracking import frame_slice, bbox2rect, MousePts, template_match


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


def main_multi_tracking(flags, full_filename, start_time_ms, finish_time_ms=None, n_points=3, actual_fps=None):
    txytheta = []
    txytheta_refined = []

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
    if not video.isOpened():  # Exit if video not opened.
        print("Could not open video")
        sys.exit()
    # Read first frame.
    ok, frame = video.read()

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
            win_name = f'Select ROI for TrackPoint {i_tp}'
            cv2.namedWindow(win_name, cv2.WINDOW_KEEPRATIO)  # [x, y, with, height]
            tp.update_roi(frame, list(cv2.selectROI(win_name, frame, False)))

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
    time_offset = 0
    src_centroid = None
    src_refined_centroid = None
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
        for tp in tps:
            tp.update_roi(frame)
            tp.match()
            if flags['update_roi']:
                tp.update_bbox_roi()
            tp.save_time_and_point(t=t + time_offset, distance_scale=distance_scale)

        # Affine
        src = np.array([[tp.txy[0][1], tp.txy[0][2]] for tp in tps], dtype=np.float)
        if src_centroid is None:
            src_centroid = np.mean(src, axis=0)
        dst = np.array([[tp.txy[-1][1], tp.txy[-1][2]] for tp in tps], dtype=np.float)
        affine_transform = cv2.estimateAffinePartial2D(src - src_centroid, dst - src_centroid)
        assert np.sum(affine_transform[1]) == n_points, "One or more points have poor quality"
        affine_transform = affine_transform[0]
        xy = [affine_transform[0, 2], affine_transform[1, 2]]
        angle = (180/np.pi)*np.arctan2(affine_transform[1, 0], affine_transform[0, 0])

        src_refined = np.array([[tp.txy_refined[0][1], tp.txy_refined[0][2]] for tp in tps], dtype=np.float)
        if src_refined_centroid is None:
            src_refined_centroid = np.mean(src_refined, axis=0)
        dst_refined = np.array([[tp.txy_refined[-1][1], tp.txy_refined[-1][2]] for tp in tps], dtype=np.float)
        affine_transform_refined = cv2.estimateAffinePartial2D(src_refined - src_refined_centroid,
                                                               dst_refined - src_refined_centroid)
        assert np.sum(affine_transform_refined[1]) == n_points, "One or more points have poor quality"
        affine_transform_refined = affine_transform_refined[0]
        xy_refined = [affine_transform_refined[0, 2], affine_transform_refined[1, 2]]
        angle_refined = (180/np.pi)*np.arctan2(affine_transform_refined[1, 0], affine_transform_refined[0, 0])
        txytheta.append([t + time_offset, xy[0], xy[1], angle])
        txytheta_refined.append([t + time_offset, xy_refined[0], xy_refined[1], angle_refined])

        # Visualization
        dst_refined /= distance_scale
        src_refined /= distance_scale
        # center of mass
        for dst_ in dst_refined:
            cv2.circle(frame, [int(_) for _ in dst_], radius=3, color=(0, 0, 255))
        center_of_mass = [int(_) for _ in np.mean(dst_refined, axis=0)]
        cv2.circle(frame, center_of_mass, radius=10, color=(255, 0, 0))
        cv2.putText(frame, 'dMC', center_of_mass,
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0))
        # src center of mass + delta_xy
        for src_ in src_refined:
            cv2.circle(frame, [int(_) for _ in src_], radius=3, color=(0, 255, 255))
        center_of_mass_src = np.mean(src_refined, axis=0)
        center_of_mass_src_delta_xy = [int(_) for _ in center_of_mass_src +
                                       np.array(txytheta_refined[-1][1:3])/distance_scale]
        cv2.circle(frame, center_of_mass_src_delta_xy, radius=5, color=(0, 255, 0))
        cv2.putText(frame, 'sMC+D', org=center_of_mass_src_delta_xy,
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 0))
        for i_tp, tp in enumerate(tps):
            # Draw bounding box for roi and matched template
            p1, p2 = bbox2rect(tp.bbox_roi)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            p1, p2 = bbox2rect(tp.bbox)
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 1, 1)
            cv2.putText(frame, f'{i_tp}', (tp.bbox[0], tp.bbox[1]),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0))
        cv2.putText(frame, f't = {np.round(t + time_offset, 4)} s',
                    (10, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 0))
        cv2.putText(frame, f'of T = {np.round(total_time, 2)} s',
                    (10, 150), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 0))
        cv2.putText(frame, f'video @ {np.round(video_fps, 1)} fps',
                    (10, 200), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 0))
        cv2.putText(frame, f'actual @ {np.round(actual_fps, 1)} fps',
                    (10, 250), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 0))
        i = 0
        for i in range(len(src_refined)):
            cv2.putText(frame,
                        f's {[np.format_float_positional(_, precision=3, unique=False, fractional=False, trim="k") for _ in src_refined[i]]}, '
                        f'd {[np.format_float_positional(_, precision=3, unique=False, fractional=False, trim="k") for _ in dst_refined[i]]}',
                        (1, 300 + 25*i), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255))
        cv2.putText(frame, f'delta_x {np.round(xy_refined[0], 4)}',
                    (10, 300 + 25*i + 25), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 0, 255))
        cv2.putText(frame, f'delta_y {np.round(xy_refined[1], 4)}',
                    (10, 300 + 25*i + 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 0, 255))
        cv2.putText(frame, f'angle {np.round(angle_refined, 4)}',
                    (10, 300 + 25*i + 75), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0))
        cv2.putText(frame, f'M={affine_transform_refined[0]}',
                    (10, 300 + 25*i + 125), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0))
        cv2.putText(frame, f'M={affine_transform_refined[1]}',
                    (10, 300 + 25 * i + 150), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0))
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

    txytheta = np.array(txytheta)
    txytheta_refined = np.array(txytheta_refined)

    return txytheta, txytheta_refined
