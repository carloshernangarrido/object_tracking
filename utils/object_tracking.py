import cv2
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
