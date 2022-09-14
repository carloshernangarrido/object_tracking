import cv2
from typing import List, Union, Tuple


def frame_slice(frame_to_slice, bbox):
    return frame_to_slice[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2]), :]


def template_match(roi, template, bbox_roi, bbox_template, frame, verbose: bool = False):
    matching = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(matching)
    bbox_match = (max_loc[0] + bbox_roi[0], max_loc[1] + bbox_roi[1], bbox_template[2], bbox_template[3])
    if verbose:
        frame_matching_rect = cv2.rectangle(frame,
                                            bbox_match[0:2],
                                            (bbox_match[0] + bbox_template[2],
                                             bbox_match[1] + bbox_template[3]),
                                            (255, 0, 0), 1)
        cv2.imshow('frame matching', frame_matching_rect)
        cv2.waitKey()
        # TODO: Matching refining
    return bbox_match


def bbox2rect(bbox: Union[List, Tuple]):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    return p1, p2
