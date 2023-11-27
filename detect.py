"""
A Python script which provides functionality for detecting a SafeTown lane
from live video feed from a calibrated webcam (see calibrate.py). This is
not intended as a standalone program, although, if ran, it will run tests.

Author: Andrew Huffman
"""

import json
import functools
import numpy as np
import cv2

SETTINGS_FILE = "settings.json"
WHITE_TAPE_HSV = (0, 0, 255)
YELLOW_TAPE_HSV = (30, 85, 255)
LEN_CENTROID_SLICE = 10 # The number of points in a horizontal centroid slice,
                        # see find_center_points
ENTER_KEY = 13 # The keycode for the enter key. Used in the test routine.


class Camera:
    """A calibrated camera for use with SafeTown."""

    class CamReadException(Exception):
        """Raised when the camera fails to read."""

    def __init__(self, cam_settings: dict):
        self.cam = cv2.VideoCapture(cam_settings["cam_device"])

        # Use a sample frame to determine the resolution of the camera
        self.rows, self.cols, _ = self.cam.read()[1].shape

        # Load the matrix used for bird's-eye view warping
        warping_params = cam_settings["cam_warp_calibration"]
        input_pts = np.float32([warping_params['top_left'],
                                warping_params['top_right'],
                                warping_params['bottom_left'],
                                warping_params['bottom_right']])
        output_pts = np.float32([(0, 0), (self.cols, 0), (0, self.rows), (self.cols, self.rows)])
        self.warping_matrix = cv2.getPerspectiveTransform(input_pts, output_pts)

    def get_frame(self):
        """Gets a frame from the camera, or excepts (CamReadException) if it fails."""
        res, frame = self.cam.read()

        if res:
            return frame
        raise Camera.CamReadException

    def warp_frame(self, frame: 'cv2.MatLike'):
        """Copies the frame and warps it based upon the warping parameters, if
        they have been given. Otherwise, no warping is done."""

        return cv2.warpPerspective(frame.copy(), self.warping_matrix, (self.cols, self.rows))

def get_lane_marker_mask(
    frame: 'cv2.MatLike',
    inside_color=YELLOW_TAPE_HSV,
    outside_color=WHITE_TAPE_HSV):
    """Gets the binary mask which corresponds to the lane markers (center dotted line and outside
    line), by color in HSV."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def apply_tolerance(val, tolerance, max_val):
        return (
            0 if val - tolerance < 0 else val - tolerance,
            max_val if val + tolerance > max_val else val + tolerance
        )

    def to_thruple(x, y, z):
        return ((x[0], y[0], z[0]), (x[1], y[1], z[1]))

    # Apply Thresholding for the outside tape.
    lower_outside, higher_outside = to_thruple(
        apply_tolerance(outside_color[0], 360, 360),
        apply_tolerance(outside_color[1], 10, 255),
        apply_tolerance(outside_color[2], 10, 255))
    mask_outside = cv2.inRange(hsv, lower_outside, higher_outside)

    # Apply Thresholding for the inside tape.
    lower_inside, higher_inside = to_thruple(
        apply_tolerance(inside_color[0], 10, 360),
        apply_tolerance(inside_color[1], 10, 255),
        apply_tolerance(inside_color[2], 10, 255))
    mask_inside = cv2.inRange(hsv, lower_inside, higher_inside)

    # Merge the masks.
    mask = mask_outside | mask_inside

    return mask

def find_center_points(marker_mask, dot_num):
    """Finds the center point between each dot in the middle line and the right line."""
    points = []
    # Find the contours in the image.
    contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the right line -- this will always be the largest contour.
    contours = list(contours)
    contours.sort(key=cv2.contourArea)
    outside_line = list(contours.pop())

    # For each contour which isn't the right line, find the shortest line between the two,
    # and add the center to the result -- this will be the center of the lane.
    for _ in range(dot_num):
        contour = contours.pop()

        # Find the centroid of the contour.
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            mid_centroid_x = int(moments["m10"] / moments["m00"])
            mid_centroid_y = int(moments["m01"] / moments["m00"])
        else:
            mid_centroid_x, mid_centroid_y = 0, 0

        # Find the centroid of the slice of the outside line which is similar
        # in the y coordinate to the center line centroid.
        def get_distance(val, rule):
            return val[0, 0] - rule
        # This has to be done because the normal lambda would freeze the centroid value.
        # The partial function solves the issue, according to the pylint documentation,
        # because the value of 'rule' is determined at runtime, and not determined using
        # lazy evalutation.
        # More details here:
        # https://pylint.readthedocs.io/en/latest/user_guide/messages/warning/cell-var-from-loop.html
        outside_line.sort(key=functools.partial(get_distance, rule=mid_centroid_y))
        sub_c = outside_line[0:LEN_CENTROID_SLICE]


    return points

def _run_test():
    """Tests the lane detection with imshow output."""
    print("Press 'enter' to exit!")

    # Load settings.
    settings = {}
    with open(SETTINGS_FILE, 'r', encoding='ascii') as settings_f:
        settings = json.load(settings_f)

    # Initialize the camera.
    cam = Camera(settings)
    color_calibration = settings["color_calibration_hsv"]

    # Loop until the user presses the enter key.
    while cv2.waitKey(1) != ENTER_KEY:
        # Get camera data.
        frame = cam.get_frame()
        frame_warped = cam.warp_frame(frame)

        # Do the magic lane detection.
        lane_mask = get_lane_marker_mask(
            frame_warped,
            color_calibration["center_line"],
            color_calibration["outside_line"])
        contours = find_center_points(lane_mask, 5)

        cv2.drawContours(frame_warped, contours, -1, (255, 0, 0), 5)

        # Show the test data to the user.
        cv2.imshow('input', frame_warped)
        cv2.imshow('output', lane_mask)

if __name__ == '__main__':
    _run_test()
