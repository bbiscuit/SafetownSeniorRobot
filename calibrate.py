"""
A standalone app to calibrate the camera for the SafeTown senior robot. This is not intended
to be imported by other scripts.

Author: Andrew Huffman
"""

import json
import functools
import cv2
import numpy as np
import detect

SETTINGS_FILENAME = "settings.json"
TRACKER_RADIUS = 10
TRACKER_COLOR = (0, 0, 255)

def warp(frame, coords):
    """Copies the frame and warps it to be rectangular, with the coordinates as the corners."""
    rows, cols, _ = frame.shape
    input_pts = np.float32(
        [coords["top_left"], coords["top_right"], coords["bottom_left"], coords["bottom_right"]]
        )
    output_pts = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    transform = cv2.getPerspectiveTransform(input_pts, output_pts)
    frame = cv2.warpPerspective(frame.copy(), transform, (cols, rows))
    return frame

def draw_warping_trackers(frame, coords):
    """Draws the calibration trackers on the given frame (doesn't copy)."""
    cv2.circle(frame, coords["top_left"], TRACKER_RADIUS, TRACKER_COLOR, -1)
    cv2.circle(frame, coords["top_right"], TRACKER_RADIUS, TRACKER_COLOR, -1)
    cv2.circle(frame, coords["bottom_left"], TRACKER_RADIUS, TRACKER_COLOR, -1)
    cv2.circle(frame, coords["bottom_right"], TRACKER_RADIUS, TRACKER_COLOR, -1)

def create_warping_tracker_window(frame, name: str, coords: dict):
    """Creates a window which contains trackbars to configure warping. Takes in coordinate values
    as default trackbar vals."""
    frame_rows, frame_cols, _ = frame.shape

    # This helper function exists to keep the size of this function down. This function was much,
    # much longer before it existed.
    def create_trackbar(text, coord, idx):
        # Update functions. This has to be a fucntion and not a lambda because it's doing
        # assignment.
        def update_coordinate(val, coord, idx):
            coords[coord][idx] = val

        cv2.createTrackbar(
            text,
            name,
            coords[coord][idx],
            frame_cols if idx == 0 else frame_rows,
            functools.partial(update_coordinate, coord=coord, idx=idx)
        )

    cv2.namedWindow(name)
    create_trackbar("Top Left X", "top_left", 0)
    create_trackbar("Top Left Y", "top_left", 1)
    create_trackbar("Top Right X", "top_right", 0)
    create_trackbar("Top Right Y", "top_right", 1)
    create_trackbar("Bottom Left X", "bottom_left", 0)
    create_trackbar("Bottom Left Y", "bottom_left", 1)
    create_trackbar("Bottom Right X", "bottom_right", 0)
    create_trackbar("Bottom Right Y", "bottom_right", 1)

def create_tolerance_trackbar_window(name: str, color_calibration: dict):
    """Creates a window for adjusting color tolerances."""
    cv2.namedWindow(name)

    def create_hsv_tolerance_trackbars(line_title: str):
        def update_val(val, title, dim):
            color_calibration[title]["tolerance"][dim] = val

        cv2.createTrackbar(
            f'{line_title} H tolerance',
            name,
            color_calibration[line_title]["tolerance"][0],
            360,
            functools.partial(update_val, title=line_title, dim=0)
        )
        cv2.createTrackbar(
            f'{line_title} S tolerance',
            name,
            color_calibration[line_title]["tolerance"][1],
            255,
            functools.partial(update_val, title=line_title, dim=1)
        )
        cv2.createTrackbar(
            f'{line_title} V tolerance',
            name,
            color_calibration[line_title]["tolerance"][2],
            255,
            functools.partial(update_val, title=line_title, dim=2)
        )

    create_hsv_tolerance_trackbars('center_line')
    create_hsv_tolerance_trackbars('outside_line')

def main():
    """Driver code for the calibration app."""
    # Load settings.
    settings = {}
    with open(SETTINGS_FILENAME, 'r', encoding='ascii') as settings_f:
        settings = json.load(settings_f)

    # Load coordinates from settings, as a meaninful default.
    coords = settings["cam_warp_calibration"]

    # Set up camera information.
    cam = cv2.VideoCapture(settings["cam_device"])

    # Initialize the window which will be used for warping calibration.
    create_warping_tracker_window(cam.read()[1], "trackers", coords)

    # Intialize the window which will be used for color tolerance calibration.
    create_tolerance_trackbar_window('color calibration', settings["color_calibration"])

    # Setup the mouse pointer, to calibrate tape colors.
    cv2.namedWindow('calibrate')
    calibrating_center = True
    def calibrate_color_with_mouse(event, x, y, _, __):
        nonlocal calibrating_center

        _, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if event == cv2.EVENT_LBUTTONDOWN:
            color_settings = settings["color_calibration"]
            color = frame[y, x]

            if calibrating_center:
                color_settings["center_line"]["hsv"] = [
                    int(color[0]),
                    int(color[1]),
                    int(color[2])]
                print("Recorded center line color!")
                calibrating_center = False
            else:
                color_settings["outside_line"]["hsv"] = [
                    int(color[0]),
                    int(color[1]),
                    int(color[2])]
                print("Recorded outside line color!")
                calibrating_center = True

    cv2.setMouseCallback('calibrate', calibrate_color_with_mouse)

    i = 0
    while True:
        ret, frame = cam.read()
        warped = warp(frame, coords)

        # Handle exit conditions.
        if not ret:
            print("Can't get the frame!")
            break
        if cv2.waitKey(1) == 13:
            break

        # Get masks for inside and outside line.
        color_calibration = settings["color_calibration"]
        mask_outside, mask_center = detect.get_lane_marker_mask(warped, color_calibration)

        # Update the frames being shown.
        draw_warping_trackers(frame, coords)
        cv2.imshow('calibrate', frame)
        cv2.imshow('warped', warped)
        cv2.imshow('outside line mask', mask_outside)
        cv2.imshow('center line mask', mask_center)
        i += 1

    # Write back the settings to file.
    with open(SETTINGS_FILENAME, 'w', encoding='ascii') as settings_f:
        json.dump(settings, settings_f)

if __name__ == "__main__":
    main()
