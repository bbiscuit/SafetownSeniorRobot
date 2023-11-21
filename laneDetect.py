################################################################################
# laneDetect.py
#
# A Python script which provides functionality for detecting a SafeTown lane
# from live video feed from a calibrated webcam (see calibrate.py). This is
# not intended as a standalone program, although, if ran, it will run tests.
#
# Author: Andrew Huffman
################################################################################

import cv2
import csv
import numpy as np

DEFAULT_CALIBRATION_FILE = "camWarpCalibration.csv"
WHITE_TAPE_HSV = (0, 0, 255)
YELLOW_TAPE_HSV = (30, 85, 255)

class Camera:
    """A calibrated camera for use with SafeTown."""

    class CamReadException(Exception):
        """Raised when the camera fails to read."""
        pass

    def __init__(self, 
                 camID=0, 
                 warpingParams: dict[str, tuple[int, int]]=None):
        self.cam = cv2.VideoCapture(camID)

        # Use a sample frame to determine the resolution of the camera
        self.rows, self.cols, _ = self.cam.read()[1].shape

        # Load the matrix used for bird's-eye view warping
        self.warpingMatrix = None
        if warpingParams is not None:
            input_pts = np.float32([warpingParams['Top Left'], 
                                    warpingParams['Top Right'], 
                                    warpingParams['Bottom Left'], 
                                    warpingParams['Bottom Right']])
            output_pts = np.float32([(0, 0), (self.cols, 0), (0, self.rows), (self.cols, self.rows)])
            self.warpingMatrix = cv2.getPerspectiveTransform(input_pts, output_pts)
    
    def getFrame(self):
        res, frame = self.cam.read()

        if res:
            return frame
        else:
            raise Camera.CamReadException

    def warpFrame(self, frame: 'cv2.MatLike'):
        """Copies the frame and warps it based upon the warping parameters, if
        they have been given. Otherwise, no warping is done."""

        if self.warpingMatrix is not None:
            return cv2.warpPerspective(frame.copy(), self.warpingMatrix, (self.cols, self.rows))
        else:
            return frame.copy()

    def initFromCSV(camID=0, camCalibrationFilename=DEFAULT_CALIBRATION_FILE):
        """Initializes and loads calibration data for the camera."""
        calibration_data = loadCameraCalibrationDataFromCSV(camCalibrationFilename)
        return Camera(camID, calibration_data)

def loadCameraCalibrationDataFromCSV(filename=DEFAULT_CALIBRATION_FILE):
    """Loads camera calibration data into a dict, or None if the file doesn't
    exist."""
    result = {}
    try:
        with open(filename, 'r') as calibration_csv:
            reader = csv.reader(calibration_csv, dialect="excel")
            for row in reader:
                if len(row) >= 3:
                    result[row[0]] = (int(row[1]), int(row[2]))
    except FileNotFoundError:
        return None
    
    return result

def getLaneMarkerMask(frame: 'cv2.MatLike', center_color=YELLOW_TAPE_HSV, outside_color=WHITE_TAPE_HSV, value_tolerance=15):
    """Gets the binary mask which corresponds to the lane markers (center dotted line and outside line), by color in HSV."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    def applySaturationTolerance(x, tolerance):
        lower_v = x[1] - tolerance
        if lower_v < 0:
            lower_v = 0

        higher_v = x[1] + tolerance
        if higher_v > 100:
            higher_v = 100
        
        return ((x[0], lower_v, x[2]), (x[0], higher_v, x[2]))
    
    # Apply Thresholding for the outside tape.
    lower_outside, higher_outside = applySaturationTolerance(outside_color, value_tolerance)
    mask_outside = cv2.inRange(hsv, lower_outside, higher_outside)
    
    # Apply Thresholding for the inside tape.
    lower_inside, higher_inside = applySaturationTolerance(center_color, value_tolerance)
    mask_inside = cv2.inRange(hsv, lower_outside, higher_inside)

    # Merge the masks.
    mask = mask_outside | mask_inside

    return mask

def findCenterPoints(marker_mask: 'cv2.MatLike'):
    """Finds the center point between each dot in the middle line and the right line."""
    # Find the contours in the image.
    # Identify the right line -- this will always be the largest contour.
    # For each contour which isn't the right line, find the shortest line between the two,
    # and add the center to the result -- this will be the center of the lane.

def _runTest():
    """Tests the lane detection with imshow output."""
    print("Press 'enter' to exit!")
    cam = Camera.initFromCSV(0)

    # Loop until the user presses the enter key.
    ENTER_KEY = 13
    while cv2.waitKey(1) != ENTER_KEY:
        # Get camera data.
        frame = cam.getFrame()
        frame_warped = cam.warpFrame(frame)

        # Do the magic lane detection.
        lane_mask = getLaneMask(getLaneMarkerMask(frame_warped))

        # Show the test data to the user.
        cv2.imshow('input', frame_warped)
        cv2.imshow('output', lane_mask)

if __name__ == '__main__':
    _runTest()