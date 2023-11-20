import cv2
import numpy as np
import csv

topLeftCoord = [0, 0]
topRightCoord = [0, 0]
bottomLeftCoord = [0, 0]
bottomRightCoord = [0, 0]

def loadCoordsFromCSV(filename):
    try:
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file, dialect="excel")
            for row in reader:
                if len(row) >= 3:
                    if row[0] == 'Top Left':
                        topLeftCoord[0] = int(row[1])
                        topLeftCoord[1] = int(row[2])
                    elif row[0] == 'Top Right':
                        topRightCoord[0] = int(row[1])
                        topRightCoord[1] = int(row[2])
                    elif row[0] == 'Bottom Left':
                        bottomLeftCoord[0] = int(row[1])
                        bottomLeftCoord[1] = int(row[2])
                    elif row[0] == 'Bottom Right':
                        bottomRightCoord[0] = int(row[1])
                        bottomRightCoord[1] = int(row[2])
    except FileNotFoundError:
        print("Could not find previous calibration information.")

def updateFrameWindow(frame, winName):
    # Draw the trackers on the frame.
    RAD = 10
    COLOR = (0, 0, 255)
    cv2.circle(frame, topLeftCoord, RAD, COLOR, -1)
    cv2.circle(frame, topRightCoord, RAD, COLOR, -1)
    cv2.circle(frame, bottomLeftCoord, RAD, COLOR, -1)
    cv2.circle(frame, bottomRightCoord, RAD, COLOR, -1)

    cv2.imshow(winName, frame)

def warp(rows, cols, frame):
    input_pts = np.float32([topLeftCoord, topRightCoord, bottomLeftCoord, bottomRightCoord])
    output_pts = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    transform = cv2.getPerspectiveTransform(input_pts, output_pts)
    frame = cv2.warpPerspective(frame, transform, (cols, rows))
    return frame

def writeCoordsToCSV(filename):
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file, dialect="excel")
        writer.writerow(['Top Left', topLeftCoord[0], topLeftCoord[1]])
        writer.writerow(['Top Right', topRightCoord[0], topRightCoord[1]])
        writer.writerow(['Bottom Left', bottomLeftCoord[0], bottomLeftCoord[1]])
        writer.writerow(['Bottom Right', bottomRightCoord[0], bottomRightCoord[1]])

# Load previous calibration information as a meaninful default.
CSV_NAME = 'camWarpCalibration.csv'
loadCoordsFromCSV(CSV_NAME)

# Set up camera information.
cam = cv2.VideoCapture(0)
FRAME_ROWS, FRAME_COLS, _ = cam.read()[1].shape

# Initialize the window which will be used for calibration.
def createCallibrationWindow(win_name):
    cv2.namedWindow(win_name)
    def updateTopLeftX(v):
        topLeftCoord[0] = v
    def updateTopLeftY(v):
        topLeftCoord[1] = v
    def updateTopRightX(v):
        topRightCoord[0] = v
    def updateTopRightY(v):
        topRightCoord[1] = v
    def updateBottomLeftX(v):
        bottomLeftCoord[0] = v
    def updateBottomLeftY(v):
        bottomLeftCoord[1] = v
    def updateBottomRightX(v):
        bottomRightCoord[0] = v
    def updateBottomRightY(v):
        bottomRightCoord[1] = v
    cv2.createTrackbar('Top Left X', win_name, topLeftCoord[0], FRAME_COLS, updateTopLeftX)
    cv2.createTrackbar('Top Left Y', win_name, topLeftCoord[1], FRAME_ROWS, updateTopLeftY)
    cv2.createTrackbar('Top Right X', win_name, topRightCoord[0], FRAME_COLS, updateTopRightX)
    cv2.createTrackbar('Top Right Y', win_name, topRightCoord[1], FRAME_ROWS, updateTopRightY)
    cv2.createTrackbar('Bottom Left X', win_name, bottomLeftCoord[0], FRAME_COLS, updateBottomLeftX)
    cv2.createTrackbar('Bottom Left Y', win_name, bottomRightCoord[1], FRAME_ROWS, updateBottomLeftY)
    cv2.createTrackbar('Bottom Right X', win_name, bottomRightCoord[0], FRAME_COLS, updateBottomRightX)
    cv2.createTrackbar('Bottom Right Y', win_name, bottomRightCoord[1], FRAME_ROWS, updateBottomRightY)

createCallibrationWindow('frame')
i = 0
while True:
    ret, frame = cam.read()
    warped = warp(FRAME_ROWS, FRAME_COLS, frame.copy())

    if not ret:
        print("Can't get the frame!")
        break
    elif cv2.waitKey(1) == 13:
        break


    updateFrameWindow(frame, 'frame')
    cv2.imshow('warped', warped)
    i += 1

# Output a cam calibration CSV file.
writeCoordsToCSV(CSV_NAME)
