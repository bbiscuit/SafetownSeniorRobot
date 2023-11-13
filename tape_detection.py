import cv2
import numpy as np

LOWER_WHITE = np.array([200, 200, 200], dtype=np.uint8)
UPPER_WHITE = np.array([255, 255, 255], dtype=np.uint8)

LOWER_YELLOW = np.array([20, 100, 100], dtype=np.uint8)
UPPER_YELLOW = np.array([30, 255, 255], dtype=np.uint8)

def getYellowContours(frame_bgr):
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, LOWER_YELLOW, UPPER_YELLOW)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def getWhiteContours(frame_bgr):
    mask = cv2.inRange(frame_bgr, LOWER_WHITE, UPPER_WHITE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def findContourCenter(contour):
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    else:
        return (-1, -1)

def fitLineThroughContours(contours, minArea, imgHeight, imgWidth):
    # Find all centers.
    centers = []
    distFromBottomLeft = 10000
    startPoint = (10000, 0)
    for contour in contours:
        if minArea <= cv2.contourArea(contour):
            center = findContourCenter(contour)
            if center != (-1, -1):
                centers.append(center)

                # If this is closer to the bottom left,
                # then select this one.
                d = np.sqrt(np.square(center[0]) + np.square(center[1] - imgHeight))
                if distFromBottomLeft > d:
                    startPoint = center
                    distFromBottomLeft = d

    # If there are not at least 2 valid centroids, quit out.
    if len(centers) < 2:
        return None

    # Find the line with the minimum error, taking the bottom left
    # contour with another conotour.
    endPoint = None
    err = 10000
    for center in centers:

        if center == startPoint:
            continue

        # Find the equation for the line between the two points.
        dx = center[0] - startPoint[0]
        dy = center[1] - startPoint[1]

        m = 10000000 # If dx == 0, then this should be "infinity."
        if dx != 0:
            m = dy / dx

        a = -m
        b = 1
        c = -startPoint[1] + m * startPoint[0]

        # Find the sum of the error with other contours, given this line.
        thisErr = 0
        for otherPoint in centers:
            if otherPoint == center or otherPoint == startPoint:
                continue
            
            thisErr += np.abs(a * otherPoint[0] + b * otherPoint[1] + c) / np.sqrt(np.square(a) + np.square(b))

        # Select this point if it has less error than the previous.
        if thisErr < err:
            err = thisErr
            endPoint = center
    
    # Return the best line, extended to the edges of the screen.
    m = (endPoint[1] - startPoint[1]) / (endPoint[0] - startPoint[0])
    b = startPoint[1] - m * startPoint[0]
    startPoint = (0, int(b))

    if -b/m >= 0 and -b/m <= imgWidth:
        endPoint = (int(-b/m), 0)
    else:
        endPoint = (imgWidth, int(m * imgWidth + b))

    return (startPoint, endPoint) # This doesn't work for some reason.

def cropFrame(frame_bgr, bottomOfCropSection):
    return cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], bottomOfCropSection), (0, 0, 0), -1)

def fitContourLine(contour, imgLength):
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0, 0.1, 0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((imgLength-x)*vy/vx)+y)
    return ((imgLength-1, righty), (0, lefty))
