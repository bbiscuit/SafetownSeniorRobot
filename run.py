import cv2
import numpy

import tape_detection

cam = cv2.VideoCapture(0)
ret, frame = cam.read()
fourcc = cv2.VideoWriter.fourcc(*'XVID')
video = cv2.VideoWriter('test.avi', fourcc, 30, (frame.shape[1], frame.shape[0]))
for i in range(0, 300):
    ret, frame = cam.read()
    #frame_cropped = tape_detection.cropFrame(frame.copy(), 225)
    frame_cropped = frame

    # Draw a line through the dotted center line.
    contoursYellow = tape_detection.getYellowContours(frame_cropped)
    line = tape_detection.fitLineThroughContours(contoursYellow, 50, frame.shape[0], frame.shape[1])

    if line is not None:
        frame = cv2.drawContours(frame, contoursYellow, -1, (0, 255, 0), 3)
        frame = cv2.line(frame, line[0], line[1], (255, 0, 0), 3)
    
    # Draw a line through the white line on the right.
    contoursWhite = tape_detection.getWhiteContours(frame_cropped)

    if len(contoursWhite) > 0:
        largestWhiteContour = max(contoursWhite, key=lambda c: cv2.contourArea(c))
        line = tape_detection.fitContourLine(largestWhiteContour, frame.shape[1])
        frame = cv2.line(frame, line[0], line[1], (0, 0, 255), 3)
        frame = cv2.drawContours(frame, contoursWhite, -1, (0, 255, 255), 3)

    video.write(frame)
    print(i)

cam.release()
video.release()
