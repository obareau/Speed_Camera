#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:49:29 2021
OpenCV BASIC
@author: macuser
"""
import cv2
import numpy as np

# Features parameters
features_params = {
         'maxCorners': 100,
         'qualityLevel': 0.3,
         'minDistance': 7
}


# App parameters
REFRESH_RATE = 20

# Looad video and red in the first frame
video_cap = cv2.VideoCapture('test.mov')
_, frame = video_cap.read()
frame_counter = 1

# Convert first frame to grayscale and picks point to track
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(image=old_gray, **features_params)

# Show features on first frame
"""
for pt in prev_pts:
    x, y = pt.ravel()
    cv2.circle(frame, (x, y), 5, (0,255,0), -1)
cv2.imshow('features', frame)
cv2.waitKey(0)
"""
# Create a mask for the lines
mask = np.zeros_like(frame)


#  Main UI loop
while True:
    # Reset the lines
    if frame_counter % REFRESH_RATE == 0:
        mask.fill(0)

    # Read in a video frame
    _, frame = video_cap.read()
    if frame is None:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow points
    next_pts, statuses, _ = cv2.calcOpticalFlowPyrLK(prevImg=old_gray, nextImg=frame_gray, prevPts=prev_pts, nextPts=None)

    # Only keep the optical flow points that are valid
    good_next_pts = next_pts[statuses == 1]
    good_old_pts = prev_pts[statuses == 1]

    # Frawraw optical flow lines
    for good_next_pt, good_old_pt in zip(good_next_pts, good_old_pts):
        # get new and old points
        x, y = good_next_pt.ravel()
        r, s = good_old_pt.ravel()

        # Draw the optical flow line
        cv2.line(mask, (x, y), (r, s), (0, 255, 0), 2)

        # Draw  the coordinate of the corner points in this frame
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    #  Combine mask with frame
    frame_final = cv2.add(frame, mask)

    cv2.imshow('frame', frame_final)
    # update for next frame
    old_gray = frame_gray.copy()
    prev_pts = good_next_pts.reshape(-1, 1, 2)
    frame_counter += 1
    if cv2.waitKey(10) == ord('q'):
        break

# clean up
cv2.destroyAllWindows()
video_cap.release()

