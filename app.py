#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

# feature parameters
feature_params = {
        'maxCorners': 100,
        'qualityLevel': 0.3,
        'minDistance': 7
}

# camera and video intrinsics
FPS = 30
PX_PER_CM = 370

# app parameters
REFRESH_RATE = 20
DISTANCE_THRESH = 20

# compute Euclidean distance (distance formula)
def d2(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))

# load video and read in the first frame
video_cap = cv2.VideoCapture('test.mov')
_, frame = video_cap.read()
frame_counter = 1

# convert first frame to grayscale and pick points to track
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(image=old_gray, **feature_params)

# show features on first frame
"""
for pt in prev_pts:
    x, y = pt.ravel()
    cv2.circle(frame, (x, y), 5, (0,255,0), -1)
cv2.imshow('features', frame)
cv2.waitKey(0)
"""

# create a mask for the lines
mask = np.zeros_like(frame)
mask_text = np.zeros_like(frame)

# main UI loop
while True:
    # reset the lines
    if frame_counter % REFRESH_RATE == 0:
        mask.fill(0)
        mask_text.fill(0)
    
    # read in a video frame
    _, frame = video_cap.read()
    if frame is None:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # compute optical flow points
    next_pts, statuses, _ = cv2.calcOpticalFlowPyrLK(prevImg=old_gray, nextImg=frame_gray, prevPts=prev_pts, nextPts=None)
    
    # only keep the optical flow points that are valid
    good_next_pts = next_pts[statuses == 1]
    good_old_pts = prev_pts[statuses == 1]
    
    # draw optical flow lines
    for good_next_pt, good_old_pt in zip(good_next_pts, good_old_pts):
        # get new and old points
        x, y = good_next_pt.ravel()
        r, s = good_old_pt.ravel()
        
        # draw the optical flow line
        cv2.line(mask, (x, y), (r, s), (0,255,0), 2)
        
        # draw the coordinate of the corner points in this frame
        cv2.circle(frame, (x, y), 5, (0,255,0), -1)
        
        # draw speed if the distance criteria is met
        distance = d2((x, y), (r, s))
        if distance > DISTANCE_THRESH:
            # compute speed
            speed_str = str(distance / PX_PER_CM * FPS) + ' cm/s'
            print(speed_str)
            cv2.putText(mask_text, speed_str, (x,y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0))
    
    # combine mask with frame
    frame_final = cv2.add(frame, mask)
    frame_final = cv2.add(frame_final, mask_text)
    
    cv2.imshow('frame', frame_final)
    
    # update for next frame
    old_gray = frame_gray.copy()
    prev_pts = good_next_pts.reshape(-1, 1, 2)
    frame_counter += 1
    if cv2.waitKey(10) == ord('q'):
        break

# clean up resources
cv2.destroyAllWindows()
video_cap.release()
