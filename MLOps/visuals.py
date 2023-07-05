'''This file outputs advice based on metrics.py'''
import cv2
import numpy as np

#STATUS BOX
def show_status(image, rep_advice, stage, rep_counter):
    #4.1 Setup status box
    status_box = cv2.rectangle(image, (0,0), (600,80), (245,117,16), -1)

    # Rep data
    reps_title = cv2.putText(image, 'REPS', (15,12),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    rep_count_text = cv2.putText(image, str(rep_counter),
                (10,70),
                cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    # TODO: delete if not needed
    # shows UP/DOWN stage status
    # stage_title = cv2.putText(image, 'STAGE', (75,12),
    #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    # stage_text = cv2.putText(image, stage,
    #             (60,70),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3, cv2.LINE_AA)

    #Rep Status Data
    stage_title = cv2.putText(image, 'REP STATUS', (75,12),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    stage_text = cv2.putText(image, rep_advice, (60,70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    return status_box, reps_title, rep_count_text, stage_title, stage_text

# Determine status of joints
def show_neck(image, neck, sideview_angle, height, width, *landmarks):
    if sideview_angle == 'left':
        neck_status = cv2.circle(image, tuple(np.multiply(landmarks[5], [width, height]).astype(int)),
                        20, neck[0], -1)
    if sideview_angle == 'right':
        neck_status = cv2.circle(image, tuple(np.multiply(landmarks[12], [width, height]).astype(int)),
                        20, neck[0], -1)
    return neck_status

def show_hip(image, hip, sideview_angle, height, width, *landmarks):
    if sideview_angle == 'left':
        hip_status = cv2.circle(image, tuple(np.multiply(landmarks[3], [width, height]).astype(int)),
                        20, hip[0], -1)
    if sideview_angle == 'right':
        hip_status = cv2.circle(image, tuple(np.multiply(landmarks[10], [width, height]).astype(int)),
                        20, hip[0], -1)
    return hip_status

def show_knee(image, knee, sideview_angle, height, width, *landmarks):
    if sideview_angle == 'left':
        knee_status = cv2.circle(image, tuple(np.multiply(landmarks[6], [width, height]).astype(int)),
                        20, knee[0], -1)
    if sideview_angle == 'right':
        knee_status = cv2.circle(image, tuple(np.multiply(landmarks[13], [width, height]).astype(int)),
                        20, knee[0], -1)
    return knee_status

def show_hand(image, hand, sideview_angle, height, width, *landmarks):
    if sideview_angle == 'left':
        hand_status = cv2.circle(image, tuple(np.multiply(landmarks[2], [width, height]).astype(int)),
                        20, hand[0], -1)
    if sideview_angle == 'right':
        hand_status = cv2.circle(image, tuple(np.multiply(landmarks[9], [width, height]).astype(int)),
                        20, hand[0], -1)
    return hand_status

def show_align(image, align, height, width, *landmarks):
    left_align_status = cv2.circle(image, tuple(np.multiply(landmarks[2], [width, height]).astype(int)),
                    20, align[0], -1)
    right_align_status = cv2.circle(image, tuple(np.multiply(landmarks[9], [width, height]).astype(int)),
                    20, align[0], -1)
    return left_align_status, right_align_status

def show_elbow(image, elbow, height, width, *landmarks):
    left_elbow_status = cv2.circle(image, tuple(np.multiply(landmarks[1], [width, height]).astype(int)),
                    20, elbow[0], -1)
    right_elbow_status = cv2.circle(image, tuple(np.multiply(landmarks[8], [width, height]).astype(int)),
                    20, elbow[0], -1)
    return left_elbow_status, right_elbow_status
