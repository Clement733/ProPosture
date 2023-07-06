'''This file takes the angles from data.py and outputs conditions for visuals.py'''

# Rep Counter and Up/Down Stage
def get_reps_and_stage(elbow_angles, rep_counter, stage):
    left_elbow_angle, right_elbow_angle = elbow_angles
    if left_elbow_angle and right_elbow_angle < 90:
        stage = "DOWN"
    if left_elbow_angle and right_elbow_angle > 110 and stage =='DOWN':
        stage = "UP"
        rep_counter +=1
    return stage, rep_counter

#Full rep counter
def get_full_reps(elbow_angles, top_full_rep_counter, bottom_full_rep_counter, full_rep_stage):
    left_elbow_angle, right_elbow_angle = elbow_angles
    average_elbow_angle=(left_elbow_angle+right_elbow_angle)/2
    if average_elbow_angle > 175 and full_rep_stage =='down':
        top_full_rep_counter +=1
    if average_elbow_angle > 175:
        full_rep_stage = "up"
    if average_elbow_angle < 60 and full_rep_stage == 'up':
        bottom_full_rep_counter +=1
    if average_elbow_angle < 60:
        full_rep_stage="down"
    return top_full_rep_counter, bottom_full_rep_counter, full_rep_stage

# Full Rep Verifier, live advices
def get_rep_advice(elbow_angles, sideview_angle = None, rep_advice = None):
    left_elbow_angle, right_elbow_angle = elbow_angles
    average_elbow_angle=(left_elbow_angle+right_elbow_angle)/2
    if sideview_angle == None:
        if average_elbow_angle > 170:
                rep_advice = "GOOD! GO DOWN!"
        if average_elbow_angle < 90:
            rep_advice="GOOD! PUSH UP!"
    if sideview_angle=="left":
        #1st rep counter
        if left_elbow_angle in range(160,180):
            rep_advice = "GO DOWN!"
        if left_elbow_angle < 140:
            rep_advice = "KEEP GOING!"
        if left_elbow_angle < 110:
            rep_advice="LOWER.."
        if left_elbow_angle < 90:
            rep_advice="GOOD! GO UP!"
    if sideview_angle=="right":
        #1st rep counter
        if right_elbow_angle in range(160,180):
            rep_advice = "GO DOWN!"
        if right_elbow_angle < 140:
            rep_advice = "KEEP GOING!"
        if right_elbow_angle < 110:
            rep_advice="LOWER.."
        if right_elbow_angle < 90:
            rep_advice="GOOD! GO UP!"
    return rep_advice

# FRONT VIEW CONDITIONS
# CHECKS HAND-SHOULDER X-AXIS ALIGNMENT AT TOP OF REP
def get_hand_align(shoulder_distance, elbow_angles, align_status_color = None, advice_align_hands = None):
    left_x_distance, right_x_distance = shoulder_distance
    left_elbow_angle, right_elbow_angle = elbow_angles
    x_distances_mean=round(abs((left_x_distance+right_x_distance)/2),4)
    average_elbow_angle=(left_elbow_angle+right_elbow_angle)/2
    if average_elbow_angle>150:
        if x_distances_mean<=0.075:
            align_status_color = (31, 194, 53)
        if x_distances_mean>0.075:
            align_status_color = (14, 14, 232)
            advice_align_hands = 'Hands too wide'
    return align_status_color, advice_align_hands

def get_shoulder_elbow_dist(shoulder_elbow_distance, elbow_angles, elbow_status_color = None, advice_elbows = None):
    shoulders_x_distance, elbows_x_distance = shoulder_elbow_distance
    shoulder_elbow_ratio = round(elbows_x_distance/shoulders_x_distance,3)
    left_elbow_angle, right_elbow_angle = elbow_angles
    average_elbow_angle = (left_elbow_angle+right_elbow_angle)/2
    if average_elbow_angle < 75:
        if shoulder_elbow_ratio < 2:
            elbow_status_color = (31, 194, 53)
        if shoulder_elbow_ratio > 2:
            elbow_status_color = (14, 14, 232)
            advice_elbows = "Tuck elbows in"
    return elbow_status_color, advice_elbows

# SIDE VIEW CONDITIONS
# NECK conditional statements
def get_neck(neck_angles, sideview_angle, advice_neck = None):
    left_neck_angle, right_neck_angle = neck_angles
    if sideview_angle == 'left':
        if left_neck_angle >= 165:
            neck_status_color= (31, 194, 53)
        if left_neck_angle in range(150,165):
            neck_status_color= (0, 255, 255)
        if left_neck_angle < 150:
            neck_status_color= (14, 14, 232)
            advice_neck = 'Tuck chin in'
    if sideview_angle == 'right':
        if right_neck_angle >= 165:
            neck_status_color= (31, 194, 53)
        if right_neck_angle in range(150,165):
            neck_status_color= (0, 255, 255)
        if right_neck_angle < 150:
            neck_status_color = (14, 14, 232)
            advice_neck = 'Tuck chin in'
    return neck_status_color, advice_neck

# TODO: this condition doesn't work well
# HAND conditional statements
def get_hand(shoulder_distance, elbow_angles, sideview_angle, hand_status_color = (31, 194, 53), advice_hand = None):
    left_x_distance, right_x_distance = shoulder_distance
    left_elbow_angle, right_elbow_angle = elbow_angles
    if sideview_angle == "left":
        if left_elbow_angle > 150:
            if left_x_distance < 0.05:
                hand_status_color = (31, 194, 53)
            if left_x_distance > 0.05:
                hand_status_color = (14, 14, 232)
                advice_hand = 'Align hands with shoulders'
    if sideview_angle == "right":
        if right_elbow_angle > 150:
            if right_x_distance < 0.05:
                hand_status_color = (31, 194, 53)
            if right_x_distance > 0.05:
                hand_status_color = (14, 14, 232)
                advice_hand = 'Align hands with shoulders'
    return hand_status_color, advice_hand

# HIP conditional statements
def get_hip(hip_angles, sideview_angle, advice_hip = None):
    left_hip_angle, right_hip_angle = hip_angles
    if sideview_angle == 'left':
        if left_hip_angle >= 170:
            hip_status_color= (31, 194, 53)
        if left_hip_angle in range(160,170):
            hip_status_color= (0, 255, 255)
        if left_hip_angle < 160:
            hip_status_color= (14, 14, 232)
            advice_hip = 'Lower hips'
    if sideview_angle == 'right':
        if right_hip_angle >= 170:
            hip_status_color= (31, 194, 53)
        if right_hip_angle in range(160,170):
            hip_status_color= (0, 255, 255)
        if right_hip_angle < 160:
            hip_status_color = (14, 14, 232)
            advice_hip = 'Lower hips'
    return hip_status_color, advice_hip

# KNEE conditional statements
def get_knee(knee_angles, sideview_angle, advice_knee = None):
    left_knee_angle, right_knee_angle = knee_angles
    if sideview_angle == 'left':
        if left_knee_angle >= 160:
            knee_status_color= (31, 194, 53)
        if left_knee_angle in range(150,160):
            knee_status_color= (0, 255, 255)
        if left_knee_angle < 150:
            knee_status_color= (14, 14, 232)
            advice_knee = 'Straighten legs'
    if sideview_angle == 'right':
        if right_knee_angle >= 160:
            knee_status_color= (31, 194, 53)
        if right_knee_angle in range(150,160):
            knee_status_color= (0, 255, 255)
        if right_knee_angle < 150:
            knee_status_color = (14, 14, 232)
            advice_knee = 'Straighten legs'
    return knee_status_color, advice_knee
