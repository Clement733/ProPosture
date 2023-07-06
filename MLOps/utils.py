# Import necessary libraries and modules
import cv2
import numpy as np
import mediapipe as mp

# Load video
def load_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    return cap

# Get video dimensions for displaying text on body
def get_video_dimensions(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return height, width

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Get coordinates of body points
def get_landmarks(results):
    try:
        landmarks = results.pose_landmarks.landmark

        # 1. GETTING COORDINATES FOR ALL ANGLES
        ## ALL LEFT BODY PARTS
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        l_ear= [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        ## ALL RIGHT BODY PARTS
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

    except:
            pass
    return l_shoulder, l_elbow, l_wrist, l_hip, l_ankle, l_ear, l_knee, r_shoulder, r_elbow, r_wrist, r_hip, r_ankle, r_ear, r_knee

# CHECKING WHICH SIDE THE PERSON IS FILMING
def get_sideview(*landmarks):
    l_ear = landmarks[5]
    r_ear = landmarks[12]
    l_hip = landmarks[3]
    r_hip = landmarks[10]
    if l_ear[0] < l_hip[0]:
        sideview_angle="left"
    if r_ear[0] > r_hip[0]:
        sideview_angle="right"
    return sideview_angle

# Function to calculate angles
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return round(angle)

# Calculate angles of joints
def get_angles(l_shoulder, l_elbow, l_wrist, l_hip, l_ankle, l_ear, l_knee, r_shoulder, r_elbow, r_wrist, r_hip, r_ankle, r_ear, r_knee):
    #Neck
    left_neck_angle = calculate_angle(l_ear,l_shoulder,l_hip)
    right_neck_angle = calculate_angle(r_ear,r_shoulder,r_hip)
    # Wrists
    wrist_l_shoulder_angle = calculate_angle(r_shoulder, l_shoulder, l_wrist)
    wrist_r_shoulder_angle = calculate_angle(l_shoulder, r_shoulder, r_wrist)
    # Elbows
    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    # Shoulders
    left_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
    right_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
    # Shoulder Distance
    left_x_distance = round(abs(l_wrist[0]-l_shoulder[0]),4)
    right_x_distance = round(abs(r_wrist[0]-r_shoulder[0]),4)
    # X-DISTANCE BETWEEN BOTH WRISTS/SHOULDERS AND BOTH ELBOWS
    shoulders_x_distance = round(abs(l_shoulder[0]-r_shoulder[0]),3)
    elbows_x_distance = round(abs(l_elbow[0]-r_elbow[0]),3)
    # Hips
    left_hip_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
    right_hip_angle = calculate_angle(r_shoulder, r_hip, r_ankle)
    # Knees
    left_knee_angle = calculate_angle(l_hip, l_knee,l_ankle)
    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

    return left_elbow_angle, right_elbow_angle, left_neck_angle, right_neck_angle, wrist_l_shoulder_angle, wrist_r_shoulder_angle, left_shoulder_angle, right_shoulder_angle, left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle, left_x_distance, right_x_distance, shoulders_x_distance, elbows_x_distance
