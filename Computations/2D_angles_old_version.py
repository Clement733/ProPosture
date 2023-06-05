import cv2
import mediapipe as mp
import numpy as np
import os
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#function to calculate angles
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return round(angle)

#path=os.path.join(os.path.dirname(os.getcwd()),"ProPosture","raw_data","new_pushup.mp4")


# CAREFUL: THIS FILE OPERATES ON A LIVEFEED

cap = cv2.VideoCapture(0)

# Pushup position variable
stage = None
quality = None
counter = 0

# Setup mediapipe instance
with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            # Break the loop if there are no more frames in the video
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # 1. GETTING COORDINATES FOR ALL ANGLES
            ## ALL LEFT BODY PARTS
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            ## ALL RIGHT BODY PARTS
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


            ## 2.1 GETTING ELBOW ANGLES

            # LEFT ELBOW
            # Calculate angle
            left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            # Visualize angle
            left_elbow_text_position = (image.shape[1] - 200, 90)
            cv2.putText(image, str(f'Left elbow angle: {left_elbow_angle}'),
                           left_elbow_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                )

            # RIGHT ELBOW
            # Calculate angle
            right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            # Visualize angle
            right_elbow_text_position = (image.shape[1] - 200, 120)
            cv2.putText(image, str(f'Left elbow angle: {right_elbow_angle}'),
                           right_elbow_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##2.2 GETTING SHOULDER ANGLES

            #LEFT SHOULDER
            # Calculate angle
            l_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
            # Visualize angle
            left_shoulder_text_position = (image.shape[1] - 200, 30)
            cv2.putText(image, str(f'Left shoulder angle: {l_shoulder_angle}'),
                           left_shoulder_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                )

            # RIGHT SHOULDER
            # Calculate angle
            r_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
            # Visualize angle
            right_shoulder_text_position = (image.shape[1] - 200, 60)
            cv2.putText(image, str(f'Right shoulder angle: {r_shoulder_angle}'),
                           right_shoulder_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            avg_shoulder_angle = round((l_shoulder_angle+r_shoulder_angle)/2)


            ##2.3 GETTING HIPS ANGLES

            # LEFT HIP
            # Calculate angle
            left_hip_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
            # Visualize angle
            left_hip_text_position = (image.shape[1] - 200, 150)
            cv2.putText(image, str(f'Left hip angle: {left_hip_angle}'),
                           left_hip_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            # RIGHT HIP
            # Calculate angle
            right_hip_angle = calculate_angle(r_shoulder, r_hip, r_ankle)
            # Visualize angle
            right_hip_text_position = (image.shape[1] - 200, 180)
            cv2.putText(image, str(f'Right hip angle: {right_hip_angle}'),
                           right_hip_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            #3.1 Quality logic
            if l_shoulder_angle>50:
                quality = 'BAD'
            elif l_shoulder_angle<50:
                quality = 'GOOD'

            #3.2 Counter and State logic
            if left_elbow_angle and right_elbow_angle > 160:
                stage = "up"
            if left_elbow_angle and right_elbow_angle < 90 and stage =='up':
                stage="down"
                counter +=1

        except:
            pass

        #4 CREATING LABELS AND STUFF

        #4.1 Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        #4.2 Status data
        cv2.putText(image, 'Quality', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, quality,
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # !! WE KEEP THIS PART TO MAYBE LATER DISPLAY NUMBER OF REPS
        #cv2.putText(image, 'STAGE', (65,12),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        #cv2.putText(image, stage,
        #            (60,60),
        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        #4.3 Render detections
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                  mp.solutions.pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )


        #5 DISPLAYING WINDOW
        cv2.imshow('Mediapipe Feed', image)

        #6 EXITING SHORTCUT
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    #CLOSING DISPLAY WINDOW
    cap.release()
    cv2.destroyAllWindows()
