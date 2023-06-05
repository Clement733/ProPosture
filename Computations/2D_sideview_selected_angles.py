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
    angle = round(np.abs(radians*180.0/np.pi))

    if angle >180.0:
        angle = 360-angle

    return angle

list_videos=["new_pushup.mp4","hip_test_pushup.mp4"]

path=os.path.join(os.path.dirname(os.getcwd()),"ProPosture","raw_data",list_videos[1])


# CAREFUL: THIS FILE OPERATES ON A LIVEFEED

cap = cv2.VideoCapture(path)

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

            # 1.1 GETTING COORDINATES FOR ALL ANGLES
            ## ALL LEFT BODY PARTS
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            l_ear= [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].z]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]

            ## ALL RIGHT BODY PARTS
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            r_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].z]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]

            # 1.2 CHECKING WHICH SIDE THE PERSON IS FILMING
            if l_ear[0] < l_hip[0]:
                sideview_angle="Left"
            if r_ear[0] > r_hip[0]:
                sideview_angle="Right"


            # 2. GETTING USEFUL ANGLES ONE-BY-ONE
            ## 2.1 GETTING ELBOW ANGLES --> ALLOW TO CHECK IF FULL REP

            if sideview_angle=="Left":
                ### LEFT ELBOW
                #### Calculate angle
                left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                #### Visualize angle
                left_elbow_text_position = (image.shape[1] - 220, 30)
                cv2.putText(image, str(f'Left elbow angle: {left_elbow_angle}'),
                            left_elbow_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                ### RIGHT ELBOW
                #### Calculate angle
                right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                #### Visualize angle
                right_elbow_text_position = (image.shape[1] - 220, 30)
                cv2.putText(image, str(f'Right elbow angle: {right_elbow_angle}'),
                            right_elbow_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##2.2 GETTING NECK-SHOULDER-HIP ANGLE --> ALLOW TO CHECK FOR NECK ALIGNMENT

            if sideview_angle=="Left":
                ### LEFT NECK
                #### Calculate angle
                left_neck_angle = calculate_angle(l_ear,l_shoulder,l_hip)
                #### Visualize angle
                left_neck_text_position = (image.shape[1] - 220, 70)
                cv2.putText(image, str(f'Left neck angle: {left_neck_angle}'),
                            left_neck_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                ### RIGHT NECK
                #### Calculate angle
                right_neck_angle = calculate_angle(r_ear,r_shoulder,r_hip)
                #### Visualize angle
                right_neck_text_position = (image.shape[1] - 220, 70)
                cv2.putText(image, str(f'Right neck angle: {right_neck_angle}'),
                            right_neck_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##2.3 GETTING HIPS ANGLES --> ALLOW TO CHECK FOR BUTT ALIGNMENT

            if sideview_angle=="Left":
                ### LEFT HIP
                #### Calculate angle
                left_hip_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
                #### Visualize angle
                left_hip_text_position = (image.shape[1] - 220, 110)
                cv2.putText(image, str(f'Left hip angle: {left_hip_angle}'),
                            left_hip_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                ### RIGHT HIP
                #### Calculate angle
                right_hip_angle = calculate_angle(r_shoulder, r_hip, r_ankle)
                #### Visualize angle
                right_hip_text_position = (image.shape[1] - 220, 110)
                cv2.putText(image, str(f'Right hip angle: {right_hip_angle}'),
                            right_hip_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##2.4 GETTING KNEE ANGLES

            if sideview_angle=="Left":
                ### LEFT KNEE
                #### Calculate angle
                left_knee_angle = calculate_angle(l_hip, l_knee,l_ankle)
                #### Visualize angle
                left_knee_text_position = (image.shape[1] - 220, 150)
                cv2.putText(image, str(f'Left knee angle: {left_knee_angle}'),
                            left_knee_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                ### RIGHT KNEE
                #### Calculate angle
                right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                #### Visualize angle
                right_knee_text_position = (image.shape[1] - 220, 150)
                cv2.putText(image, str(f'Right knee angle: {right_knee_angle}'),
                            right_knee_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##2.5 COMPUTE X-DISTANCE BETWEEN SHOULDER AND WRIST

            if sideview_angle=="Left":
                ### LEFT X-DISTANCE
                left_x_distance = round(abs(l_wrist[0]-l_shoulder[0]),4)
                left_x_distance_text_position = (image.shape[1] - 220, 190)
                cv2.putText(image, str(f'Left_x_distance: {left_x_distance}'),
                            left_x_distance_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                ### RIGHT X-DISTANCE
                right_x_distance = round(abs(r_wrist[0]-r_shoulder[0]),4)
                right_x_distance_text_position = (image.shape[1] - 220, 190)
                cv2.putText(image, str(f'Right_x_distance: {right_x_distance}'),
                            right_x_distance_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            #3 QUALITY LOGIC TESTING

            ##3.1 REP COUNTER
            if sideview_angle=="Left":
                if left_elbow_angle > 160 and stage =='down':
                    counter +=1
                if left_elbow_angle > 160:
                    stage = "up"
                if left_elbow_angle < 90:
                    stage="down"
                rep_counter_text_position = (30, 30)
                #rep_stage_text_position = (30, 60)
                cv2.putText(image, str(f'Reps count: {counter}'),
                            rep_counter_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                # cv2.putText(image, str(f'Rep stage: {stage}'),
                #             rep_stage_text_position,
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                if right_elbow_angle > 160 and stage =='down':
                    counter +=1
                if right_elbow_angle > 160:
                    stage = "up"
                if right_elbow_angle < 90:
                    stage="down"
                rep_counter_text_position = (30, 30)
                rep_stage_text_position = (30, 60)
                cv2.putText(image, str(f'Reps count: {counter}'),
                            rep_counter_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(f'Rep stage: {stage}'),
                            rep_stage_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##3.2 FULL REP VERIFIER
            if sideview_angle=="Left":
                #1st rep counter
                if left_elbow_angle in range(170,180):
                    advice = "GOOD JOB U LOSER ! Now down"
                if left_elbow_angle in range(0,65):
                    advice="GOOD JOB U LOSER ! Now push up"
                rep_advice_text_position = (30, 60)
                cv2.putText(image, str(f'{advice}'),
                            rep_advice_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                #1st rep counter
                if right_elbow_angle in range(170,180):
                    advice = "GOOD JOB U LOSER ! Now down"
                if right_elbow_angle in range(0,65):
                    advice="GOOD JOB U LOSER ! Now push up"
                rep_advice_text_position = (30, 60)
                cv2.putText(image, str(f'{advice}'),
                            rep_advice_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##3.3 NECK ALIGNMENT
            if sideview_angle=="Left":
                if left_neck_angle in range(155,180):
                    neck_status= "Good"
                if left_neck_angle < 155:
                    neck_status= "Bad"
                    advice_neck = 'Tuck chin in'
                neck_status_text_position = (30,100)
                cv2.putText(image, str(f'Neck status: {neck_status}'),
                            neck_status_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                if right_neck_angle in range(155,180):
                    neck_status= "Good"
                if right_neck_angle < 155:
                    neck_status= "Bad"
                    advice_neck = 'Tuck chin in'
                neck_status_text_position = (30,100)
                cv2.putText(image, str(f'Neck status: {neck_status}'),
                            neck_status_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##3.4 BUTT ALIGNMENT
            if sideview_angle=="Left":
                if left_hip_angle in range(170,180):
                    hip_status= "Good"
                if left_hip_angle < 170:
                    hip_status= "Bad"
                    advice_butt = 'Lower hips'
                hip_status_text_position = (30,140)
                cv2.putText(image, str(f'Hips status: {hip_status}'),
                            hip_status_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                if right_hip_angle in range(170,180):
                    hip_status= "Good"
                if right_hip_angle < 170:
                    hip_status= "Bad"
                    advice_butt = 'Lower hips'
                hip_status_text_position = (30,140)
                cv2.putText(image, str(f'Hips status: {hip_status}'),
                            hip_status_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            ##3.5 KNEE ALIGNMENT
            if sideview_angle=="Left":
                if left_knee_angle in range(155,180):
                    knee_status= "Good"
                if left_knee_angle < 155:
                    knee_status= "Bad"
                    advice_knee = 'Straighten legs'
                knee_status_text_position = (30,180)
                cv2.putText(image, str(f'Knees status: {knee_status}'),
                            knee_status_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                if right_knee_angle in range(155,180):
                    knee_status= "Good"
                if right_knee_angle < 155:
                    knee_status= "Bad"
                    advice_knee = 'Straighten legs'
                knee_status_text_position = (30,180)
                cv2.putText(image, str(f'Knees status: {knee_status}'),
                            knee_status_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##3.6 SHOULDERS
            #checking first if person is at the top of his pushup, then test x-distance
            if sideview_angle=="Left":
                if left_elbow_angle>150:
                    if left_x_distance<0.05:
                        hands_status="Good"
                    if left_x_distance>0.05:
                        hands_status="Bad"
                        advice_hands = 'Align hands with shoulders'
                hands_status_text_position = (30,220)
                cv2.putText(image, str(f'Hands position: {hands_status}'),
                            hands_status_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            if sideview_angle=="Right":
                if left_elbow_angle>150:
                    if left_x_distance<=0.05:
                        hands_status="Good"
                    if left_x_distance>0.05:
                        hands_status="Bad"
                        advice_hands = 'Align hands with shoulders'
                hands_status_text_position = (30,220)
                cv2.putText(image, str(f'Hands position: {hands_status}'),
                            hands_status_text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


        except:
            pass

        #4 CREATING LABELS AND STUFF

        # ##4.1 Setup status box
        # cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        # ##4.2 Status data
        # cv2.putText(image, 'Quality', (15,12),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, quality,
        #             (10,60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # ###!! WE KEEP THIS PART TO MAYBE LATER DISPLAY NUMBER OF REPS
        # cv2.putText(image, 'STAGE', (65,12),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, stage,
        #            (60,60),
        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        ##4.3 Render detections
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
