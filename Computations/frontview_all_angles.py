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

    #radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    #angle = np.abs(radians*180.0/np.pi)

    #Calculate the vectors ba and bc
    ba = a - b
    bc = c - b

    # Calculate the dot product and the magnitudes of the vectors
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Calculate the cosine of the angle using the dot product and magnitudes
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    # Calculate the angle in radians using the arccosine function
    radians = np.arccos(cosine_angle)

    # Convert radians to degrees
    angle = np.degrees(radians)

    if angle >180.0:
        angle = 360-angle

    return round(angle)

path=os.path.join(os.path.dirname(os.getcwd()),"ProPosture","raw_data","pushups.mp4")


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

            # 2. GETTING USEFUL ANGLES ONE-BY-ONE
            ## 2.1 GETTING ELBOW ANGLES --> ALLOW TO CHECK IF FULL REP

            ### LEFT ELBOW
            #### Calculate angle
            left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            #### Visualize angle
            left_elbow_text_position = (image.shape[1] - 220, 30)
            cv2.putText(image, str(f'Left elbow angle: {left_elbow_angle}'),
                           left_elbow_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                )

            ### RIGHT ELBOW
            #### Calculate angle
            right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            #### Visualize angle
            right_elbow_text_position = (image.shape[1] - 220, 60)
            cv2.putText(image, str(f'Right elbow angle: {right_elbow_angle}'),
                           right_elbow_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##2.2 GETTING SHOULDER ANGLES

            ### LEFT SHOULDER
            #### Calculate angle
            l_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
            #### Visualize angle
            left_shoulder_text_position = (image.shape[1] - 220, 90)
            cv2.putText(image, str(f'Left shoulder angle: {l_shoulder_angle}'),
                           left_shoulder_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                )

            ### RIGHT SHOULDER
            #### Calculate angle
            r_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
            #### Visualize angle
            right_shoulder_text_position = (image.shape[1] - 220, 120)
            cv2.putText(image, str(f'Right shoulder angle: {r_shoulder_angle}'),
                           right_shoulder_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            avg_shoulder_angle = round((l_shoulder_angle+r_shoulder_angle)/2)


            ##2.3 GETTING HIPS ANGLES

            ### LEFT HIP
            #### Calculate angle
            left_hip_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
            #### Visualize angle
            left_hip_text_position = (image.shape[1] - 220, 150)
            cv2.putText(image, str(f'Left hip angle: {left_hip_angle}'),
                           left_hip_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            ### RIGHT HIP
            #### Calculate angle
            right_hip_angle = calculate_angle(r_shoulder, r_hip, r_ankle)
            #### Visualize angle
            right_hip_text_position = (image.shape[1] - 220, 180)
            cv2.putText(image, str(f'Right hip angle: {right_hip_angle}'),
                           right_hip_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


            ##2.4 GETTING NEW SHOULDER ANGLES

            ### NEW LEFT SHOULDER
            #### Calculate angle
            new_l_shoulder_angle = calculate_angle(r_shoulder, l_shoulder, l_elbow)
            #### Visualize angle
            new_left_shoulder_text_position = (image.shape[1] - 220, 210)
            cv2.putText(image, str(f'New L shoulder angle: {new_l_shoulder_angle}'),
                           new_left_shoulder_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                )

            ### NEW RIGHT SHOULDER
            #### Calculate angle
            new_r_shoulder_angle = calculate_angle(l_shoulder, r_shoulder, r_elbow)
            #### Visualize angle
            new_right_shoulder_text_position = (image.shape[1] - 220, 240)
            cv2.putText(image, str(f'New R shoulder angle: {new_r_shoulder_angle}'),
                           new_right_shoulder_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            new_avg_shoulder_angle = round((new_l_shoulder_angle+new_r_shoulder_angle)/2)


            ##2.5 GETTING SHOULDER ANGLES WITH WRISTS


            ### WRIST LEFT SHOULDER
            #### Calculate angle
            wrist_l_shoulder_angle = calculate_angle(r_shoulder, l_shoulder, l_wrist)
            #### Visualize angle
            wrist_left_shoulder_text_position = (image.shape[1] - 220, 270)
            cv2.putText(image, str(f'Wrist L shoulder angle: {wrist_l_shoulder_angle}'),
                           wrist_left_shoulder_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                )

            ### WRIST RIGHT SHOULDER
            #### Calculate angle
            wrist_r_shoulder_angle = calculate_angle(l_shoulder, r_shoulder, r_wrist)
            #### Visualize angle
            wrist_right_shoulder_text_position = (image.shape[1] - 220, 300)
            cv2.putText(image, str(f'Wrist R shoulder angle: {wrist_r_shoulder_angle}'),
                           wrist_right_shoulder_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            wrist_avg_shoulder_angle = round((wrist_l_shoulder_angle+wrist_r_shoulder_angle)/2)



            ##2.6 GETTING HIPS ANGLES --> ALLOW TO CHECK FOR BUTT ALIGNMENT

            ### LEFT HIP
            #### Calculate angle
            left_hip_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
            #### Visualize angle
            left_hip_text_position = (image.shape[1] - 220, 330)
            cv2.putText(image, str(f'Left hip angle: {left_hip_angle}'),
                           left_hip_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            ### RIGHT HIP
            #### Calculate angle
            right_hip_angle = calculate_angle(r_shoulder, r_hip, r_ankle)
            #### Visualize angle
            right_hip_text_position = (image.shape[1] - 220, 360)
            cv2.putText(image, str(f'Right hip angle: {right_hip_angle}'),
                           right_hip_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            ##2.7 GETTING NECK-SHOULDER-HIP ANGLE --> ALLOW TO CHECK FOR NECK ALIGNMENT

            ### LEFT NECK
            #### Calculate angle
            left_neck_angle = calculate_angle(l_ear,l_shoulder,l_hip)
            #### Visualize angle
            left_neck_text_position = (image.shape[1] - 220, 390)
            cv2.putText(image, str(f'Left neck angle: {left_neck_angle}'),
                           left_neck_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                )

            ### RIGHT NECK
            #### Calculate angle
            right_neck_angle = calculate_angle(r_ear,r_shoulder,r_hip)
            #### Visualize angle
            right_neck_text_position = (image.shape[1] - 220, 420)
            cv2.putText(image, str(f'Right neck angle: {right_neck_angle}'),
                           right_neck_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            avg_neck_angle = round((left_neck_angle+right_neck_angle)/2)

            ##2.8 GETTING KNEE ANGLES

            ### LEFT KNEE
            #### Calculate angle
            left_knee_angle = calculate_angle(l_hip, l_knee,l_ankle)
            #### Visualize angle
            left_knee_text_position = (image.shape[1] - 220, 450)
            cv2.putText(image, str(f'Left knee angle: {left_knee_angle}'),
                           left_knee_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                )

            ### RIGHT KNEE
            #### Calculate angle
            right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            #### Visualize angle
            right_knee_text_position = (image.shape[1] - 220, 480)
            cv2.putText(image, str(f'Right knee angle: {right_knee_angle}'),
                           right_knee_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            avg_knee_angle = round((left_knee_angle+right_knee_angle)/2)


            ## 2.9 SHOWING Z coordinates of a few different objects
            ### Display text 1
            l_wrist_z_text_position = (image.shape[1] - 250, 520)
            cv2.putText(image, str(f'z of L wrist: {round(l_wrist[2],4)}'),
                           l_wrist_z_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            ### Display text 2
            r_wrist_z_text_position = (image.shape[1] - 250, 550)
            cv2.putText(image, str(f'z of R wrist: {round(r_wrist[2],4)}'),
                           r_wrist_z_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            ### Display text 3
            l_shoulder_z_text_position = (image.shape[1] - 250, 580)
            cv2.putText(image, str(f'z of L shoulder: {round(l_shoulder[2],4)}'),
                           l_shoulder_z_text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            ### Display text 4
            r_shoulder_z_text_position = (image.shape[1] - 250, 610)
            cv2.putText(image, str(f'z of R shoulder: {round(r_shoulder[2],4)}'),
                           r_shoulder_z_text_position,
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
        # cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        # #4.2 Status data
        # cv2.putText(image, 'Quality', (15,12),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, quality,
        #             (10,60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

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

# print(l_shoulder)
# print(r_shoulder)
# print(l_hip)
# print(r_hip)
# print(l_ear)
# print(r_ear)
