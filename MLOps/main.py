import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import time
from utils import load_video, get_angles, get_landmarks, get_video_dimensions, get_sideview
from metrics import get_reps_and_stage, get_rep_advice, get_neck, get_hip, get_knee, get_hand, get_hand_align, get_shoulder_elbow_dist
from visuals import show_neck, show_hip, show_knee, show_hand, show_align, show_elbow, show_status

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument('--static_image_mode', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--rev_color', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    static_image_mode = args.static_image_mode
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    rev_color = args.rev_color

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    while True:

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image01 = copy.deepcopy(image)
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]),
                    bg_color,
                    thickness=-1)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks is not None:
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
            )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('Tokyo2020 Debug', debug_image01)
        cv.imshow('Tokyo2020 Pictogram', debug_image02)

    cap.release()
    cv.destroyAllWindows()

def predict_on_stream(vid, writer, width: int, height :int):
    """
    Calculate synchronization scores & write frames to video
    """
    fps = int(vid.get(cv.CAP_PROP_FPS))
    width  = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    if writer =="avi":
        writer = cv.VideoWriter("video_proposture.avi",
        cv.VideoWriter_fourcc(*"MJPG"), fps, (width,height))
    elif writer =="mp4":
        writer = cv.VideoWriter("video_proposture.mp4",
        cv.VideoWriter_fourcc(*"mp4v"), fps, (width,height))

    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret==True:
            image = frame.copy()

            # Making prediction
            keypoints = draw_landmarks(image)
            keypoints= np.squeeze(np.multiply(keypoints, [height,width,1]))

            # write video
            writer.write(keypoints)
        else:
            break

    writer.release()

    return None

def draw_landmarks(
    image,
    landmarks='Show',
    # upper_body_only,
    visibility_th=0.5,
    video_settings='None',
    view='front',
    rep_counter = 0,
    stage = 'START'
):
    image_width, image_height = image.shape[1], image.shape[0]
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        results = pose.process(image)

        try:
                landmarks = results.pose_landmarks.landmark
        except:
                pass
        if video_settings == 'Show':
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                  mp.solutions.pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )
            return image
        if video_settings == 'Pushups aide':
            advice_list = []
            landmarks = get_landmarks(results)

            #if sideview, get sideview angle
            if view == 'side':
                sideview_angle = get_sideview(*landmarks)
            if view == 'front':
                sideview_angle = None

            # Calculate angles of joints
            angles = get_angles(*landmarks)

            # Name joint angle variables
            elbow_angles = angles[:2]
            neck_angles = angles[2:4]
            wrist_angles = angles[4:6]
            shoulder_angles = angles[6:8]
            shoulder_distance = angles[12:14]
            shoulder_elbow_distance = angles[14:16]
            hip_angles = angles[8:10]
            knee_angles = angles[10:12]

            #get status box
            reps_stage = get_reps_and_stage(elbow_angles, rep_counter, stage)
            rep_advice = get_rep_advice(elbow_angles, sideview_angle)
            show_status(image, rep_advice, *reps_stage)

            #update rep_advice, stage, rep_counter for next loop
            rep_advice = rep_advice
            stage = reps_stage[0]
            rep_counter = reps_stage[1]

            # TODO: based on view, implement different get_metrics function
            if view == 'front':
                align = get_hand_align(shoulder_distance, elbow_angles)
                elbow = get_shoulder_elbow_dist(shoulder_elbow_distance, elbow_angles)

                # Visualize status on joints
                align_status = show_align(image, align, image_height, image_width, *landmarks)
                align_status
                if not (align[1] in advice_list):
                    advice_list.append(align[1])

                elbow_status = show_elbow(image, elbow, image_height, image_width, *landmarks)
                elbow_status
                if not (elbow[1] in advice_list):
                    advice_list.append(elbow[1])

            #get status and advice
            if view == 'side':
                neck = get_neck(neck_angles, sideview_angle)
                hand = get_hand(shoulder_distance, wrist_angles, sideview_angle)
                hip = get_hip(hip_angles, sideview_angle)
                knee = get_knee(knee_angles, sideview_angle)

                # Visualize status on joints
                neck_status = show_neck(image, neck, sideview_angle, image_height, image_width, *landmarks)
                neck_status
                if not (neck[1] in advice_list):
                    advice_list.append(neck[1])

                hand_status = show_hand(image, hand, sideview_angle, image_height, image_width, *landmarks)
                hand_status
                if not (hand[1] in advice_list):
                    advice_list.append(hand[1])

                hip_status = show_hip(image, hip, sideview_angle, image_height, image_width, *landmarks)
                hip_status
                if not (hip[1] in advice_list):
                    advice_list.append(hip[1])

                knee_status = show_knee(image, knee, sideview_angle, image_height, image_width, *landmarks)
                knee_status
                if not (knee[1] in advice_list):
                    advice_list.append(knee[1])

            # Display advice text
            if len(advice_list) > 0:
                cv.rectangle(image, (0, 80), (400, 120), (36, 237, 227), -1)
                cv.putText(image, "POSTURE ADVICE:", (15, 100),
                            cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

                for i, advice in enumerate(advice_list):
                    rectangle_y = 120 + i * 30  # Adjust the y-coordinate of the rectangle based on the index
                    text_y = 130 + i * 30  # Adjust the y-coordinate of the text based on the index

                    cv.rectangle(image, (0, rectangle_y), (400, rectangle_y + 30), (36, 237, 227), -1)
                    cv.putText(image, advice, (15, text_y),
                                cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)




            #Render detections
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                  mp.solutions.pose.POSE_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                                  mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )

            return image
    return image

if __name__ == '__main__':
    main()
