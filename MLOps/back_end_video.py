from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import subprocess
import os
import uuid
import cv2
import numpy as np
from main import draw_landmarks

app = FastAPI()
gcs = storage.Client()
bucket = gcs.get_bucket(os.environ.get('BUCKET'))

def load_video_and_release(path : str, output_format: str, output_name :str):
    """
    load video and define output format
    """
    # Conversion on the video in a opencv Videocapture (collection of frames)
    vid = cv2.VideoCapture(path)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video analysed: \n fps: {fps}, *\
          \n frame count: {frame_count} , \n width : {width}, \n height : {height}")

    # creation onf the writer to recompose the video later on
    if output_format =="avi":
        writer = cv2.VideoWriter(f"{output_name}.avi",
        cv2.VideoWriter_fourcc(*"MJPG"), fps,(width,height))
    elif output_format =="mp4":
        writer = cv2.VideoWriter(f"{output_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"), fps,(width,height))

    return vid, writer, fps, frame_count, width, height

@app.post("/vid_stats")
def stats_to_st(file: UploadFile = File(...)):
    # video file loading
    vid_name = file.filename
    uploaded_video = file.file
    output_name = 'output'


    # open video file
    with open(vid_name, mode='wb') as f:
        f.write(uploaded_video.read())


    # video stats
    cap = cv2.VideoCapture(vid_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    return {
        'frame_count': frame_count,
        'fps': fps,
        'dim': f'{width} x {height}',
        'width': width,
        'height': height,
        'vid_name': vid_name,
        'output_name': output_name
            }

@app.get("/vid_process")
def process_vid(vid_name, output_name, frame_count, fps, width, height, dancers, face_ignored, conf_threshold, confidence_display):

    # load video
    vid, writer, _, _, _, _ = load_video_and_release(vid_name, output_format="mp4", output_name=output_name)

    #return vid , all_scores, all_people, all_link_mae , worst_link_scores , worst_link_names
    vid, all_scores, _, _, worst_link_scores, worst_link_names, ignore_frame = draw_landmarks(vid, writer, app.state.model, int(width), int(height), int(dancers), bool(face_ignored), float(conf_threshold), confidence_display=False)

    # time in seconds
    timestamps = np.arange(int(frame_count))/int(fps)

    # compress video output to smaller size
    my_uuid = uuid.uuid4()
    output_lite = f'output_lite_{my_uuid}.mp4'
    current_dir = os.path.abspath('.')
    result = subprocess.run(f'ffmpeg -i {current_dir}/{output_name}.mp4 -b:v 2500k {current_dir}/{output_lite} -y', shell=True)
    print(result)


    # upload video to google cloud storage
    vid_blob = bucket.blob(output_lite)
    vid_blob.upload_from_filename(output_lite)


    # # clean screencaps in google cloud storage
    # blobs = bucket.list_blobs(prefix='screencaps')
    # for blob in blobs:
    #     blob.delete()


    # upload screencaps to google cloud storage
    for i in range(int(frame_count)):
        blob = bucket.blob(f"screencaps/{my_uuid}/frame{i}.jpg")
        blob.upload_from_filename(f"{os.path.abspath('.')}/api/screencaps/frame{i}.jpg")


    return {
        'output_url': vid_blob.public_url,
        'timestamps': list(timestamps),
        'scores': all_scores,
        'my_uuid': my_uuid,
        'link_scores': worst_link_scores,
        'link_names': worst_link_names,
        'scores_bool': ignore_frame
    }
