import os
import cv2
import time
import pandas as pd
from datetime import datetime

def timestamp_to_seconds(timestamp_str):
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %I:%M:%S %p')     
    return timestamp.timestamp()

def get_video_capture_times(video_path):
    if not os.path.exists(video_path):
        print('haha')
        return None  # Return None if the video file doesn't exist
    timestamp = os.path.getmtime(video_path)
    capture_end_date_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %I:%M:%S %p')
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = 30
    video_duration_seconds = frame_count / frame_rate
    end_timestamp = timestamp - video_duration_seconds
    capture_start_date_time = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %I:%M:%S %p')
    cap.release()
    return capture_start_date_time, capture_end_date_time

def extract_segment(input_video, output_video, start_time, end_time, start_time_seconds, end_time_seconds):
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Error: Could not open input video file.")
        return

    frame_rate = 30
    start_frame = int((start_time - start_time_seconds) * frame_rate)
    end_frame = int((end_time - start_time_seconds) * frame_rate)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break
        out.write(frame)

    cap.release()   
    out.release()      

def process_videos_in_folder(video_folder, csv_file, output_dir, time_offset):    
    #check the offset time manually by figuring out the timestamp difference between Meta Aria and GoPro
    if not os.path.exists(video_folder):  
        print(f"Video folder '{video_folder}' not found.")        
        return

    if not os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_file)
    
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            video_path = os.path.join(root, file)
            start_time, end_time = get_video_capture_times(video_path)
            start_time_seconds = timestamp_to_seconds(start_time)
            end_time_seconds = timestamp_to_seconds(end_time)

            for index, row in df.iterrows():
                telemetry_id = row['Telemetry ID']  
                csv_start_time = row['Start Time (IST)']     
                csv_end_time = row['End Time (IST)']
                start_seconds = timestamp_to_seconds(csv_start_time) + time_offset -1
                end_seconds = timestamp_to_seconds(csv_end_time) + time_offset +1
                #import pdb;pdb.set_trace()
                if start_time_seconds <= start_seconds <= end_time_seconds and start_time_seconds <= end_seconds <= end_time_seconds:
                    #Fprint('yaaaey')
                    output_file = os.path.join(output_dir, f'{telemetry_id}.mp4')
                    extract_segment(video_path, output_file, start_seconds, end_seconds, start_time_seconds, end_time_seconds)
                
  

offset_path= "/home2/shankar.gangisetty/dip_data/Preprocessing/output_csv/offset_rearview.csv"
df = pd.read_csv(offset_path)
df_dict = df.to_dict(orient='records')

for i in df_dict:
    day=i['day']
    time_offset=i['offset']

    directory= f"/scratch/cvit/mukil/Rear_View/{day}"
    csv_file = "/home2/shankar.gangisetty/dip_data/Preprocessing/output.csv"
    output_dir = f'/scratch/cvit/mukil/output/Rear_View_trimmed/{day}'
    process_videos_in_folder(directory, csv_file, output_dir,time_offset)






