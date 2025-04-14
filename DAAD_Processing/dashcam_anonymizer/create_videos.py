import os
import glob
import json
import cv2
import pybboxes as pbx
import yaml
import argparse
from ultralytics import YOLO
import shutil
import pdb
parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)



#Annotation JSONS stored in annot_jsons folder.

def blur_regions(image, regions):
    """
    Blurs the image, given the x1,y1,x2,y2 cordinates using Gaussian Blur.
    15,15 gaussian radius enough to anonymyze (may increase if needed)
    """
    for region in regions:
        x1,y1,x2,y2 = region
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        roi = image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (config["blur_radius"], config["blur_radius"]), 0)
        image[y1:y2, x1:x2] = blurred_roi
    return image



#model = YOLO(config["model_path"])


#import pdb;pdb.set_trace()
videos = glob.glob(config["videos_path"]+"/*.mp4")
if not(os.path.exists(config["output_folder"])):
    os.mkdir(config["output_folder"])

anonymized_videos_path = config["output_folder"]

for video in videos:
    #pdb.set_trace()
    video_number = video.split('/')[-1].replace(".mp4","")
    json_path = '/scratch/cvit/new/annot_jsons2/'+video_number+'.json'
    if(os.path.exists(json_path)):
        with open(json_path) as F:
            #Data is the json dictionary in which key is the frame, and value is a list of lists.
            data = json.load(F)

            #video writer setup
            #import pdb;pdb.set_trace()
            video_capture = cv2.VideoCapture(video)
            out_vid_path = anonymized_videos_path+video.split('/')[-1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_width = int(video_capture.get(3))
            frame_height = int(video_capture.get(4))
            frame_size = (frame_width,frame_height)
            fps = config["vid_fps"]
            output_video = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'avc1'), fps, frame_size)
            #output_video = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'H264'), fps, frame_size)
            count = 1
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                try:
                    #the frame has some detection
                    frame = blur_regions(frame, data[str(count)])
                except:
                    #otherwise write the frame as is
                    frame = frame
                output_video.write(frame)
                count+=1
            video_capture.release()
            output_video.release()
        print(f"Processed Video {video_number}")
    else:
        
        video_number = video.split('/')[-1].replace(".mp4","")
        print(f"No objects in file {video}, copying file as is. PLEASE CHECK DETECTOR AGAIN.")
        shutil.copy(video, anonymized_videos_path)
        print(f"Processed Video {video_number}")

print(f"@@ The bluured videos are saved in Directory -------> {config['output_folder']}")
