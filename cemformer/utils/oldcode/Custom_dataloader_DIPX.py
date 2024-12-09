import argparse

import torch.optim as optim

from torchvision import datasets, transforms
import torch
from tqdm.auto import tqdm
import os
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms.functional import to_pil_image, to_grayscale
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import cv2 
import numpy as np
import pandas as pd


def decode_video(video_path, start,end, num_frames=16):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)


    frames = []

    if start <=4:
            
        start_time = 0
        end_time = end*frame_rate
    else:
        start_time = start*frame_rate - 4*frame_rate
        end_time = end*frame_rate     
    total_frames = end_time - start_time
    interval = max(total_frames // num_frames, 1)

    for i in range(int(start_time), int(end_time), int(interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        if len(frames) == num_frames:
            break

    cap.release()
    try:
        while len(frames) < num_frames:
            frames.append(frames[-1])
    except:
        print(len(frames),start,end,frame_rate)
        print("AAAAAAAAAAAAAA",video_path)
    return np.array(frames)


class CustomDataset(Dataset):
    def __init__(self, debug=None, transform=None):
        self.transform = transform
        self.debug = debug
        self.data = []
        #self.classes_dict = {"rturn": 0, "rchange": 1, "lturn": 2, "lchange": 3, "endaction": 4}
        self.resize_transform = transforms.Resize((224, 224))
        self.road_path = '/scratch/mukil/dipx/common/front_view_common'
        self.face_path = '/scratch/mukil/dipx/common/driver_common'
        self.time = '/scratch/mukil/dipx/time.csv'
        self.df = pd.read_csv(self.time)


        if self.debug:
            
            self.csv_path = f'/scratch/mukil/dipx/train_{self.debug}.csv'
        else:
            self.csv_path = '/scratch/mukil/dipx/train.csv'

        self._load_data()

    def _load_data(self):
        with open(self.csv_path, 'r') as csv_file:

            reader = csv.reader(csv_file)
            for row in reader:
                
                name = row[0]
                target = row[1]
                gaze = row[2]
                ego = row[3]
                ego = list(map(int, ego.strip('[]').split()))
                ego = [torch.tensor(ego,dtype=torch.float) for i in ego]
                gaze = torch.tensor(int(gaze))
                road_path = os.path.join(self.road_path, name)
                face_path = os.path.join(self.face_path, name)
                if os.path.exists(road_path) and os.path.exists(face_path) :

                    
                    # Store the paths and target without loading video frames
                    self.data.append((face_path, road_path,int(target),gaze,ego[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path1, road_path, target, gaze, ego = self.data[idx]
        start,end=0,0
        for i, row in self.df.iterrows():
            if row['name'] in video_path1.split('/')[-1]:
                start=int(row['start'].split(':')[-1])
                end=int(row['end'].split(':')[-1])
                break
        if end == 0:
            cap = cv2.VideoCapture(video_path1)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video {video_path1}")

            end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load video frames only when accessing an item
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        video_frames1 = decode_video(video_path1,start,end)
        video_frames2 = decode_video(road_path,start,end)

        video_frames1 = torch.from_numpy(video_frames1).permute((0, -1, 1, 2))
        video_frames2 = torch.from_numpy(video_frames2).permute((0, -1, 1, 2))

        video_frames1 = self.resize_transform(video_frames1)
        video_frames2 = self.resize_transform(video_frames2)

        return video_frames1, video_frames2, target, gaze, ego