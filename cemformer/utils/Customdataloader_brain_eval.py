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

def decode_video(video_path, num_frames=16):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = max(total_frames // num_frames, 1)
    frames = []

    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        if len(frames) == num_frames:
            break

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array(frames)

class CustomDataset(Dataset):
    def __init__(self,  debug=None, transform=None):
        
        self.transform = transform
        self.debug = debug
        self.data = []
        self.classes_dict = {"rturn": 0, "rchange": 1, "lturn": 2, "lchange": 3, "endaction": 4}
        self.resize_transform = transforms.Resize((224, 224))
        self.road_path = '/scratch/mukil/brain4cars_data/road_cam/'
        if self.debug:
            
            self.csv_path = f'/scratch/mukil/brain4cars_data/face_cam/train_{self.debug}.csv'
        else:
            import pdb;pdb.set_trace()
            self.csv_path = '/scratch/mukil/brain4cars_data/face_cam/val.csv'

        self._load_data()

    def _load_data(self):
        with open(self.csv_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                name = os.path.basename(row[0])
                road_path = os.path.join(self.road_path, name)

                if os.path.exists(road_path):
                    target = row[1]
                    context_file_path = road_path.replace('.avi', '.txt')
                    
                    # Load context data
                    with open(context_file_path, 'r') as context_file:
                        context = list(map(int, context_file.read().split(',')))
                    
                    # Store the paths and target without loading video frames
                    self.data.append((row[0], road_path, int(target), context))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path1, road_path, target, context = self.data[idx]

        # Load video frames only when accessing an item
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        video_frames1 = decode_video(video_path1)
        video_frames2 = decode_video(road_path)

        video_frames1 = torch.from_numpy(video_frames1).permute((0, -1, 1, 2))
        video_frames2 = torch.from_numpy(video_frames2).permute((0, -1, 1, 2))

        video_frames1 = self.resize_transform(video_frames1)
        video_frames2 = self.resize_transform(video_frames2)

        return video_frames1, video_frames2, target, context