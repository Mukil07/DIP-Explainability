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
import random
import mukil.cemformer.utils.videotransforms as videotransforms

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

class flip:
    def __init__(self):
        self.horizontal_flip = videotransforms.RandomHorizontalFlip()

    def process_frames(self, video_frames1):
        return [self.horizontal_flip(img) for img in video_frames1]

class train_transform:
    def __init__(self):
        self.train_transforms = transforms.Compose([videotransforms.DriverFocusCrop(),
                                           ToTensor()
    ])

    def process_frames(self, video_frames1):
        return [self.train_transforms(img) for img in video_frames1]

class test_transform:
    def __init__(self):
        self.test_transforms = transforms.Compose([videotransforms.DriverCenterCrop(224),
                                          ToTensor()
                                        ])

    def process_frames(self, video_frames1):
        return [self.test_transforms(img) for img in video_frames1]
    
class CustomDataset(Dataset):
    def __init__(self,debug=None, mode= 'train'):
        self.mode = mode
        self.horizontal_flip = videotransforms.RandomHorizontalFlip()
        self.transform_train = train_transform()
        self.transform_test = test_transform()
        self.debug = debug
        self.data = []
        self.classes_dict = {"rturn": 0, "rchange": 1, "lturn": 2, "lchange": 3, "endaction": 4}
        self.resize_transform = transforms.Resize((224, 224))
        self.road_path = '/scratch/mukil/brain4cars_data/road_cam/'

        if self.debug:
            
            self.csv_path = f'/scratch/mukil/brain4cars_data/face_cam/train_{self.debug}.csv'
        else:
            self.csv_path = '/scratch/mukil/brain4cars_data/face_cam/train.csv'

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


        video_frames1 = decode_video(video_path1)
        video_frames2 = decode_video(road_path)
        

        video_frames1=  [Image.fromarray(img) for img in video_frames1]
        video_frames2 = [Image.fromarray(img) for img in video_frames2]
        self.horizontal_flip = flip()
        if self.mode == 'train':
            p = random.random()
            if p < 0.5:
                video_frames1 = self.horizontal_flip.process_frames(video_frames1)
                video_frames2 = self.horizontal_flip.process_frames(video_frames2)
                if target == 0:
                    target = 2
                elif target == 1:
                    target = 3
                elif target == 2:
                    target = 0
                elif target == 3:
                    target = 1 

        if  self.mode == 'train':
            try:
                video_frames1 = self.transform_train.process_frames(video_frames1)
                video_frames2 = self.transform_train.process_frames(video_frames2) 
            except:
                import pdb;pdb.set_trace()
        if  self.mode == 'test':
            video_frames1 = self.transform_test.process_frames(video_frames1)
            video_frames2 = self.transform_test.process_frames(video_frames2) 



        # if self.crop is not None: 


        # video_frames1 = torch.from_numpy(video_frames1).permute((0, -1, 1, 2))
        # video_frames2 = torch.from_numpy(video_frames2).permute((0, -1, 1, 2))
  
        video_frames1 = torch.vstack([img.unsqueeze(0) for img in video_frames1])
        video_frames2 = torch.vstack([img.unsqueeze(0) for img in video_frames2])
        # video_frames1 = video_frames1.permute((0, -1, 1, 2))
        # video_frames2 = video_frames2.permute((0, -1, 1, 2))

        video_frames1 = torch.nn.functional.interpolate(video_frames1, size=(224, 224), mode='bilinear')
        video_frames2 = torch.nn.functional.interpolate(video_frames2, size=(224, 224), mode='bilinear')

        ### Normalize the resized batch
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Add batch and spatial dimensions
        # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # #import pdb;pdb.set_trace()
        # video_frames1 = (video_frames1/255 - mean) / std
        # video_frames2 = (video_frames2/255 - mean) / std


        return video_frames1, video_frames2, target, context