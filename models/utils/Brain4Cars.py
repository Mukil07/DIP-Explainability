import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.transforms.functional import to_pil_image, to_grayscale
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import cv2

def visualize(frames):
    save_dir = 'visualize'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    for i,img in enumerate(frames):
        
        img = Image.fromarray(img)
        save = os.path.join(save_dir,f'img_{i}.jpg')
        img.save(save)
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def decode_video(video_path, num_frames=16):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = max(total_frames // num_frames, 1)
    frames = []

    for i in range(0, total_frames, interval):
        #import pdb;pdb.set_trace()
        x =  random.randint(0,interval-1)
        sum_ = i+x
        if sum_ <= total_frames:
            i = sum_
        else:
            i=i
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # frame = (frame/255.)*2 - 1
        frames.append(frame)

        if len(frames) == num_frames:
            break

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array(frames)


class CustomDataset(Dataset):
    def __init__(self,debug=None, transform=None):

        self.transform = transform
        self.debug = debug
        self.data = []
        self.classes_dict = {"rturn": 0, "rchange": 1, "lturn": 2, "lchange": 3, "endaction": 4}
        self.road_path = 'DATA/road_cam/'

        if self.debug:
            
            self.csv_path = f'DATA/face_cam/train_{self.debug}.csv'
        else:
            self.csv_path = 'DATA/face_cam/train.csv'

        self._load_data()

    def flip(self,target):

        if target == 0:
            return 2
        elif target == 2:
            return 0
        elif target == 1:
            return 3
        elif target == 3:
            return 1
        else:
            return target
        
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
        #import pdb;pdb.set_trace()
        if random.random() < 0.5:  
            video_frames1 = np.flip(video_frames1, axis=2)  # Flip width axis (W)
            video_frames2 = np.flip(video_frames2, axis=2) 
            target = self.flip(target)
        
        if self.transform:
            video_frames1 = self.transform(video_frames1)
            video_frames2 = self.transform(video_frames2)

        #import pdb;pdb.set_trace()
        video_frames1,video_frames2 = video_to_tensor(video_frames1.copy()),video_to_tensor(video_frames2.copy())
        video_frames1 = torch.nn.functional.interpolate(video_frames1, size=(224, 224), mode='bilinear')
        video_frames2 = torch.nn.functional.interpolate(video_frames2, size=(224, 224), mode='bilinear')
        return video_frames1,video_frames2,target, context

    def __len__(self):
        return len(self.data)