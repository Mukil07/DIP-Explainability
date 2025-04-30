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
from torchvision import datasets, transforms
import pandas as pd 

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

def visualize(frames):
    save_dir = 'visualize'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    for i,img in enumerate(frames):
        
        img = Image.fromarray(img)
        save = os.path.join(save_dir,f'img_{i}.jpg')
        img.save(save)

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

def decode_video(video_path, start,end, num_frames=16):
    flag = False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)


    frames = []

    if start <=4:
            
        start_time = 0
        end_time = end*frame_rate + 1*frame_rate
    else:
        start_time = start*frame_rate - 4*frame_rate
        end_time = end*frame_rate + 1*frame_rate
    total_frames = end_time - start_time
    interval = max(total_frames // num_frames, 1)

    for i in range(int(start_time), int(end_time), int(interval)):
        x =  random.randint(0,int(interval)-1)
        sum_ = i+x
        if sum_ <= int(end_time):
            i = sum_
        else:
            i=i
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
        # print(len(frames),start,end,frame_rate)
        # print("AAAAAAAAAAAAAA",video_path)
        flag=True
        
    return np.array(frames),flag


class CustomDataset(Dataset):
    def __init__(self,csv_type, debug=None, transform=None):

        self.transform = transform
        self.debug = debug
        self.data = []
        #self.classes_dict = {"rturn": 0, "rchange": 1, "lturn": 2, "lchange": 3, "endaction": 4}


        self.resize_transform = transforms.Resize((224, 224))
        self.road_path = '/scratch/mukilv2/dipx/common/front_view_common'
        self.face_path = '/scratch/mukilv2/dipx/common/aria_common'
        self.time = '/scratch/mukilv2/dipx/time.csv'
        self.df = pd.read_csv(self.time)

        if self.debug:
            
            self.csv_path = f'/scratch/mukilv2/dipx/train_{self.debug}.csv'
        else:
            self.csv_path = csv_type

        self._load_data()

    def flip(self,target):

        if target == 2:
            return 4
        elif target == 4:
            return 2
        elif target == 3:
            return 5
        elif target == 5:
            return 3
        else:
            return target 
    def _load_data(self):
        with open(self.csv_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                name = os.path.basename(row[0])
                face_path = os.path.join(self.face_path, name)
                road_path = os.path.join(self.road_path, name)

                if os.path.exists(road_path):
                    #import pdb;pdb.set_trace()
                    target = int(row[1])
                    gaze = int(row[2])
                    ego = row[3]
                    
                    ego = list(map(int, ego.strip('[]').split()))
                    
                    #ego = [ego for i in ego]

                    self.data.append((face_path, road_path, target, gaze, ego))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            face_path, road_path, target, gaze, ego = self.data[idx]

            start,end=0,0
            for i, row in self.df.iterrows():
                if row['name'] in face_path.split('/')[-1]:
                    start=int(row['start'].split(':')[-1])
                    end=int(row['end'].split(':')[-1])
                    break
            if end == 0:
                cap = cv2.VideoCapture(face_path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video {face_path}")

                end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Load video frames only when accessing an item
            #import pdb;pdb.set_trace()
            
            fl1,fl2= False,False
            video_frames1,fl1 = decode_video(face_path, start, end)
            video_frames2,fl2 = decode_video(road_path, start, end)

            if fl1==True or fl2==True: 
                
                idx += 1
                if idx >= len(self.data):
                    idx = 0  
                    print("Reached end of dataset, resetting idx to 0")
                continue  
            
            if random.random() < -1:  
                video_frames1 = np.flip(video_frames1, axis=2)  # Flip width axis (W)
                video_frames2 = np.flip(video_frames2, axis=2) 
                target = self.flip(target)
                video_frames1,video_frames2 = video_to_tensor(video_frames1.copy()),video_to_tensor(video_frames2.copy())
            else:
                video_frames1,video_frames2 = video_to_tensor(video_frames1),video_to_tensor(video_frames2)
            if self.transform:
                
 

                video_frames1 = video_frames1.permute(1,0,2,3)
                video_frames2 = video_frames2.permute(1,0,2,3)
             
                for frame in video_frames1:
                    frame = self.transform((frame.to(torch.float32)/255))
                    
                for frame in video_frames2:
                    frame = self.transform((frame.to(torch.float32)/255))    
                                 
                video_frames1 = video_frames1.permute(1,0,2,3)
                video_frames2 = video_frames2.permute(1,0,2,3)  
            
            video_frames1 = torch.nn.functional.interpolate(video_frames1, size=(224, 224), mode='bilinear')
            video_frames2 = torch.nn.functional.interpolate(video_frames2, size=(224, 224), mode='bilinear')
            
            return video_frames1,video_frames2,target, gaze, ego 

    def __len__(self):
        return len(self.data)