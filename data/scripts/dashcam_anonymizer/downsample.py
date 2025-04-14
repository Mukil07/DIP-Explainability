import os
import torch 
import torchvision.transforms as transforms
import cv2
from torchvision.io import write_video

directory='/scratch/cvit/all_videos'
output_dir='/scratch/cvit/downsampled/'

def downsample_video(input_path, output_path):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return


    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

    #new_width = int(width * scale_factor)
    #new_height = int(height * scale_factor)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    frames = []
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frame_resized = frame.permute(1, 2, 0).numpy() * 255
        frames.append(frame_resized.astype('uint8'))
    
    cap.release()
    

    frames_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
    write_video(output_path, frames_tensor, fps=fps)

for videos in os.listdir(directory):
    source_path= os.path.join(directory,videos)
    destination_path= os.path.join(output_dir,videos)
    
    downsample_video(source_path, destination_path)
    


