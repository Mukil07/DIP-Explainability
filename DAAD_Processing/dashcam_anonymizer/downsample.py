import os
import torch 
import torchvision.transforms as transforms
import cv2
from torchvision.io import write_video

directory='/scratch/cvit/all_videos'
output_dir='/scratch/cvit/downsampled/'

def downsample_video(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate new dimensions
    #new_width = int(width * scale_factor)
    #new_height = int(height * scale_factor)
    
    # Prepare transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Initialize a list to store the processed frames
    frames = []
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply the transformation
        frame_tensor = transform(frame_rgb)
        
        # Convert back to numpy and append to frames list
        frame_resized = frame_tensor.permute(1, 2, 0).numpy() * 255
        frames.append(frame_resized.astype('uint8'))
    
    # Release the video capture object
    cap.release()
    
    # Convert frames to a tensor
    frames_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
    
    # Write the video
    write_video(output_path, frames_tensor, fps=fps)

for videos in os.listdir(directory):
    source_path= os.path.join(directory,videos)
    destination_path= os.path.join(output_dir,videos)
    
    downsample_video(source_path, destination_path)
    


