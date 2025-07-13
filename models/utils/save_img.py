import numpy as np
import os
from PIL import Image
import cv2

def normalize(img):
    im = ((img - img.min()) / (img.max() - img.min())).permute((1, 2, 0)) * 255
    return im

def visualize(frames, num, label, video_name="./visualize_camvid/output_video.avi", fps=5):

    save_dir = f'visualize_cam'
    save_dirvid = f'visualize_camvid_{label}'
    if not os.path.exists(save_dirvid):
        os.mkdir(save_dirvid)
    # Save frames as images
    video_name = f"./visualize_camvid_{label}/vid{num}.avi"
    #import pdb;pdb.set_trace()os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    for i,img in enumerate(frames):
        
        # if not os.path.exists(save_dir):
            
        # for i, img in enumerate(batch):
            img = normalize(img)
            img = img.cpu().numpy().astype(np.uint8)
            img = Image.fromarray(img)
            save = os.path.join(save_dir, f'img_{i}.jpg')
            img.save(save)

 
    def extract_index(filename):
        return int(filename.split('_')[-1].split('.')[0])  #Extract the numeric part from "img_{num}.jpg"

    image_files = sorted(
        [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".jpg")],
        key=lambda x: extract_index(os.path.basename(x))
    )

    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    video.release()

    print(f"Video saved as {video_name}")