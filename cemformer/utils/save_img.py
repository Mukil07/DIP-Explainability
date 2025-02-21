import numpy as np
import os
from PIL import Image
import cv2

def normalize(img):
    im = ((img - img.min()) / (img.max() - img.min())).permute((1, 2, 0)) * 255
    return im

def visualize(frames, num, video_name="./visualize_camvid/output_video.avi", fps=5):
    save_dir = 'visualize_cam'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dirvid = f'visualize_camvid'
    if not os.path.exists(save_dirvid):
        os.mkdir(save_dirvid)
    # Save frames as images
    video_name = f"./visualize_camvid/vid{num}.avi"
    for i, img in enumerate(frames):
        img = normalize(img)
        img = img.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        save = os.path.join(save_dir, f'img_{i}.jpg')
        img.save(save)

    # Gather all image files and sort numerically by the index
    def extract_index(filename):
        return int(filename.split('_')[-1].split('.')[0])  # Extract the numeric part from "img_{num}.jpg"

    image_files = sorted(
        [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".jpg")],
        key=lambda x: extract_index(os.path.basename(x))
    )

    # Read the first image to determine video frame size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'mp4v' for MP4
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Write each image to the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    # Release the video writer
    video.release()

    print(f"Video saved as {video_name}")