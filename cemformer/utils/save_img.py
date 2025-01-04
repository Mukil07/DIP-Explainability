import numpy as np 
import os 
from PIL import Image

def normalize(img):
    im = ((img - img.min()) / (img.max() - img.min())).permute((1,2,0))*255
    return im

def visualize(frames):
    save_dir = 'visualize_cam'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

   # import pdb;pdb.set_trace()
    for i,img in enumerate(frames):
        img = normalize(img)
        img = img.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        save = os.path.join(save_dir,f'img_{i}.jpg')
        img.save(save)

