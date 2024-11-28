import os 
import shutil
src = '/scratch/mukil/brain4cars_data/road_camera/'
dstn = '/scratch/mukil/brain4cars_data/road_cam/'

for root,dirs,files in os.walk(src):

    for f  in files:

        if f.endswith('txt'):
            src_path = os.path.join(root,f)
            shutil.copy(src_path,dstn)