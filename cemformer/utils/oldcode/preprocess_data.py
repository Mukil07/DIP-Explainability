import os 
import shutil
src = '/scratch/mukil/brain4cars_data/face_cam/'
dstn = '/scratch/mukil/brain4cars_data/face_cam/'

for root,dirs,files in os.walk(src):

    for f  in files:

        if f.endswith('avi'):

            new_name = f.replace('video_', '')
            # Full path for the old and new file
            old_file = os.path.join(root, f)
            new_file = os.path.join(root, new_name)

            # Rename the file
            os.rename(old_file, new_file)