import os
#from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.editor as mp

source_dir = '/scratch/cvit/v3/Driver_View/'  # Current directory 
output_dir = '/scratch/cvit/v3/driver_view_256/'  # Output directory 

os.makedirs(output_dir, exist_ok=True)

def resize_video(input_file, output_file):
    clip = mp.VideoFileClip(input_file)

    width, height = clip.size
    if width > height:
        new_height = 256
        new_width = int(clip.w * (256 / clip.h))
    else:
        new_width = 256
        new_height = int(clip.h * (256 / clip.w))

    resized_clip = clip.resize((new_width, new_height))
    resized_clip.write_videofile(output_file, codec='libx264')

def process(source_dir, output_dir):
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.mp4'):

                #import pdb;pdb.set_trace()
                input_file = os.path.join(root, file)
                relative_path = input_file.split('/')[-2]
                output_subdir = os.path.join(output_dir, relative_path)
                
                os.makedirs(output_subdir, exist_ok=True)
                output_file = os.path.join(output_subdir, file)

                print(f"Resizing {input_file} -> {output_file}")
                resize_video(input_file, output_file)

if __name__ == '__main__':
    # resize all the videos in the folder. The largest side will be resized to 256 while maintaining the aspect ratio.   
    process(source_dir, output_dir)

