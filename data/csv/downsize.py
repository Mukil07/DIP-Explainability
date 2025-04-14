import os
#from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.editor as mp
# Define source and destination directories
source_dir = '/scratch/cvit/v3/Driver_View/'  # Current directory 
output_dir = '/scratch/cvit/v3/driver_view_256/'  # Output directory 

# Create the output directory if it doesn't exist
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

def process_directory(source_dir, output_dir):
    """Recursively process all .mp4 files in the source directory and resize them."""
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.mp4'):
                # Construct full file paths
                #import pdb;pdb.set_trace()
                input_file = os.path.join(root, file)
                relative_path = input_file.split('/')[-2]
                output_subdir = os.path.join(output_dir, relative_path)
                
                # Ensure the output directory exists
                os.makedirs(output_subdir, exist_ok=True)
                
                # Output file path
                output_file = os.path.join(output_subdir, file)
                
                # Resize the video
                print(f"Resizing {input_file} -> {output_file}")
                resize_video(input_file, output_file)

if __name__ == '__main__':
    # Call the function to process the directory
    process_directory(source_dir, output_dir)

