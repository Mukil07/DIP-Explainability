import os 
import shutil
directory='/scratch/cvit/mukil/output/Rear_View_trimmed' 
output_dir='/scratch/cvit/mukil/output/rear_view_all'

for classes in os.listdir(directory):
    cls_path= os.path.join(directory,classes)
    for videos in os.listdir(cls_path):
        source_path= os.path.join(cls_path,videos)
        destination_path= os.path.join(output_dir,videos)
        shutil.move(source_path,destination_path)
        
        #with open(source_path, 'rb') as src_file:
        #  with open(destination_path, 'wb') as dest_file:
        #    dest_file.write(src_file.read())