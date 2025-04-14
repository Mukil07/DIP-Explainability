import os
import numpy as np
import csv
with open("/scratch/cvit/v3/Annotation_labels_final.csv", 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
data= np.array(data)
data= data[1:]
 
#import pdb;pdb.set_trace()
#directory='/mnt/base/dip/DIP_Views/Annotated videos (gopro)/'
directory= '/scratch/cvit/v3/left_view_blurred/'
new_dir='/scratch/cvit/v3/processed/'
d1= os.listdir(directory)
for file in data:
  #import pdb;pdb.set_trace()
  for i in d1: # i is day folder
    #day_path= os.path.join(directory,i)
    #for k in os.listdir(day_path): # gives all videos names
    v_path=os.path.join(directory,i) # gives all video name

      
      #for j in os.listdir(v_path):
    print(i,v_path)
    if file[0].split('.')[0]==i.split('.')[0]:
        #source_path= os.path.join(v_path,i)
        source_path = v_path

        if file[1] == 'Left':
            if file[2] == 'Turn':
                path = os.path.join(new_dir, 'Left_Turn')
            elif file[2] == 'Lane Change':
                path = os.path.join(new_dir, 'Left_Change')
        
        elif file[1] == 'Right':
            if file[2] == 'Turn':
                path = os.path.join(new_dir, 'Right_Turn')
            elif file[2] == 'Lane Change':
                path = os.path.join(new_dir, 'Right_Change')
        
        elif file[1] == 'Straight':
            path = os.path.join(new_dir, 'Straight')
        elif file[1] == 'Slow Down':
            path = os.path.join(new_dir, 'Slow_Down')
        elif file[1] == 'U Turn':
            path = os.path.join(new_dir, 'U_Turn')
        elif file[1] == 'Stop':
            path = os.path.join(new_dir, 'Slow_Down')
        
        destination_path = os.path.join(path, i)

        if not os.path.exists(path):
            os.makedirs(path)

        with open(source_path, 'rb') as src_file:
            with open(destination_path, 'wb') as dest_file:
                dest_file.write(src_file.read())
