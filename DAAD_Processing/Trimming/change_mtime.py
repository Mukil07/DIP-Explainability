import os 
import json 
def load_and_apply_mod_times(mod_times_file, dest_dir):
    with open(mod_times_file, 'r') as f:
        mod_times = json.load(f)
    for relative_path, mod_time in mod_times.items():
        # Construct the destination file path
        #import pdb;pdb.set_trace()
        dest_file_path = os.path.join(dest_dir, relative_path)
        
        if os.path.exists(dest_file_path):
            os.utime(dest_file_path, (mod_time, mod_time))
            print(f"Updated modification time for {dest_file_path}")
        else:
            print(f"File {dest_file_path} does not exist in the destination directory") 
            
if __name__ == "__main__":
    mod_times_file = "/home2/shankar.gangisetty/dip_data/Preprocessing/mod_time/mod_times_rearview.json" # File with modification times
    dest_dir = "/scratch/cvit/mukil/Rear_View/" # Destination directory
    
    load_and_apply_mod_times(mod_times_file, dest_dir)
