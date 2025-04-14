import os
import json

def save_mod_times(directory, output_file):
    mod_times = {}

    # Walk through the directory and get modification times
    for days in os.listdir(directory):
        day_path= os.path.join(directory,days)
        
        for video in os.listdir(day_path):
            
    #for root, dirs, files in os.walk(directory):
        #for file in files:
            file_path = os.path.join(day_path, video)
            relative_path = os.path.relpath(file_path, start=directory)
            mod_times[relative_path] = os.path.getmtime(file_path)
    
    # Save modification times to a file
    with open(output_file, 'w') as f:
        json.dump(mod_times, f)

if __name__ == "__main__":
    directory = "/mnt/base/dip/DIP_Views/other_views/Left_View/"  # Specify the directory to scan
    output_file = "mod_times.json"     # File to save modification times
    
    save_mod_times(directory, output_file)
    print(f"Modification times saved to {output_file}")
