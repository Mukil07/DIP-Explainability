import os 
import pandas as pd 
import re
import datetime
from moviepy.editor import VideoFileClip
import csv 
def numeric_key(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else 0
    

def subtract_time(t1,t2):
    time_format = "%H:%M:%S"

    t1 = datetime.strptime(time1, time_format) 
    t2 = datetime.strptime(time2, time_format)

    time_diff = t1 - t2

    is_negative = time_diff.total_seconds() < 0
    sign = "-" if is_negative else ""
    abs_time_diff = abs(time_diff)
    import pdb;pdb.set_trace()
    hours, remainder = divmod(abs_time_diff.seconds, 3600) 
    minutes, seconds = divmod(remainder, 60)
    
    if sign =='-':
        seconds= -1*seconds    
    print(f"{sign}{seconds} seconds")
    return seconds

directory= '/scratch/cvit/mukil/New/Right_View/'
offset_path= "/home2/shankar.gangisetty/dip_data/Preprocessing/output_csv/output_rightview.csv"
csv_filename = "/home2/shankar.gangisetty/dip_data/Preprocessing/output_csv/offset_rightview2.csv"


df = pd.read_csv(offset_path)
df_dict = df.to_dict(orient='records')

#import pdb;pdb.set_trace()
with open(csv_filename, mode='w', newline='') as file:
    
    writer = csv.writer(file)
    for days in os.listdir(directory):
    
        if days == 'Day11' or days == 'Day8':
            continue
        days_path= os.path.join(directory,days)
        video_list=[]
        for video in os.listdir(days_path):
        
            video_list.append(video)
        #import pdb;pdb.set_trace()  
        for i in df_dict:
        
            if days == 'Day'+ str(i['day']): 
                
                offset= i['offset']
                start_time= i['time'] #start time in the output.csv file (aria time)
                start_time_local= i['start_time'] #last date modified 
                break
            else: 
                #print(" day not found ")
                pass
                    
        sorted_list= sorted(video_list, key=numeric_key) 
        video_path= os.path.join(days_path,sorted_list[0])
        
        clip = VideoFileClip(video_path)
        duration = clip.duration  
        
        total_time_start= start_time_local - duration # subtract video_creation time with video duration 
        
        #import pdb;pdb.set_trace()
        #if days=='Day8'or days=='Day11':
            #i#@mport pdb;pdb.set_trace()
        #    continue
        #print(type(start_time), days)
        h1,m1,s1= int(start_time.split(':')[0]),int(start_time.split(':')[1]),int(start_time.split(':')[2])
        start_time= h1*3600+m1*60+s1
        total_time_start= total_time_start + offset 
        
        final_offset= total_time_start - start_time
        
     
            
    
        writer.writerow([days, final_offset])

        print(f"day {days} offset {final_offset}")
        
       # if days=='Day13':
        
            #import pdb;pdb.set_trace()

    
    
    


       
         