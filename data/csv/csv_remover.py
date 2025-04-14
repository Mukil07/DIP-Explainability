import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import os

file_path = '/scratch/cvit/v3/val.csv'
new_file_path = '/scratch/cvit/v3/val_final.csv'  # Replace with your CSV file path


G=[]
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        #import pdb;pdb.set_trace()
        files = row[0].split(' ')
        if os.path.exists(files[1]):
            G.append(row)


with open(new_file_path, mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(G)