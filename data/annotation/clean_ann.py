import csv
import ast

# Input and output file paths
input_file = '/scratch/mukil_wasi/cvit/v3/exps/FINAL.csv'
output_file = '/scratch/mukil_wasi/cvit/v3/exps/DIPX_new2.csv'

# Initialize a dictionary to store the final results
Final_dict = {}

# Open the input file for reading
with open(input_file, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    
    for row in reader:
        try:
            lst = ast.literal_eval(row[0])  # Extract the .mp4 file list
        except:
            print("Error parsing file list:", row)
            continue

        try:
            dict_data = ast.literal_eval(row[1])  # Extract the dictionary
        except:
            print("Error parsing dictionary:", row)
            continue

        try:
            dict_data['Activity'] = dict_data['Activity'].replace('\n', '')
        except:
            print("Error replacing newline in 'Activity':", row)
            continue

        word = lst[0]
        if word not in Final_dict:
            Final_dict[word] = {'Level-1': '', 'Level-2': '', 'Level-3': ''}

        # Store the activity for different phases
        try:
            if dict_data['Phase'] == 'Level-1 Annotation':
                Final_dict[word]['Level-1'] = dict_data['Activity']  # Store Level-1 activity
        except:
            print("Phase 'Level-1' not found:", row)
        
        try:
            if dict_data['Phase'] == 'Level-2 Annotation':
                Final_dict[word]['Level-2'] = dict_data['Activity']  # Store Level-2 activity
        except:
            print("Phase 'Level-2' not found:", row)
        
        try:
            if dict_data['Phase'] == 'Level-3 Annotation':
                if Final_dict[word]['Level-3']:  # If Level-3 already has a value, append to it
                    Final_dict[word]['Level-3'] += ', ' + dict_data['Activity']
                else:
                    Final_dict[word]['Level-3'] = dict_data['Activity']
        except:
            print("Phase 'Level-3' not found:", row)

# Writing the output to the CSV file
with open(output_file, mode='w', newline='') as outfile:
    # Define the CSV headers
    fieldnames = ['Name', 'Level-1', 'Level-2', 'Level-3']
    
    # Create a DictWriter object
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write each row to the output CSV file
    for name, annotations in Final_dict.items():
        row_data = {
            'Name': name,
            'Level-1': annotations.get('Level-1', ''),
            'Level-2': annotations.get('Level-2', ''),
            'Level-3': annotations.get('Level-3', ''),  # Comma-separated Level-3 annotations
        }
        writer.writerow(row_data)

print(f"Data written to {output_file}")
