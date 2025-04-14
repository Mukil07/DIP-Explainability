import csv
import ast


input_file = '/scratch/mukil_wasi/cvit/v3/exps/FINAL.csv'
output_file = '/scratch/mukil_wasi/cvit/v3/exps/DIPX_new2.csv'

Final_dict = {}

with open(input_file, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    
    for row in reader:
        try:
            lst = ast.literal_eval(row[0]) 
        except:
            print("Error parsing file list:", row)
            continue

        try:
            dict_data = ast.literal_eval(row[1]) 
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

        try:
            if dict_data['Phase'] == 'Level-1 Annotation':
                Final_dict[word]['Level-1'] = dict_data['Activity']
        except:
            print("Phase 'Level-1' not found:", row)
        
        try:
            if dict_data['Phase'] == 'Level-2 Annotation':
                Final_dict[word]['Level-2'] = dict_data['Activity']
        except:
            print("Phase 'Level-2' not found:", row)
        
        try:
            if dict_data['Phase'] == 'Level-3 Annotation':
                if Final_dict[word]['Level-3']: 
                    Final_dict[word]['Level-3'] += ', ' + dict_data['Activity']
                else:
                    Final_dict[word]['Level-3'] = dict_data['Activity']
        except:
            print("Phase 'Level-3' not found:", row)


with open(output_file, mode='w', newline='') as outfile:
    
    fieldnames = ['Name', 'Level-1', 'Level-2', 'Level-3']

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()

    for name, annotations in Final_dict.items():
        row_data = {
            'Name': name,
            'Level-1': annotations.get('Level-1', ''),
            'Level-2': annotations.get('Level-2', ''),
            'Level-3': annotations.get('Level-3', ''),  
        }
        writer.writerow(row_data)

print(f"Data written to {output_file}")
