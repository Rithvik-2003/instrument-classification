import csv
import os

data_path = '../IRMAS-TrainingData/'

classes = {
    'cel': 0,
    'cla': 1,
    'flu': 2,
    'gac': 3,
    'gel': 4,
    'org': 5,
    'pia': 6,
    'sax': 7,
    'tru': 8,
    'vio': 9,
    'voi': 10
}

csv_file = 'dataset.csv'

csv_data = []

for instrument_class, label_number in classes.items():
    folder_path = os.path.join(data_path, instrument_class)
    files = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]
    for filename in files:
        if filename != 'desktop.ini':
            csv_data.append([filename, instrument_class, label_number])

with open(csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['filename', 'label', 'label_number'])
    csv_writer.writerows(csv_data)
    