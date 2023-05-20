import csv
import os
from datetime import datetime


def write_to_csv(directory, class_idx, instance, **kwargs):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if the directory exists, create it if necessary
    print(str(class_idx))
    print(directory)
    class_folder = os.path.join(directory, str(class_idx))
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Construct the CSV filename based on the box_and_key
    csv_name = f"{instance}"

    # Open the CSV file in append mode
    with open(os.path.join(class_folder, csv_name + ".csv"), 'a', newline='') as file:
        writer = csv.writer(file)

        # If the file is empty, write the header row
        if file.tell() == 0:
            writer.writerow(['Timestamp'] + list(kwargs.keys()))

        # Write the data row to the CSV file
        writer.writerow([timestamp] + list(kwargs.values()))
