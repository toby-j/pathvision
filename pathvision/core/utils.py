import csv
import os
from datetime import datetime


def write_to_csv(frame, class_name, percentage_overlap, csv_name):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if the file exists
    file_exists = os.path.isfile(csv_name + ".csv")

    # Open the CSV file in append mode
    with open(csv_name + ".csv", 'a', newline='') as file:
        writer = csv.writer(file)

        # If the file doesn't exist, write the header row
        if not file_exists:
            writer.writerow(['Timestamp', 'String Value', 'Integer Value'])

        # Write the data row to the CSV file
        writer.writerow([frame, timestamp, class_name, percentage_overlap])