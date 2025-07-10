import csv
import glob
import ntpath
import random
from math import floor

# Paths and URLs
survey_folder = r"/path/to/image-file/"  # Folder containing the images
base_url = r"https://model-dataset-name-bucket1.s3.eu-central-1.amazonaws.com/"
csv_file = r"dataset_mturk_urls_file.csv"              #r"imagenet_mturk_links.csv"
extra_csv_file = r" urls-file-of extra-links.csv"      # if any then put that file here#r"imagenet_mturk_links_extra.csv"
acq_url = base_url + "ACQ.png"                         #"vr_0_e281_pNone.png"

# Get all images and shuffle them randomly
images = glob.glob(survey_folder + '/*')
random.shuffle(images)

# Parameters
number_questions = 10  # Number of images per row is adjustable according to the plan
number_surveys = floor(len(images) / number_questions)  # Number of rows
remainder = len(images) % number_questions

# Adjust for extra row if images don't divide evenly
extra_survey = remainder != 0
if extra_survey:
    number_surveys += 1

# Create partitions for rows
partitions = [images[i:i + number_questions] for i in range(0, len(images), number_questions)]

# Writing the main CSV file
with open(csv_file, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Header with numbering starting from 0
    header = ["image_url" + str(i) for i in range(number_questions + 1)]
    writer.writerow(header)

    # Write rows for main CSV
    for i in range(number_surveys - (1 if extra_survey else 0)):
        row = []
        for element in partitions[i]:
            element_name = ntpath.basename(element)
            link = base_url + element_name
            row.append(link)
        row.append(acq_url)  # Add acq_url as the last image URL
        random.shuffle(row)  # Shuffle the row before saving
        writer.writerow(row)

# Writing the extra CSV file if there are leftover images
if extra_survey:
    with open(extra_csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # Header with adjusted numbering for extra file
        header = ["image_url" + str(i) for i in range(len(partitions[-1]) + 1)]
        writer.writerow(header)
        row = []
        for element in partitions[-1]:
            element_name = ntpath.basename(element)
            link = base_url + element_name
            row.append(link)
        row.append(acq_url)  # Add acq_url as the last image URL
        random.shuffle(row)
        writer.writerow(row)

print("CSV files created successfully.")
