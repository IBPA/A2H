import sys 
import os
import argparse
import csv
from shutil import copy

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="specify directory of files to filter")
parser.add_argument("--filter", help="specify a file with list of acceptable entities")
parser.add_argument("--output", help="specify directory to copy filtered files to")
args = parser.parse_args()

source_location = args.input if args.input else 'parsed_files'
filter_location = args.filter if args.filter else 'interventions_processed.csv'
output_location = args.output if args.output else 'filtered_files'


print(f"Reading from: {source_location}")
print(f"Filtering with word list: {filter_location}")
print(f"Outputting to: {output_location}")

# load word list
word_list = []
with open(filter_location, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for row in csv_reader:
        word_list.append(row[4].lower())

# drop header
word_list.pop(0)
# make a set to remove duplicates
word_set = set(word_list)
print(f"Number of filter words: {len(word_set)}")

def file_contains_intervention(file_location, word_set):
    with open(file_location, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line_set = set(line.lower().split())
            b = line_set.intersection(word_set)
            if len(b) > 0:
                # for some reason just returning the if gives fewer matches???
                return True
                
files = os.listdir(source_location)
print(f"Number of input files: {len(files)}")
keep_files = [file for file in files if file_contains_intervention(os.path.join(source_location, file), word_set)]
os.makedirs(output_location, exist_ok = True)
for file in keep_files:
    copy(os.path.join(source_location, file), output_location)
print(f"Number of filtered files: {len(keep_files)}")