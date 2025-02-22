import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
args = parser.parse_args()

# Specify the input and output file paths
input_file = args.input_file
output_file = args.input_file[:-4]+'_sampled200.txt'

# Read all lines from the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Sample 200 lines randomly
sampled_lines = random.sample(lines, 200)

# Write the sampled lines to the output file
with open(output_file, 'w') as f:
    for line in sampled_lines:
        f.write(line)
