import os
import argparse
import statistics

parser = argparse.ArgumentParser(description='Parsing for the clip scores')
parser.add_argument('--filename', required=True, help='The name of the logfile to parse')
args = parser.parse_args()

scores = []
accuracies = []
with open(args.filename, 'r') as f:
    for line in f:
        line_split = line.split(':')
        if line_split[0].strip() == 'CLIP score is':
            scores.append(float(line_split[-1].strip()))
        elif line_split[0].strip() == 'CLIP accuracy':
            accuracies.append(float(line_split[-1].strip().split(' ')[0].strip()[2:]))

print(f'Mean clip score is: {statistics.fmean(scores)}')
print(f'Mean clip accuracy is: {statistics.fmean(accuracies)}')
