# Authors: Anudeep Das, Vasisht Duddu, Rui Zhang, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv 
import subprocess

def execute_bash_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Print the command output and error (if any)
    print("Command output:")
    print(stdout.decode())
    print("Command error:")
    print(stderr.decode())
    print("Return code:", returncode)


def begin_test(experiment = None, num_steps = None, take = 'take1', loss_type='loss1356'):
    model_dir = 'oct13_new_exp/'
    with open('assets/eval_prompts/coco-dataset/test2014_subset.csv', 'r') as f:
        csv_reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(csv_reader):
            column_value = row['title']
            execute_bash_command(f"python scripts/txt2img.py --prompt '{column_value}' --experiment {experiment} --num_steps {num_steps} --take {take} --loss_type {loss_type} --plms --testing_coco --model_dir {model_dir}")

if __name__ == '__main__':
    experiments = ['nudity', 'violence', 'hateful', 'disturbing', 'marvel', 'grumpy', 'nemo', 'snoopy', 'r2d2', 'jolie', 'swift', 'musk','pitt']
    num_steps = ['12' for _ in range(len(experiments))]
    for i, exp in enumerate(experiments):
        loss_type = 'loss1356'
        begin_test(exp, num_steps = num_steps[i], loss_type=loss_type)