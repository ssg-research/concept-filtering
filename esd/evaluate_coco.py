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
    starting_idx = 219
    ending_idx = 200
    # if experiment == 'nemo':
        # starting_idx += 99
    model_dir = 'oct13_new_exp/'
    if experiment == 'vangogh':
        model_dir = 'new_vangogh_oct10/'
    with open('../coco-dataset/test2014_subset.csv', 'r') as f:
        csv_reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(csv_reader):
            # if i < starting_idx:
            #     continue
            # elif i >= (219 + 200):
            #     break
            # if i < ending_idx:
                # break
            column_value = row['title']
            execute_bash_command(f"python scripts/txt2img.py --prompt '{column_value}' --experiment {experiment} --num_steps {num_steps} --take {take} --loss_type {loss_type} --plms --testing_coco --model_dir {model_dir}")

if __name__ == '__main__':
    experiments = ['musk','pitt', 'dali', 'monet', 'gregrut']
    # experiments = ['grumpy', 'vangogh', 'r2d2']
    # experiments = ['nudity']
    num_steps = ['12','12', '12', '12', '12', '12']
    # num_steps = ['12', '12', '12']
    # num_steps = ['12']
    for i, exp in enumerate(experiments):
        loss_type = 'loss1356'
        # if exp == 'vangogh' or exp == 'nemo':
            # loss_type = 'loss13456'
        begin_test(exp, num_steps = num_steps[i], loss_type=loss_type)