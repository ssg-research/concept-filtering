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


def begin_test(experiment = None, num_steps = None, take = 'take1', loss_type='loss1356', seed = 42, ckpt = '/u1/test/selective-amnesia/sd/nudity.ckpt'):
    # starting_idx = 219+5
    # if experiment == 'nemo':
        # starting_idx += 99
    with open('../../stable-diffusion/assets/eval-prompts/coco-dataset/test2014_subset.csv', 'r') as f:
        csv_reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(csv_reader):
            
            column_value = row['title']
    
            print(column_value)
            execute_bash_command(f"python scripts/txt2img.py --prompt '{column_value}' --plms --ckpt {ckpt} --seed {seed} --experiment {experiment}_coco")
            # with open('nudity_prompts_used_i2p.log', 'a') as f:
            #     for i in range(6):
            #         f.write(f'{column_value}\n')

if __name__ == '__main__':
    # experiments = ['nemo', 'marvel','grumpy', 'snoopy', 'r2d2', 'swift', 'musk', 'violence']
    # experiments = ['violence']
    experiments = ['nudity', 'pitt', 'jolie']
    experiments += ['nemo', 'marvel','grumpy', 'snoopy', 'r2d2', 'swift', 'musk', 'violence']
    seeds = [42,43,44]
    # # num_steps = ['12', '12','8', '12', '6',
    # num_steps = ['12']
    # for i, exp in enumerate(experiments):
    #     loss_type = 'loss1356'
    #     if exp == 'vangogh' or exp == 'nemo':
    #         loss_type = 'loss13456'
    for experiment in experiments:
        if experiment == 'nudity':
            ckpt = '/u1/test/selective-amnesia/sd/nudity.ckpt'
        elif experiment == 'jolie':
            ckpt = '/u1/test/selective-amnesia/angelina_jolie_middle_aged_woman.ckpt'
        elif experiment == 'pitt':
            ckpt = '/u1/test/selective-amnesia/brad_pitt_middle_aged_man.ckpt'
        elif experiment == 'nemo':
            ckpt = '/u1/test/selective-amnesia/sd/logs/2024-01-31T05-04-46_forget_nemo/checkpoints/last.ckpt'
        elif experiment == 'marvel':
            ckpt = '/u1/test/selective-amnesia/sd/logs/2024-01-31T06-35-46_forget_marvel/checkpoints/last.ckpt'
        elif experiment == 'grumpy':
            ckpt = '/u1/test/selective-amnesia/sd/logs/2024-01-31T03-35-52_forget_grumpy/checkpoints/last.ckpt'
        elif experiment == 'snoopy':
            ckpt = '/u1/test/selective-amnesia/sd/logs/2024-01-31T08-03-56_forget_snoopy/checkpoints/last.ckpt'
        elif experiment == 'r2d2':
            ckpt = '/u1/test/selective-amnesia/sd/logs/2024-01-31T08-05-44_forget_r2d2/checkpoints/last.ckpt'
        elif experiment == 'swift':
            ckpt = '/u1/test/selective-amnesia/sd/logs/2024-01-31T02-07-07_forget_swift/checkpoints/last.ckpt'
        elif experiment == 'musk':
            ckpt = '/u1/test/selective-amnesia/sd/logs/2024-01-31T00-38-42_forget_musk/checkpoints/last.ckpt'
        elif experiment == 'violence':
            ckpt = '/u1/test/selective-amnesia/sd/logs/2024-02-03T00-32-36_forget_violence/checkpoints/last.ckpt'
        for seed in seeds:
            begin_test(experiment=experiment, seed = seed, ckpt = ckpt)