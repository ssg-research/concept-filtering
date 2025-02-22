from cleanfid import fid
import argparse

parser = argparse.ArgumentParser(description='Calculating the FID scores')
parser.add_argument('--output_folder', required=True, help='The folder containing the generated images')
args = parser.parse_args()
score = fid.compute_fid(args.output_folder, 'coco-baseline/samples',  mode="clean")
print(f'FID Score: {score}')