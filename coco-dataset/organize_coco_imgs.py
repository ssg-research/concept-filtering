import pandas as pd
import shutil

sampled_df = pd.read_csv('coco-dataset/val2014_sampled50.csv', delimiter = '\t')
original_df = pd.read_csv('coco-dataset/test2014.csv', delimiter = '\t')

merged_df = pd.merge(sampled_df, original_df, on='title', how='inner')
# print(merged_df.head())

for filename in merged_df['filepath']:
    shutil.copy(f"coco-dataset/val2014/{filename}", f'coco-dataset/val2014_sample50/{filename}')

merged_df.to_csv("coco-dataset/val2014_sampled50_mapped.csv", index=False)