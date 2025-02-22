import pandas as pd
import random

# Read the original CSV file into a DataFrame
original_df = pd.read_csv('coco-dataset/val_sampled.csv', delimiter = '\t')

# Skip the first 200 rows
# skipped_df = original_df.iloc[427:]

# Sample 200 rows from the DataFrame
# Make sure to use a random seed for reproducibility if needed
sampled_df = original_df.sample(n=50, random_state=42)



# Sample 200 rows from the DataFrame
# sampled_df = original_df.sample(n=50, random_state=42)  # Use a specific random seed for reproducibility

# column_to_save = sampled_df['title']


# Specify the filename for the CSV file
output_file = 'coco-dataset/val_sampled50.csv'

# Save the column data to a CSV file without the header
# column_to_save.to_csv(output_file, index=False, header=False)
sampled_df.to_csv(output_file, index=False)
