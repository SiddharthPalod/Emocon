import os
import pandas as pd

# Directory containing the CSV files
csv_dir = ''

# List to hold dataframes
dataframes = []

# Iterate over all files in the directory
for filename in os.listdir(csv_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_dir, filename)
        # Read the CSV file, skipping the first row
        df = pd.read_csv(file_path, skiprows=1)
        dataframes.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('/d:/New folder (4)/SidFiles/Coding/Ml/Emotion detector/combined.csv', index=False)