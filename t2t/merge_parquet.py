import polars as pl
import glob
import os

# Configurations
PARQUET_FOLDER = "data/parquet"
OUTPUT_FILE = "data/dataset.parquet"

# Get all Parquet files in the folder
parquet_files = glob.glob(os.path.join(PARQUET_FOLDER, "*"))

if not parquet_files:
    print("No Parquet files found to merge.")
    exit()

# Read and concatenate all Parquet files
df_list = [pl.read_parquet(file) for file in parquet_files]
merged_df = pl.concat(df_list, how="vertical")

# Save merged DataFrame
merged_df.write_parquet(OUTPUT_FILE)
print(f"Saved merged file: {OUTPUT_FILE}")
