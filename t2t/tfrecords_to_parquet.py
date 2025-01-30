import tensorflow as tf
import polars as pl
import os
import glob

# Configurations
TFRECORD_FOLDER = "data/t2t"
PARQUET_FOLDER = "data/parquet"

# Ensure output directory exists
os.makedirs(PARQUET_FOLDER, exist_ok=True)

# Get all TFRecord files
tfrecord_files = glob.glob(os.path.join(TFRECORD_FOLDER, "translate_ende_wmt32k*"))

# Get already created Parquet files
created = [x.split('/')[-1] for x in glob.glob(os.path.join(PARQUET_FOLDER, "translate_ende_wmt32k*"))]

# Filter out already processed files
tfrecord_files = [x for x in tfrecord_files if x.split('/')[-1] not in created]

print(f"Files to process: {len(tfrecord_files)}")

# Function to parse TFRecord example
def parse_tfrecord(example_proto):
    feature_description = {
        'inputs': tf.io.VarLenFeature(tf.int64),
        'targets': tf.io.VarLenFeature(tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    return parsed_example

# Function to convert TFRecord to Polars DataFrame and save as Parquet
def convert_tfrecord_to_parquet(tfrecord_path):
    parquet_path = os.path.join(PARQUET_FOLDER, os.path.basename(tfrecord_path).replace(".tfrecord", ".parquet"))
    
    print(f"Processing: {tfrecord_path}")

    # Read TFRecord
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_data = [parse_tfrecord(record) for record in raw_dataset]

    # Convert to Polars DataFrame
    df = pl.DataFrame({
        "inputs": [item["inputs"].values.numpy().tolist() for item in parsed_data],
        "targets": [item["targets"].values.numpy().tolist() for item in parsed_data]
    })

    # Save as Parquet
    df.write_parquet(parquet_path)
    print(f"Saved {parquet_path}")

# Process files sequentially
for tfrecord_file in tfrecord_files:
    convert_tfrecord_to_parquet(tfrecord_file)

print("All files processed.")
