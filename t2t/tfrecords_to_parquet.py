# this script converts the translate_ende_wmt32k dataset from 
# sharded TFRecord format into a parquet file 
# for compatibility and software independence. 

import tensorflow as tf
import os
import glob
import polars as pl


T2T_DATASET_FOLDER = 't2t/results/dataset'
PARQUET_OUTPUT_FILE = 'data/translate_ende_wmt32k.parquet'

# get all TFRecord files
tfrecord_files = glob.glob(os.path.join(T2T_DATASET_FOLDER, "translate_ende_wmt32k-train-*"))
tfrecord_files = tfrecord_files[:5]

# define the feature description dictionary
feature_description = {
    'inputs': tf.io.VarLenFeature(tf.int64),
    'targets': tf.io.VarLenFeature(tf.int64),
}

def parse_tfrecord(proto):
    """Parse a single TFRecord."""
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    inputs = tf.sparse.to_dense(parsed_features['inputs']).numpy()
    targets = tf.sparse.to_dense(parsed_features['targets']).numpy()
    return inputs.tolist(), targets.tolist()


all_inputs, all_targets = [], []

for file in tfrecord_files:
    dataset = tf.data.TFRecordDataset(file).map(lambda x: tf.py_function(parse_tfrecord, [x], [tf.int64, tf.int64]))
    for inputs, targets in dataset:
        all_inputs.append(inputs.numpy())
        all_targets.append(targets.numpy())

# convert to polars dataframe
df = pl.DataFrame({
    "inputs": all_inputs,
    "targets": all_targets
})

# Save as parquet
df.write_parquet(PARQUET_OUTPUT_FILE)