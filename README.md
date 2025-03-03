# **Transformer Implementations**

This repository contains Transformer model implementations in **Rust and PyTorch**, based on the paper **[Attention is All You Need](https://arxiv.org/pdf/1706.03762)**. Furthermore, we provide code to reproduce results using the original **[Tensor2Tensor (T2T)](https://github.com/tensorflow/tensor2tensor.git)** implementation. A report can be found under https://mkleinegger.github.io/transforming-attention/docs/report.pdf

## **Dataset**
To ensure consistent sentence embeddings across all implementations, we provide **tokenized translation datasets** in **Parquet format** and a **vocabulary file** (`vocab.ende`) in the `data/` directory. The dataset was generated using **[Tensor2Tensor (T2T)](https://github.com/tensorflow/tensor2tensor.git)**, the original implementation of the **Attention is All You Need** paper, with **subword tokenization** applied to the `translate_ende_wmt32k` dataset.

Each dataset file contains sentence pairs, where:
- **`inputs` (English sentence tokens)**  
- **`targets` (German translation tokens)**  

Example structure of the dataset:  
```
shape: (45_782, 2)
┌───────────────────┬────────────────────┐
│ inputs            ┆ targets            │
│ ---               ┆ ---                │
│ list[i64]         ┆ list[i64]          │
╞═══════════════════╪════════════════════╡
│ [5374, 8907, … 1] ┆ [2606, 12727, … 1] │
│ [29, 379, … 1]    ┆ [1096, 10, … 1]    │
│ [124, 6618, … 1]  ┆ [111, 16146, … 1]  │
│ [316, 25, … 1]    ┆ [806, 103, … 1]    │
│ [75, 8664, … 1]   ┆ [168, 4021, … 1]   │
│ …                 ┆ …                  │
└───────────────────┴────────────────────┘
```

## **Structure**
- `rust/` - Transformer implementation in Rust.
- `pytorch/` - Transformer implementation in PyTorch.
- `jax/` - Transformer implementation in JAX.
- `t2t/` - Transformer implementation from Tensor2Tensor.
- `data/` - Tokenized datasets and vocabulary file.
- `examples/` - Example scripts.

## **Running**
Use `create_env.sh` to create an environment. After activating it, the code should run with just `python script.py`. If you want to use multi-gpu just run it with `torchrun torchrun --nproc-per-node=num_nodes train.py` and `torchrun torchrun --nproc-per-node=num_nodes predict.py`

After training and predicting just use `bleu.py` to compute the result. Before running just check that you load the right checkpoints for each file, before each step.

## **References**
- **[Attention is All You Need](https://arxiv.org/pdf/1706.03762)**
- **[Tensor2Tensor (T2T)](https://github.com/tensorflow/tensor2tensor.git)**
