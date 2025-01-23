# **Transformer Implementations**

This repository contains Transformer model implementations in **Rust, PyTorch, and JAX**, based on the paper **[Attention is All You Need](https://arxiv.org/pdf/1706.03762)**.

## **Dataset**
o ensure consistent sentence embeddings across all implementations, we provide **tokenized translation datasets** in **Parquet format** and a **vocabulary file** (`vocab.ende`) in the `data/` directory. The dataset was generated using **[Tensor2Tensor (T2T)](https://github.com/tensorflow/tensor2tensor.git)**, the original implementation of the **Attention is All You Need** paper, with **subword tokenization** applied to the `translate_ende_wmt32k` dataset.

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

## **References**
- **[Attention is All You Need](https://arxiv.org/pdf/1706.03762)**
- **[Tensor2Tensor (T2T)](https://github.com/tensorflow/tensor2tensor.git)**