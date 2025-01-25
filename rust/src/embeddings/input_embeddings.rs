use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};
use std::collections::HashMap;

use crate::config::Config;

pub struct InputEmbeddings {
    d_model: usize,
    embedding: Embedding,
    tensor_standardization: Tensor,
}

impl InputEmbeddings {
    pub fn new(vocab_size: usize, config: &Config, vb: VarBuilder) -> Result<Self> {
        let tensor_standardization = Tensor::new((config.d_model as f32).sqrt(), vb.device())?;
        let embedding = embedding(vocab_size, config.d_model, vb)?;

        Ok(Self {
            d_model: config.d_model,
            embedding,
            tensor_standardization,
        })
    }

    /// Get indices of shape (batch, seq_len) and add embedding vectors
    /// resulting in shape (batch, seq_len, n_model)
    pub fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        // let tensor = Tensor::from_slice(indices, (indices.len(),), device)?;

        self.embedding
            .forward(indices)?
            .broadcast_mul(&self.tensor_standardization)
    }
}
