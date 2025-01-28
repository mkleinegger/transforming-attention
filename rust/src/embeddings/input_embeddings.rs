use candle_core::{Module, Result, Tensor};
use candle_nn::{embedding, Dropout, Embedding, VarBuilder};

use crate::config::Config;

pub struct InputEmbeddings {
    // _d_model: usize,
    embedding: Embedding,
    tensor_standardization: Tensor,
    dropout: Dropout,
}

impl InputEmbeddings {
    pub fn new(vocab_size: usize, config: &Config, vb: VarBuilder) -> Result<Self> {
        let tensor_standardization = Tensor::new((config.d_model as f32).sqrt(), vb.device())?;
        let embedding = embedding(vocab_size, config.d_model, vb)?;

        Ok(Self {
            // _d_model: config.d_model,
            embedding,
            tensor_standardization,
            dropout: Dropout::new(config.embedding_dropout),
        })
    }

    /// Get indices of shape (batch, seq_len) and add embedding vectors
    /// resulting in shape (batch, seq_len, n_model)
    pub fn forward(&self, indices: &Tensor, train: bool) -> Result<Tensor> {
        self.dropout.forward(
            self.embedding
                .forward(indices)?
                .broadcast_mul(&self.tensor_standardization)?
                .as_ref(),
            train,
        )
    }
}
