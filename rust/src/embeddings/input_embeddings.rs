use candle_core::{Device, Result};
use candle_nn::Embedding;

pub struct InputEmbeddings {
    d_model: usize,
    vocab_size: usize,
    embedding: Embedding,
}

impl InputEmbeddings {
    pub fn new(vocab_size: usize, d_model: usize, device: &Device) -> Result<Self> {}
}
