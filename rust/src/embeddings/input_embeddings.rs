use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};
use std::collections::HashMap;

pub struct InputEmbeddings {
    d_model: usize,
    vocab_size: usize,
    embedding: Embedding,
}

impl InputEmbeddings {
    pub fn new(vocab_size: usize, d_model: usize, device: &Device) -> Result<Self> {
        let mut map = HashMap::new();
        map.insert(
            String::from("weight"),
            Tensor::randn(0f32, 1., (vocab_size, d_model), &device)?,
        );

        let var_builder = VarBuilder::from_tensors(map, candle_core::DType::F32, &device);
        let embedding = embedding(vocab_size, d_model, var_builder)?;

        Ok(Self {
            d_model,
            vocab_size,
            embedding,
        })
    }

    pub fn forward(&self, indices: &[u32], device: &Device) -> Result<Tensor> {
        let tensor = Tensor::from_slice(indices, (indices.len(),), device)?;
        let dmodel_sqrt = (self.d_model as f32).sqrt();
        let tensor_standardization = Tensor::new(dmodel_sqrt, device)?;

        self.embedding
            .forward(&tensor)?
            .broadcast_mul(&tensor_standardization)
    }
}
