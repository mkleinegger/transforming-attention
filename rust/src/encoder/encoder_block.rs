use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::{attention::multihead::MultiHeadAttention, config::Config};

pub struct EncoderBlock {
    attention: MultiHeadAttention,
}

impl EncoderBlock {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        // let mut residual_connections = Vec::with_capacity(2);
        for conn in 0..2 {
            // residual_connections.push()
        }

        Ok(Self {
            attention: MultiHeadAttention::new(vb.pp("attention"), config)?,
        })
    }

    pub fn forward(&self, xs: &Tensor, src_mask: bool) -> Result<Tensor> {
        // let x =
        Ok(xs.clone())
    }
}
