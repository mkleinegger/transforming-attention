use candle_core::{Device, Result, Tensor};
use candle_nn::{LayerNorm, Module};

use crate::config::Config;

#[derive(Clone, Debug)]
pub struct NormalizationLayer(LayerNorm);

/// Normalize a vector as described in the paper using `LayerNorm`
/// from candle. We wrap around that implementation to set the values
/// as described in the paper and then call the LayerNorm implementation.
impl NormalizationLayer {
    pub fn new(config: &Config, device: &Device) -> Result<Self> {
        let omega = Tensor::full(1f32, config.d_model, device)?.contiguous()?;
        let beta = Tensor::full(0f32, config.d_model, device)?.contiguous()?;
        let layer = LayerNorm::new(omega, beta, 1e-5);
        Ok(Self(layer))
    }

    // Normalize `tensor` as descriced in the paper
    pub fn forward(&self, tensor: &Tensor) -> Result<Tensor> {
        self.0.forward(tensor)
    }
}
