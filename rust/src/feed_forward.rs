use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::Config;

pub struct FeedForwardBlock {
    layers: Vec<Linear>,
}

impl FeedForwardBlock {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        // create 2 linear layers according to the paper:
        // W1: [512 * 2048] and [512]
        // W2: [2048 * 512] and [2048]

        let layers = vec![
            linear(config.d_model, config.d_feed_forward, vb.pp("0"))?,
            linear(config.d_feed_forward, config.d_model, vb.pp("1"))?,
        ];

        Ok(Self { layers })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let res_layer0 = &self.layers[0].forward(xs)?.relu()?;

        self.layers[1].forward(&res_layer0)
    }
}
