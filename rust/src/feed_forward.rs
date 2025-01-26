use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, Dropout, Linear, Module, VarBuilder};

use crate::config::Config;

pub struct FeedForwardBlock {
    dropout: Dropout,
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

        Ok(Self {
            layers,
            dropout: Dropout::new(config.feed_forward_dropout),
        })
    }

    // TODO: implement FeedForwardBlock
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let res_layer0 = self
            .dropout
            .forward(&self.layers[0].forward(xs)?.relu()?, true)?;

        self.layers[1].forward(&res_layer0)
    }
}
