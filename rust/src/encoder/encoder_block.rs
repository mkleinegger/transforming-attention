use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::{
    attention::multihead::MultiHeadAttentionBlock,
    config::Config,
    feed_forward::FeedForwardBlock,
    residual_connections::{ResidualConnection, SubLayer},
};

pub struct EncoderBlock {
    attention: MultiHeadAttentionBlock,
    feed_forward: FeedForwardBlock,
    residual_connections: Vec<ResidualConnection>,
}

impl EncoderBlock {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let mut residual_connections = Vec::with_capacity(2);
        for _ in 0..2 {
            residual_connections.push(ResidualConnection::new(config, vb.device())?)
        }

        Ok(Self {
            attention: MultiHeadAttentionBlock::new(vb.pp("attention"), config)?,
            feed_forward: FeedForwardBlock::new(vb.pp("feed_forward"), config)?,
            residual_connections,
        })
    }

    pub fn forward(&self, xs: &Tensor, src_mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let x = self.residual_connections[0].forward(
            xs,
            None,
            src_mask,
            SubLayer::Attention(&self.attention),
            train
        )?;
        self.residual_connections[1].forward(
            &x,
            None,
            None,
            SubLayer::FeedForward(&self.feed_forward),
            train
        )
    }
}
