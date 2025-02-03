use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::attention::multihead::MultiHeadAttentionBlock;
use crate::config::Config;
use crate::feed_forward::FeedForwardBlock;
use crate::residual_connections::{self, ResidualConnection};

pub struct DecoderBlock {
    attention: MultiHeadAttentionBlock,
    cross_attention: MultiHeadAttentionBlock,
    feed_forward: FeedForwardBlock,
    residual_connections: Vec<ResidualConnection>,
}

impl DecoderBlock {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let mut residual_connections = Vec::with_capacity(2);
        for _ in 0..3 {
            residual_connections.push(ResidualConnection::new(config, vb.device())?);
        }

        Ok(Self {
            attention: MultiHeadAttentionBlock::new(vb.pp("attention"), config)?,
            cross_attention: MultiHeadAttentionBlock::new(vb.pp("cross_attention"), config)?,
            feed_forward: FeedForwardBlock::new(vb.pp("feed_forward"), config)?,
            residual_connections,
        })
    }

    pub fn forward(
        &self,
        mut xs: Tensor,
        encoder_output: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
        train: bool
    ) -> Result<Tensor> {
        xs = self.residual_connections[0].forward(
            &xs,
            None,
            tgt_mask,
            residual_connections::SubLayer::Attention(&self.attention),
            train
        )?;

        xs = self.residual_connections[1].forward(
            &xs,
            Some(encoder_output),
            src_mask,
            residual_connections::SubLayer::Attention(&self.cross_attention),
            train
        )?;

        xs = self.residual_connections[2].forward(
            &xs,
            None,
            None,
            residual_connections::SubLayer::FeedForward(&self.feed_forward),
            train
        )?;

        Ok(xs)
    }
}
