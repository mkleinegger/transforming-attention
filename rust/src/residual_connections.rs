use crate::{
    attention::multihead::MultiHeadAttentionBlock, config::Config, feed_forward::FeedForwardBlock,
    normalization::NormalizationLayer,
};
use candle_core::{Result, Tensor};
use candle_nn::Dropout;

pub struct ResidualConnection {
    dropout: Dropout,
    norm: NormalizationLayer,
}

pub enum SubLayer<'a> {
    Attention(&'a MultiHeadAttentionBlock),
    FeedForward(&'a FeedForwardBlock),
}

impl ResidualConnection {
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            dropout: Dropout::new(config.residual_dropout),
            norm: NormalizationLayer::new(config)?,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        xs_encoder: Option<&Tensor>,
        mask: bool,
        sublayer: SubLayer,
    ) -> Result<Tensor> {
        let sublayer_res = match sublayer {
            SubLayer::Attention(mha) => match xs_encoder {
                Some(xs_encoder) => mha.forward(&xs, &xs_encoder, &xs_encoder, mask)?,
                None => mha.forward(&xs, &xs, &xs, mask)?,
            },
            SubLayer::FeedForward(ff) => ff.forward(&xs)?,
        };

        self.norm.forward(&xs.add(&sublayer_res)?)
    }
}
