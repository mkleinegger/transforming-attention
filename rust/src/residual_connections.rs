use crate::{
    attention::multihead::MultiHeadAttentionBlock, config::Config, feed_forward::FeedForwardBlock,
    normalization::NormalizationLayer,
};
use candle_core::{Device, Result, Tensor};
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
    pub fn new(config: &Config, device: &Device) -> Result<Self> {
        Ok(Self {
            dropout: Dropout::new(config.residual_dropout),
            norm: NormalizationLayer::new(config, device)?,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        xs_encoder: Option<&Tensor>,
        mask: Option<&Tensor>,
        sublayer: SubLayer,
        train: bool,
    ) -> Result<Tensor> {
        let sublayer_res = match sublayer {
            SubLayer::Attention(mha) => match xs_encoder {
                Some(xs_encoder) => mha.forward(xs, xs_encoder, xs_encoder, mask)?,
                None => mha.forward(xs, xs, xs, mask)?,
            },
            SubLayer::FeedForward(ff) => ff.forward(&xs)?,
        };
        // do dropout for all sublayers (self attention, cross attention and feed forward)
        let sublayer_res = self.dropout.forward(&sublayer_res, train)?;

        self.norm.forward(&xs.add(&sublayer_res)?)
    }
}
