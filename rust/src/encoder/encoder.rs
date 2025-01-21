// use ta::encoder::encoder_block::EncoderBlock;

use crate::config::Config;
use crate::encoder::encoder_block::EncoderBlock;
use crate::normalization::NormalizationLayer;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use tracing;

pub struct Encoder {
    layers: Vec<EncoderBlock>,
    norm: NormalizationLayer,
}

impl Encoder {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let norm = NormalizationLayer::new(config)?;
        let layers = (0..config.n_decoders)
            .map(|index| EncoderBlock::new(vb.pp(format!("encoder.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(Self { layers, norm })
    }

    pub fn forward(&self, mut xs: Tensor, src_mask: bool) -> Result<Tensor> {
        for blk in self.layers.iter() {
            xs = blk.forward(&xs, src_mask)?
        }
        self.norm.forward(&xs)
    }
}
