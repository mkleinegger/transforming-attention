use crate::config::Config;
use crate::encoder::encoder_block::EncoderBlock;
use crate::normalization::NormalizationLayer;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct Encoder {
    layers: Vec<EncoderBlock>,
    norm: NormalizationLayer,
}

impl Encoder {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let norm = NormalizationLayer::new(config, vb.device())?;
        let layers = (0..config.n_encoders)
            .map(|index| EncoderBlock::new(vb.pp(format!("{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        // TODO: add logging using tracing crate
        // let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(Self { layers, norm })
    }

    pub fn forward(&self, mut xs: Tensor, src_mask: Option<&Tensor>, train: bool) -> Result<Tensor> {
        for blk in self.layers.iter() {
            xs = blk.forward(&xs, src_mask, train)?
        }
        self.norm.forward(&xs)
    }
}
