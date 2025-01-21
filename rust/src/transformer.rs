use candle_core::{Device, Result};
use candle_nn::VarBuilder;

use crate::config::Config;
// TODO: create Transformer

pub struct Transformer {
    pub device: Device,
}

impl Transformer {
    //
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            device: vb.device().clone(),
        })
    }

    pub fn encode() {}

    pub fn decode() {}

    pub fn project() {}
}
